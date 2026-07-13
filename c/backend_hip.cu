#include "backend_hip.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            std::fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

struct ColiHipTensor {
    void *weights;       /* device pointer from hipHostGetDevicePointer */
    void *host_ptr;      /* original host pointer (mmap'd or malloc'd) */
    float *scales;        /* device pointer */
    void *scales_host;    /* original host pointer */
    size_t weight_bytes;
    int fmt, I, O, device;
    int is_mmap;         /* 1 = registered mmap memory, 0 = malloc'd buffer */
    int tracked;
};

typedef struct {
    int device;
    float *x, *y;
    size_t x_cap, y_cap;
    size_t tensor_count, tensor_bytes;
} DeviceContext;

static DeviceContext g_ctx[COLI_HIP_MAX_DEVICES];
static int g_nctx;

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    HIP_CHECK(hipSetDevice(ctx->device), "select device");
    return 1;
}

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        return static_cast<float>(((i & 1) ? (v >> 4) : (v & 15)) - 8);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int /*S*/, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) { HIP_CHECK(hipFree(*ptr), "scratch free"); *ptr = nullptr; }
    *cap = 0;
    HIP_CHECK(hipMalloc(ptr, bytes), "scratch allocation");
    *cap = bytes;
    return 1;
}

extern "C" int coli_hip_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_HIP_MAX_DEVICES) return 0;
    HIP_CHECK(hipGetDeviceCount(&available), "device discovery");
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[HIP] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[HIP] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, device), "device properties");
        g_nctx++;
        std::fprintf(stderr, "[HIP] device %d: %s, %.1f GB VRAM, ip_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_hip_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) { HIP_CHECK(hipFree(ctx->x), "x free"); ctx->x = nullptr; }
        if (ctx->y) { HIP_CHECK(hipFree(ctx->y), "y free"); ctx->y = nullptr; }
        ctx->x_cap = ctx->y_cap = 0;
    }
    g_nctx = 0;
}

extern "C" int coli_hip_device_count(void) { return g_nctx; }

extern "C" int coli_hip_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_hip_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    HIP_CHECK(hipMemGetInfo(free_bytes, total_bytes), "memory info");
    return 1;
}

extern "C" void coli_hip_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

/**
 * Upload a tensor to the HIP device.
 *
 * On UMA (Strix Halo / gfx1151) this uses hipHostRegisterMapped + hipHostGetDevicePointer
 * to give the GPU a direct pointer to the host memory — no copy.
 * The host memory is typically mmap'd from a file, so the GPU walks the file's pages
 * through the device pointer, faulting them in on first access.
 *
 * For non-UMA (discrete GPU), this falls back to hipMalloc + hipMemcpy.
 * The caller can detect UMA mode by checking if host_ptr == device_ptr after registration.
 */
extern "C" int coli_hip_tensor_upload(ColiHipTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiHipTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiHipTensor *t = static_cast<ColiHipTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device;
    t->weight_bytes = rb * (size_t)O;

    /* Register the host pointer so the GPU can walk it.
     * hipHostRegisterMapped tells the driver this is pageable memory
     * that the GPU may access — on UMA devices this means the GPU
     * can fault in pages from the mmap'd file.
     * On discrete GPUs this pins pages in the IOMMU for DMA access.
     */
    HIP_CHECK(hipHostRegister((void *)weights, t->weight_bytes, hipHostRegisterMapped),
              "register host memory");
    t->host_ptr = (void *)weights;
    t->is_mmap = 1;

    /* Get the device pointer that the GPU uses to access this memory.
     * On UMA (Strix Halo), this is typically the same as the host pointer
     * but always use the API call for correctness across devices.
     * On discrete GPUs, this is the IOMMU-mapped address. */
    HIP_CHECK(hipHostGetDevicePointer((void **)&t->weights, (void *)weights, 0),
              "get device pointer");

    /* Scales: same treatment. They are small (one per output row) so
     * the copy cost is negligible, but we use the same pattern for consistency.
     * For very small scale buffers, a single hipMalloc + hipMemcpy is also fine. */
    if (fmt) {
        size_t scales_bytes = (size_t)O * sizeof(float);
        HIP_CHECK(hipHostRegister((void *)(uintptr_t)scales, scales_bytes, hipHostRegisterMapped),
                  "register scale memory");
        t->scales_host = (void *)(uintptr_t)scales;
        HIP_CHECK(hipHostGetDevicePointer((void **)&t->scales, (void *)(uintptr_t)scales, 0),
                  "get scale device pointer");
    } else {
        t->scales_host = nullptr;
    }

    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_hip_matmul(ColiHipTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_hip_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiHipTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;

    /* Copy input to device scratch, run kernel, copy output back. */
    HIP_CHECK(hipMemcpy(ctx->x, x, xb, hipMemcpyHostToDevice), "input upload");
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    HIP_CHECK(hipGetLastError(), "kernel launch");
    HIP_CHECK(hipMemcpy(y, ctx->y, yb, hipMemcpyDeviceToHost), "output download");
    return 1;
}

extern "C" void coli_hip_tensor_free(ColiHipTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) {
        if (tensor->is_mmap) {
            /* Was registered with hipHostRegisterMapped — unregister instead of free */
            hipHostUnregister(tensor->host_ptr);
        } else {
            HIP_CHECK(hipFree(tensor->weights), "weights free");
        }
        tensor->weights = nullptr;
    }
    if (tensor->scales) {
        if (tensor->scales_host) {
            /* Scales were also registered — unregister */
            hipHostUnregister(tensor->scales_host);
            tensor->scales_host = nullptr;
        } else {
            HIP_CHECK(hipFree(tensor->scales), "scales free");
        }
        tensor->scales = nullptr;
    }
    std::free(tensor);
}

extern "C" size_t coli_hip_tensor_bytes(const ColiHipTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_hip_tensor_device(const ColiHipTensor *tensor) {
    return tensor ? tensor->device : -1;
}
