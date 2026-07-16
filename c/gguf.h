/* Minimal GGUF indexer — header-only.
 * Legge solo gli indici dei tensori (non i valori), mappa i dati nel
 * processo tramite mmap per accesso diretto GPU (HIP/UMA: hipHostRegisterMapped).
 *
 * GGUF v3 layout:
 *   [0:4]   "GGUF" magic
 *   [4:8]   version (uint32) → 3
 *   [8:16]  tensor_count (uint64)
 *   [16:24] kv_count (uint64) → SKIP
 *   [24+...] KV pairs → SKIP
 *   tensor_index[]: name(len+data), dtype(uint32), ndim(uint32), shape(ndim×uint64), offset(uint64)
 */
#ifndef GGUF_H
#define GGUF_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <math.h>
#include "compat.h"

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_BF16 = 3,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_NVFP4 = 40,  /* OCP FP4 E2M1 */
    GGML_TYPE_NVFP8 = 41,  /* OCP FP8 E4M3 */
};

static inline int64_t ggml_type_bytes(int t) {
    switch (t) {
    case GGML_TYPE_F32: return 4;
    case GGML_TYPE_F16: return 2;
    case GGML_TYPE_BF16: return 2;
    case GGML_TYPE_Q4_0: return 0;
    case GGML_TYPE_NVFP4: return 0;
    case GGML_TYPE_NVFP8: return 0;
    default: return -1;
    }
}

/* GGUF v3 layout with pre-quantized weights:
 *   [0:4]   "GGUF" magic
 *   [4:8]   version (uint32) → 3
 *   [8:16]  tensor_count (uint64)
 *   [16:24] kv_count (uint64) → SKIP
 *   [24+...] KV pairs → SKIP
 *   tensor_index[]: name(len+data), dtype(uint32), ndim(uint32), shape(ndim×uint64), offset(uint64)
 *   data_offset[]: (uint64 per tensor, aligned to 8 bytes)
 *   tensor_data[]: data_bytes (quantized), scales_bytes (F32 per-row, optional)
 */

typedef struct {
    char   *name;
    int     fd;
    int64_t off;           /* data offset in file */
    int64_t nbytes;        /* data size (for F32/F16) */
    int     dtype;
    int64_t shape[8];
    int     ndim;
    void    *mmap_ptr;
    /* Pre-quantized tensors: scales */
    float   *scales;        /* per-row scales (F32), NULL for F32/F16 */
    int64_t n_scales;       /* number of scales */
} gguf_tensor;

typedef struct {
    gguf_tensor *t;
    int          n, cap;
    char        *path;
    int          fd;
    int         *hidx;
    int          hcap;
    uint32_t     version;
} gguf_ctx;
#define GGUF_MAX_TENSORS 16384

static inline int64_t gguf_pread_u64(int fd, int64_t off, uint64_t *out) {
    return (pread(fd, out, 8, off) == 8) ? 0 : -1;
}
static inline int64_t gguf_pread_u32(int fd, int64_t off, uint32_t *out) {
    return (pread(fd, out, 4, off) == 4) ? 0 : -1;
}
static inline int64_t gguf_pread_str(int fd, int64_t off, char **out) {
    uint64_t len;
    if (gguf_pread_u64(fd, off, &len) < 0) return -1;
    char *s = malloc(len + 1);
    if (len && pread(fd, s, len, off + 8) != (ssize_t)len) { free(s); return -1; }
    s[len] = 0;
    *out = s;
    return 0;
}

static inline uint64_t gguf_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

static int gguf_init(gguf_ctx *ctx, const char *path) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->path = strdup(path);
    /* If path is a directory, look for a .gguf file inside */
    char resolved[4096];
    const char *fp = path;
    struct stat st;
    if (stat(path, &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR) {
        DIR *d = opendir(path);
        if (d) {
            struct dirent *ent;
            while ((ent = readdir(d)) != NULL) {
                if (strstr(ent->d_name, ".gguf") && !strstr(ent->d_name, "_")) {
                    snprintf(resolved, sizeof(resolved), "%s/%s", path, ent->d_name);
                    ctx->path = strdup(resolved);
                    fp = resolved;
                    break;
                }
            }
            closedir(d);
        }
    }
    ctx->fd = open(fp, COMPAT_O_RDONLY);
    if (ctx->fd < 0) { perror(fp); return 0; }
    fprintf(stderr, "[GGUF] opening %s fd=%d\n", path, ctx->fd);
    ctx->cap = GGUF_MAX_TENSORS;
    ctx->t = calloc(ctx->cap, sizeof(gguf_tensor));
    if (!ctx->t) { close(ctx->fd); return 0; }

    char magic[4];
    if (pread(ctx->fd, magic, 4, 0) != 4 || memcmp(magic, "GGUF", 4) != 0) goto error;
    uint32_t version;
    if (gguf_pread_u32(ctx->fd, 4, &version) < 0) goto error;
    ctx->version = version;
    if (ctx->version != 3) goto error;
    uint64_t tensor_count, kv_count;
    if (gguf_pread_u64(ctx->fd, 8, &tensor_count) < 0) goto error;
    if (gguf_pread_u64(ctx->fd, 16, &kv_count) < 0) goto error;

    int64_t pos = 24;
    fprintf(stderr, "[GGUF] kv_count=%lu tensor_count=%lu\n", (unsigned long)kv_count, (unsigned long)tensor_count);
    for (uint64_t i = 0; i < kv_count; i++) {
        int64_t kv_start = pos;
        char *kv_name;
        if (gguf_pread_str(ctx->fd, pos, &kv_name) < 0) { fprintf(stderr,"[GGUF] failed to read kv name at pos %lld\n", (long long)pos); goto error; }
        int64_t nlen = (int64_t)strlen(kv_name);
        uint32_t kv_type;
        if (gguf_pread_u32(ctx->fd, pos + 8 + nlen, &kv_type) < 0) { fprintf(stderr,"[GGUF] failed to read kv type at pos %lld\n", (long long)(pos + 8 + nlen)); free(kv_name); goto error; }
        /* GGUF v3 spec: type 0=BOOL(1B), 1=ARRAY, 2=STRING(Q+len), 3=U8(1B),
         * 4=U16(2B), 5=U32(4B), 6=I32(4B), 7=F32(4B), 8+ = BOOL_ARRAY/STRING_ARRAY(Q+len) */
        int64_t kv_span;
        if (kv_type == 0) { kv_span = 8 + nlen + 1 + 4; pos += kv_span; }  /* BOOL: 1 byte value */
        else if (kv_type == 2) {
            /* STRING: read value length from file */
            uint64_t vlen;
            if (gguf_pread_u64(ctx->fd, pos + 8 + nlen + 4, &vlen) < 0) { free(kv_name); goto error; }
            kv_span = 8 + nlen + 4 + 8 + vlen; pos += kv_span;  /* key(Q+len) + type(4) + value(Q+vlen) */
            fprintf(stderr, "[GGUF] kv[%lu]=%s type=%u vlen=%lu kv_span=%lld new_pos=%lld\n",
                    i, kv_name, kv_type, (unsigned long)vlen, (long long)kv_span, (long long)pos);
        }
        else if (kv_type == 5) { kv_span = 8 + nlen + 4 + 4; pos += kv_span; }  /* U32: 4 byte value */
        else if (kv_type == 7) { kv_span = 8 + nlen + 4 + 4; pos += kv_span; }  /* F32: 4 byte value */
        else if (kv_type >= 8) { uint64_t slen; gguf_pread_u64(ctx->fd, pos + 8 + nlen, &slen); kv_span = 8 + nlen + 8 + slen + 4; pos += kv_span; }  /* ARRAY/STRING_ARRAY */
        else { kv_span = 8 + nlen + 8 + 4; pos += kv_span; }  /* default: 8 byte value */
        fprintf(stderr, "[GGUF] kv[%lu]=%s type=%u pos_start=%lld pos_end=%lld\n", i, kv_name, kv_type, (long long)kv_start, (long long)pos);
        free(kv_name);
    }

    /* Read tensor info: name(Q+len), type(U32), ndim(U32), shape(ndim*U64) */
    fprintf(stderr, "[GGUF] reading tensor info at pos=%lld\n", (long long)pos);
    for (uint64_t ti = 0; ti < tensor_count && ctx->n < ctx->cap; ti++) {
        char *name;
        if (gguf_pread_str(ctx->fd, pos, &name) < 0) { fprintf(stderr,"[GGUF] failed to read tensor name at pos=%lld\n", (long long)pos); goto error; }
        int64_t nlen = (int64_t)strlen(name);
        uint32_t dtype;
        if (gguf_pread_u32(ctx->fd, pos + 8 + nlen, &dtype) < 0) { free(name); fprintf(stderr,"[GGUF] failed to read tensor dtype at pos=%lld\n", (long long)(pos + 8 + nlen)); goto error; }
        uint32_t ndim;
        if (gguf_pread_u32(ctx->fd, pos + 8 + nlen + 4, &ndim) < 0) { free(name); goto error; }
        int64_t shp = 0;
        for (int d = 0; d < (int)ndim && d < 8; d++) {
            uint64_t sv;
            if (gguf_pread_u64(ctx->fd, pos + 8 + nlen + 8 + d*8, &sv) < 0) { free(name); goto error; }
            ctx->t[ctx->n].shape[d] = (int64_t)sv;
            shp += 8;
        }
        ctx->t[ctx->n].dtype = (int)dtype;
        ctx->t[ctx->n].ndim = (int)ndim;
        ctx->t[ctx->n].scales = NULL;
        ctx->t[ctx->n].n_scales = 0;
        ctx->t[ctx->n].name = name;
        ctx->t[ctx->n].fd = ctx->fd;
        ctx->t[ctx->n].mmap_ptr = NULL;
        fprintf(stderr, "[GGUF] tensor[%d]=%s dtype=%d ndim=%d\n", ctx->n, name, dtype, ndim);
        ctx->n++;
        pos += 8 + nlen + 4 + 4 + shp;  /* name(Q+len) + dtype(4) + ndim(4) + shape(ndim*8) */
    }

    /* GGUF v3: read tensor data offsets from separate section (contiguous uint64, no padding) */
    fprintf(stderr, "[GGUF] reading tensor data offsets at pos=%lld\n", (long long)pos);
    for (uint64_t i = 0; i < tensor_count && i < ctx->n; i++) {
        uint64_t off;
        if (gguf_pread_u64(ctx->fd, pos, &off) < 0) { fprintf(stderr,"[GGUF] failed to read offset[%lu] at pos=%lld\n", i, (long long)pos); goto error; }
        ctx->t[i].off = (int64_t)off;
        fprintf(stderr, "[GGUF] tensor[%lu] offset=%lu\n", i, (unsigned long)off);
        int64_t eb = ggml_type_bytes(ctx->t[i].dtype);
        if (eb > 0) {
            int64_t numel = 1;
            for (int d = 0; d < ctx->t[i].ndim; d++) numel *= ctx->t[i].shape[d];
            ctx->t[i].nbytes = numel * eb;
        }
        pos += 8;  /* next offset — offsets section is contiguous, no padding */
    }
    fprintf(stderr, "[GGUF] tensor data offsets done, pos=%lld\n", (long long)pos);

    fprintf(stderr, "[GGUF] n=%d, computing hcap...\n", ctx->n);
    ctx->hcap = 1; while (ctx->hcap < ctx->n * 2) ctx->hcap <<= 1;
    fprintf(stderr, "[GGUF] hcap=%d\n", ctx->hcap);
    ctx->hidx = calloc(ctx->hcap, sizeof(int));
    if (!ctx->hidx) goto error;
    fprintf(stderr, "[GGUF] building hash table...\n");
    memset(ctx->hidx, -1, ctx->hcap * sizeof(int));
    for (int i = 0; i < ctx->n; i++) {
        if (!ctx->t[i].name) { fprintf(stderr,"[GGUF] tensor[%d] has NULL name\n", i); goto error; }
        uint64_t h = gguf_hash(ctx->t[i].name) & (ctx->hcap - 1);
        while (ctx->hidx[h] >= 0) h = (h + 1) & (ctx->hcap - 1);
        ctx->hidx[h] = i;
        if (i % 100 == 0) fprintf(stderr, "[GGUF] hash %d/%d\n", i, ctx->n);
    }
    fprintf(stderr, "[GGUF] hash table built, n=%d\n", ctx->n);
    lseek(ctx->fd, 0, SEEK_SET);
    fprintf(stderr, "[GGUF] gguf_init done\n");
    return 1;

error:
    fprintf(stderr, "gguf_init: parse error\n");
    free(ctx->t); ctx->t = NULL;
    free(ctx->hidx); ctx->hidx = NULL;
    free(ctx->path); ctx->path = NULL;
    close(ctx->fd);
    return 0;
}

static gguf_tensor *gguf_find(gguf_ctx *ctx, const char *name) {
    if (!ctx->hidx) return NULL;
    uint64_t h = gguf_hash(name) & (ctx->hcap - 1);
    while (ctx->hidx[h] >= 0) {
        gguf_tensor *t = &ctx->t[ctx->hidx[h]];
        if (!strcmp(t->name, name)) return t;
        h = (h + 1) & (ctx->hcap - 1);
    }
    return NULL;
}

static int gguf_has(gguf_ctx *ctx, const char *name) {
    return gguf_find(ctx, name) != NULL;
}

static void *gguf_mmap(gguf_ctx *ctx, const char *name) {
    gguf_tensor *t = gguf_find(ctx, name);
    if (!t || t->nbytes <= 0) return NULL;
    if (t->mmap_ptr) return t->mmap_ptr;
    void *ptr = mmap(NULL, t->nbytes, PROT_READ, MAP_SHARED, ctx->fd, t->off);
    if (ptr == MAP_FAILED) {
        perror("gguf mmap");
        return NULL;
    }
    t->mmap_ptr = ptr;
    return ptr;
}

static void gguf_unmap(gguf_ctx *ctx, const char *name) {
    gguf_tensor *t = gguf_find(ctx, name);
    if (!t || !t->mmap_ptr) return;
    munmap(t->mmap_ptr, t->nbytes);
    t->mmap_ptr = NULL;
}

static void gguf_unmap_all(gguf_ctx *ctx) {
    for (int i = 0; i < ctx->n; i++) {
        if (ctx->t[i].mmap_ptr) {
            munmap(ctx->t[i].mmap_ptr, ctx->t[i].nbytes);
            ctx->t[i].mmap_ptr = NULL;
        }
    }
}

static void gguf_free(gguf_ctx *ctx) {
    for (int i = 0; i < ctx->n; i++) {
        if (ctx->t[i].mmap_ptr) {
            munmap(ctx->t[i].mmap_ptr, ctx->t[i].nbytes);
            ctx->t[i].mmap_ptr = NULL;
        }
        if (ctx->t[i].scales) free(ctx->t[i].scales);
        free(ctx->t[i].name);
    }
    free(ctx->t);
    free(ctx->hidx);
    free(ctx->path);
    if (ctx->fd >= 0) close(ctx->fd);
}

/* Read pre-quantized tensor data with scales into a float buffer.
 * For F32/F16: reads directly. For NVFP4/NVFP8: dequantizes on-the-fly.
 * Returns number of elements read, or -1 on error.
 * out: caller-allocated buffer of at least numel * sizeof(float)
 */
static int64_t gguf_read_tensor(gguf_ctx *ctx, const char *name, float *out, int64_t *out_numel) {
    gguf_tensor *t = gguf_find(ctx, name);
    if (!t) { fprintf(stderr, "gguf_read_tensor: tensor %s not found\n", name); return -1; }
    if (out_numel) *out_numel = 0;

    int64_t numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= t->shape[d];
    if (out_numel) *out_numel = numel;

    if (t->dtype == GGML_TYPE_F32) {
        /* Read F32 directly */
        uint8_t *buf = malloc(t->nbytes);
        if (pread(t->fd, buf, t->nbytes, t->off) != (ssize_t)t->nbytes) { free(buf); return -1; }
        memcpy(out, buf, t->nbytes);
        free(buf);
        return numel;
    } else if (t->dtype == GGML_TYPE_F16) {
        /* Read F16 and convert to F32 */
        uint16_t *buf = malloc(t->nbytes);
        if (!buf) { fprintf(stderr,"[GGUF] F16 alloc fail numel=%lld nbytes=%lld\n",(long long)numel,(long long)t->nbytes); return -1; }
        ssize_t got = pread(t->fd, buf, t->nbytes, t->off);
        if (got != (ssize_t)t->nbytes) { fprintf(stderr,"[GGUF] F16 pread fail got=%lld expected=%lld off=%lld\n",(long long)got,(long long)t->nbytes,(long long)t->off); free(buf); return -1; }
        /* Proper F16 -> F32 conversion */
        for (int64_t i = 0; i < numel; i++) {
            uint16_t h = buf[i];
            uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t man  = h & 0x3FF;
            uint32_t u;
            if (exp == 0) {
                if (man == 0) u = sign;
                else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
            } else if (exp == 0x1F) {
                u = sign | 0x7F800000 | (man << 13);
            } else {
                u = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            memcpy(&out[i], &u, 4);
        }
        free(buf);
        return numel;
    } else if (t->dtype == GGML_TYPE_NVFP4 || t->dtype == GGML_TYPE_NVFP8) {
        /* Pre-quantized tensor: read data + scales, dequantize on-the-fly */
        /* Find the data offset in the file */
        int64_t data_off = t->off;
        int64_t data_nbytes = 0;
        int64_t scales_nbytes = 0;
        int64_t scales_off = 0;

        /* For NVFP4/NVFP8, we need to find the scales.
         * The GGUF writer stores: data_bytes (quantized), then scales_bytes (F32 per-row) */
        /* Read scales first (they come after data in the file) */
        if (t->scales) {
            /* scales already loaded in model_init */
        } else {
            /* Read scales from file: need to find scales offset */
            /* The scales are stored right after the quantized data */
            /* For a [O, I] tensor with NVFP4: data = (O*I+1)/2 bytes, scales = O*4 bytes */
            if (t->dtype == GGML_TYPE_NVFP4) {
                data_nbytes = (numel + 1) / 2;
            } else {
                data_nbytes = numel;
            }
            scales_nbytes = t->shape[0] * 4;  /* O rows * 4 bytes per scale */
            scales_off = data_off + data_nbytes;

            /* Read scales */
            uint8_t *sbuf = malloc(scales_nbytes);
            if (pread(t->fd, sbuf, scales_nbytes, scales_off) != (ssize_t)scales_nbytes) { free(sbuf); return -1; }
            t->scales = malloc(t->shape[0] * sizeof(float));
            memcpy(t->scales, sbuf, scales_nbytes);
            t->n_scales = t->shape[0];
            free(sbuf);
        }

        /* Read quantized data */
        uint8_t *qdata = malloc(data_nbytes);
        if (pread(t->fd, qdata, data_nbytes, data_off) != (ssize_t)data_nbytes) { free(qdata); return -1; }

        /* Dequantize */
        if (t->dtype == GGML_TYPE_NVFP4) {
            /* FP4 E2M1: unpack 2 values per byte */
            const float *sc = t->scales;
            int64_t O = t->shape[0];
            int64_t I = numel / O;
            for (int64_t o = 0; o < O; o++) {
                float scale = sc[o];
                for (int64_t i = 0; i < I; i++) {
                    int64_t idx = o * I + i;
                    int byte_idx = idx / 2;
                    int bit_pos = (idx % 2) * 4;
                    uint8_t code = (qdata[byte_idx] >> bit_pos) & 0x0F;
                    /* Decode FP4 E2M1 */
                    float val = 0.0f;
                    int sign = (code >> 3) & 1;
                    int exp = (code >> 1) & 0x3;
                    int mant = code & 0x1;
                    if (code == 0) {
                        val = 0.0f;
                    } else if (exp == 0) {
                        /* Subnormal: val = mant/2 * 2^(-1) = mant * 0.25 */
                        val = mant * 0.25f;
                    } else if (exp == 3 && mant == 1) {
                        /* Max: val = (1+0.5) * 2^2 = 6.0 */
                        val = 6.0f;
                    } else {
                        /* Normal: val = (1 + mant/2) * 2^(exp-1) */
                        val = (1.0f + mant * 0.5f) * powf(2.0f, exp - 1);
                    }
                    if (sign) val = -val;
                    out[idx] = val * scale;
                }
            }
        } else if (t->dtype == GGML_TYPE_NVFP8) {
            /* FP8 E4M3: unpack 1 value per byte */
            const float *sc = t->scales;
            int64_t O = t->shape[0];
            int64_t I = numel / O;
            for (int64_t o = 0; o < O; o++) {
                float scale = sc[o];
                for (int64_t i = 0; i < I; i++) {
                    int64_t idx = o * I + i;
                    uint8_t code = qdata[idx];
                    /* Decode FP8 E4M3 */
                    float val = 0.0f;
                    int sign = (code >> 7) & 1;
                    int exp = (code >> 4) & 0x7;
                    int mant = code & 0xF;
                    if (code == 0) {
                        val = 0.0f;
                    } else if (exp == 0) {
                        /* Subnormal: val = mant * 2^(-8) */
                        val = mant * 0.00390625f;
                    } else if (exp == 15) {
                        /* Inf/NaN */
                        val = sign ? -1e30f : 1e30f;
                    } else {
                        /* Normal: val = (1 + mant/8) * 2^(exp-7) */
                        val = (1.0f + mant * 0.125f) * powf(2.0f, exp - 7);
                    }
                    if (sign) val = -val;
                    out[idx] = val * scale;
                }
            }
        }
        free(qdata);
        return numel;
    }
    return -1;
}

#endif
