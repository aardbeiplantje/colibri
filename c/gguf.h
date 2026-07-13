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
#include "compat.h"

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_BF16 = 3,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_NVFP4 = 40,
};

static inline int64_t ggml_type_bytes(int t) {
    switch (t) {
    case GGML_TYPE_F32: return 4;
    case GGML_TYPE_F16: return 2;
    case GGML_TYPE_BF16: return 2;
    case GGML_TYPE_Q4_0: return 0;
    case GGML_TYPE_NVFP4: return 0;
    default: return -1;
    }
}

typedef struct {
    char   *name;
    int     fd;
    int64_t off;
    int64_t nbytes;
    int     dtype;
    int64_t shape[8];
    int     ndim;
    void    *mmap_ptr;
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
    ctx->fd = open(path, COMPAT_O_RDONLY);
    if (ctx->fd < 0) { perror(path); return 0; }
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
    for (uint64_t i = 0; i < kv_count; i++) {
        char *kv_name;
        if (gguf_pread_str(ctx->fd, pos, &kv_name) < 0) goto error;
        int64_t nlen = (int64_t)strlen(kv_name);
        uint32_t kv_type;
        if (gguf_pread_u32(ctx->fd, pos + 8 + nlen, &kv_type) < 0) { free(kv_name); goto error; }
        if (kv_type == 0) pos += 8 + nlen + 1 + 4;
        else if (kv_type <= 7) pos += 8 + nlen + 8 + 4;
        else if (kv_type <= 15) pos += 8 + nlen + 4 + 4;
        else { uint64_t slen; gguf_pread_u64(ctx->fd, pos + 8 + nlen, &slen); pos += 8 + nlen + 8 + slen + 4; }
        free(kv_name);
    }

    for (uint64_t ti = 0; ti < tensor_count && ctx->n < ctx->cap; ti++) {
        char *name;
        if (gguf_pread_str(ctx->fd, pos, &name) < 0) goto error;
        int64_t nlen = (int64_t)strlen(name);
        uint32_t dtype;
        if (gguf_pread_u32(ctx->fd, pos + 8 + nlen, &dtype) < 0) { free(name); goto error; }
        uint32_t ndim;
        if (gguf_pread_u32(ctx->fd, pos + 8 + nlen + 4, &ndim) < 0) { free(name); goto error; }
        int64_t shp = 0;
        for (int d = 0; d < (int)ndim && d < 8; d++) {
            uint64_t sv;
            if (gguf_pread_u64(ctx->fd, pos + 8 + nlen + 8 + d*8, &sv) < 0) { free(name); goto error; }
            ctx->t[ctx->n].shape[d] = (int64_t)sv;
            shp += 8;
        }
        uint64_t off;
        if (gguf_pread_u64(ctx->fd, pos + 8 + nlen + 8 + shp, &off) < 0) { free(name); goto error; }
        ctx->t[ctx->n].dtype = (int)dtype;
        ctx->t[ctx->n].ndim = (int)ndim;
        ctx->t[ctx->n].off = (int64_t)off;
        int64_t eb = ggml_type_bytes(ctx->t[ctx->n].dtype);
        if (eb > 0) {
            int64_t numel = 1;
            for (int d = 0; d < ctx->t[ctx->n].ndim; d++) numel *= ctx->t[ctx->n].shape[d];
            ctx->t[ctx->n].nbytes = numel * eb;
        }
        ctx->t[ctx->n].name = name;
        ctx->t[ctx->n].fd = ctx->fd;
        ctx->t[ctx->n].mmap_ptr = NULL;
        ctx->n++;
        pos += 8 + nlen + 4 + 4 + shp + 8 + 4;
    }

    ctx->hcap = 1; while (ctx->hcap < ctx->n * 2) ctx->hcap <<= 1;
    ctx->hidx = calloc(ctx->hcap, sizeof(int));
    if (!ctx->hidx) goto error;
    for (int i = 0; i < ctx->n; i++) {
        uint64_t h = gguf_hash(ctx->t[i].name) & (ctx->hcap - 1);
        while (ctx->hidx[h] >= 0) h = (h + 1) & (ctx->hcap - 1);
        ctx->hidx[h] = i;
    }
    lseek(ctx->fd, 0, SEEK_SET);
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
        free(ctx->t[i].name);
    }
    free(ctx->t);
    free(ctx->hidx);
    free(ctx->path);
    if (ctx->fd >= 0) close(ctx->fd);
}

#endif
