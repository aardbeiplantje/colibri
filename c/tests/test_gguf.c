/* Test del minimal GGUF indexer.
 * Crea un file GGUF minimale con 3 tensori, lo legge, e verifica l'accesso. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include "../gguf.h"

static void gguf_write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void gguf_write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void gguf_write_string(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    gguf_write_u64(f, len);
    fwrite(s, 1, len, f);
}

static int create_test_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 0; }

    /* Header */
    fwrite("GGUF", 1, 4, f);
    gguf_write_u32(f, 3);              /* version 3 */
    gguf_write_u64(f, 3);              /* 3 tensors */
    gguf_write_u64(f, 0);              /* 0 KV pairs */

    /* Tensor 0: "embed.weight" — F32 [32000, 128] = 32000×128×4 = 16,384,000 bytes */
    gguf_write_string(f, "embed.weight");
    gguf_write_u32(f, GGML_TYPE_F32);
    gguf_write_u32(f, 2);
    gguf_write_u64(f, 32000);
    gguf_write_u64(f, 128);
    uint64_t off0 = ftell(f);
    gguf_write_u64(f, off0);
    /* Skip to offset for data */
    uint64_t data0 = off0;
    fseek(f, data0 + 32000L*128*4, SEEK_SET);

    /* Tensor 1: "norm.weight" — F32 [128] = 512 bytes */
    gguf_write_string(f, "norm.weight");
    gguf_write_u32(f, GGML_TYPE_F32);
    gguf_write_u32(f, 1);
    gguf_write_u64(f, 128);
    uint64_t off1 = ftell(f);
    gguf_write_u64(f, off1);
    uint64_t data1 = off1;
    fseek(f, data1 + 128*4, SEEK_SET);

    /* Tensor 2: "mlp.up.weight" — F32 [512, 128] = 512×128×4 = 262,144 bytes */
    gguf_write_string(f, "mlp.up.weight");
    gguf_write_u32(f, GGML_TYPE_F32);
    gguf_write_u32(f, 2);
    gguf_write_u64(f, 512);
    gguf_write_u64(f, 128);
    uint64_t off2 = ftell(f);
    gguf_write_u64(f, off2);
    uint64_t data2 = off2;
    fseek(f, data2 + 512*128*4, SEEK_SET);

    /* Write data: embed row 0 has values 0..127 */
    for (int i = 0; i < 128; i++) {
        float v = (float)i;
        fwrite(&v, 4, 1, f);
    }
    /* Rest of embed is zeros (we only check row 0) */
    uint64_t remaining = (32000L * 128 * 4) - (128 * 4);
    if (remaining > 0) {
        char *zero = calloc(1, 4096);
        uint64_t written = 0;
        while (written < remaining) {
            uint64_t chunk = remaining - written > 4096 ? 4096 : remaining - written;
            fwrite(zero, 1, chunk, f);
            written += chunk;
        }
        free(zero);
    }

    /* norm weight: values 1.0 */
    { float v = 1.0f; for (int i = 0; i < 128; i++) fwrite(&v, 4, 1, f); }

    /* mlp.up: values 0..127 per row, 512 rows */
    {
        for (int row = 0; row < 512; row++)
            for (int i = 0; i < 128; i++) {
                float v = (float)i;
                fwrite(&v, 4, 1, f);
            }
    }

    fclose(f);
    return 1;
}

int main(void) {
    const char *path = "/tmp/test_gguf.gguf";

    /* Step 1: create test file */
    if (!create_test_gguf(path)) return 1;
    printf("Created %s\n", path);

    /* Step 2: open index */
    gguf_ctx ctx;
    if (!gguf_init(&ctx, path)) return 1;
    printf("Opened index: %d tensors, version %u\n", ctx.n, ctx.version);

    /* Step 3: find and mmap tensors */
    gguf_tensor *t0 = gguf_find(&ctx, "embed.weight");
    gguf_tensor *t1 = gguf_find(&ctx, "norm.weight");
    gguf_tensor *t2 = gguf_find(&ctx, "mlp.up.weight");

    if (!t0 || !t1 || !t2) { fprintf(stderr, "missing tensor\n"); return 1; }
    printf("Found tensors:\n");
    printf("  embed.weight: dtype=%d shape=[%ld,%ld] off=%ld nbytes=%ld\n",
           t0->dtype, (long)t0->shape[0], (long)t0->shape[1], (long)t0->off, (long)t0->nbytes);
    printf("  norm.weight:  dtype=%d shape=[%ld] off=%ld nbytes=%ld\n",
           t1->dtype, (long)t1->shape[0], (long)t1->off, (long)t1->nbytes);
    printf("  mlp.up.weight: dtype=%d shape=[%ld,%ld] off=%ld nbytes=%ld\n",
           t2->dtype, (long)t2->shape[0], (long)t2->shape[1], (long)t2->off, (long)t2->nbytes);

    /* Step 4: mmap embed and read row 0 */
    void *embed_ptr = gguf_mmap(&ctx, "embed.weight");
    if (!embed_ptr) { fprintf(stderr, "mmap embed failed\n"); return 1; }
    const float *embed = (const float *)embed_ptr;
    int match = 1;
    for (int i = 0; i < 128; i++) {
        if (embed[i] != (float)i) {
            fprintf(stderr, "embed row0[%d]: expected %.1f got %.1f\n", i, (float)i, embed[i]);
            match = 0;
        }
    }
    printf("embed.row0: %s\n", match ? "OK" : "FAIL");

    /* Step 5: mmap norm and verify */
    void *norm_ptr = gguf_mmap(&ctx, "norm.weight");
    if (!norm_ptr) { fprintf(stderr, "mmap norm failed\n"); return 1; }
    const float *norm = (const float *)norm_ptr;
    match = 1;
    for (int i = 0; i < 128; i++) {
        if (norm[i] != 1.0f) {
            fprintf(stderr, "norm[%d]: expected 1.0 got %.1f\n", i, norm[i]);
            match = 0;
        }
    }
    printf("norm: %s\n", match ? "OK" : "FAIL");

    /* Step 6: mmap mlp and verify first row */
    void *mlp_ptr = gguf_mmap(&ctx, "mlp.up.weight");
    if (!mlp_ptr) { fprintf(stderr, "mmap mlp failed\n"); return 1; }
    const float *mlp = (const float *)mlp_ptr;
    match = 1;
    for (int i = 0; i < 128; i++) {
        if (mlp[i] != (float)i) {
            fprintf(stderr, "mlp.row0[%d]: expected %.1f got %.1f\n", i, (float)i, mlp[i]);
            match = 0;
        }
    }
    printf("mlp.up.row0: %s\n", match ? "OK" : "FAIL");

    /* Step 7: verify nonexistent tensor */
    if (gguf_has(&ctx, "nonexistent")) {
        fprintf(stderr, "should not find nonexistent\n");
        return 1;
    }
    printf("nonexistent: correctly not found\n");

    /* Cleanup */
    gguf_unmap_all(&ctx);
    gguf_free(&ctx);
    unlink(path);
    printf("All tests passed.\n");
    return 0;
}
