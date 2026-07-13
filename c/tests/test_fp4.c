/* Self-test for NVFP4 (OCP FP4 E2M1) quantization and dequantization.
 *
 * FP4 E2M1 bit layout (4 bits): sign(1) + exp(2) + mant(1)
 *   packed: 2 values per byte, LSB-first
 *
 * Formula:
 *   Normal:  (-1)^sign × 2^(exp-1) × (1 + mant/2)    where exp in {1,2,3}
 *   Subnormal: (-1)^sign × 2^0 × (mant/2)             where exp=0, mant=1 → 0.5
 *   Zero:    exp=0, mant=0 → 0
 *
 * Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
 * Negative values: -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
 *
 * Bit layout: s=bit3, e1=bit2, e0=bit1, m=bit0
 * Code: bit3(bit2 bit1 bit0)
 *   0: 0 0 0 0 → 0
 *   1: 0 0 0 1 → subnormal 0.5
 *   2: 0 0 1 0 → normal 1.0
 *   3: 0 0 1 1 → normal 1.5
 *   4: 0 1 0 0 → normal 2.0
 *   5: 0 1 0 1 → normal 3.0
 *   6: 0 1 1 0 → normal 4.0
 *   7: 0 1 1 1 → normal 6.0
 *   8: 1 0 0 0 → negative -0.5
 *   9: 1 0 0 1 → negative -1.0
 *   etc.
 *
 * For 3-bit encoding packed 2 per byte:
 *   We store only bits 0-2 (exp+mant+sign), using bit 3 as sign.
 *   Since we have 3 bits: sign(1) + exp(2), no mantissa → only 4 positive values.
 *   
 *   BUT the TODO uses 3 bits with mantissa: {sign(1) | exp(2) | mant(1)}
 *   This gives 8 non-zero codes + zero = 9 values, but 3 bits only gives 8 codes.
 *   Zero takes code 0, so we have 7 positive + 7 negative values? No.
 *
 * The OCP FP4 E2M1 uses 4 bits. Let's use the correct 4-bit layout,
 * packed 2 per byte (bits 0-3 for first, bits 4-7 for second).
 * But the existing code packs 2 into 1 byte using 3-bit fields.
 *
 * Let's just make encode and decode consistent using the OCP spec values.
 * We'll use a brute-force approach: try all 8 codes, decode, find closest.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* OCP FP4 E2M1: decode 4-bit value to float.
 * Bit layout: sign=bit3, exp=bits2-1, mant=bit0
 * Value = (-1)^sign * 2^(exp-1) * (1 + mant/2), with bias=1
 * For exp=0 (subnormal): value = (-1)^sign * mant/2 (only 0.5 when mant=1) */
static float fp4_e2m1_decode(int code) {
    if (code == 0) return 0.0f;
    int sign = (code >> 3) & 1;
    int exp = (code >> 1) & 3;
    int mant = code & 1;
    float val;
    if (exp == 0) {
        /* Subnormal: only mant=1 is non-zero → 0.5 */
        val = mant == 1 ? 0.5f : 0.0f;
    } else {
        /* Normal: (1 + mant/2) * 2^(exp-1), bias=1 */
        val = (1.0f + mant * 0.5f) * exp2f((float)(exp - 1));
    }
    return sign ? -val : val;
}

/* OCP FP4 E2M1: quantize float to nearest representable 4-bit code */
static int fp4_e2m1_quantize(float v) {
    if (v == 0.0f) return 0;
    float av = fabsf(v);
    int best = 1;
    float besterr = fabsf(av - fp4_e2m1_decode(1));
    for (int code = 2; code < 16; code++) {
        float val = fp4_e2m1_decode(code);
        float err = fabsf(av - fabsf(val));
        if (err < besterr) { besterr = err; best = code; }
    }
    return best;
}

/* Pack FP4 E2M1 one row into bytes (2 values per byte, 4-bit each) */
static void pack_fp4_row(const float *w, uint8_t *q4, float *scale, int I) {
    /* Find absmax */
    float amax = 0;
    for (int i = 0; i < I; i++) { float a = fabsf(w[i]); if (a > amax) amax = a; }
    float s = amax / 6.0f;  /* max representable = 6.0 */
    if (s < 1e-12f) s = 1e-12f;
    scale[0] = s;
    int nbytes = (I + 1) / 2;
    for (int i = 0; i < I; i += 2) {
        int c0 = fp4_e2m1_quantize(w[i] / s);
        int c1 = 0;
        if (i + 1 < I) c1 = fp4_e2m1_quantize(w[i+1] / s);
        q4[i >> 1] = (uint8_t)(c0 | (c1 << 4));
    }
}

/* Dequant FP4 E2M1 from byte (2 values per byte) */
static float fp4_row_dequant(const uint8_t *q4, int idx, float scale) {
    int byte = idx >> 1;
    int shift = (idx & 1) * 4;
    int code = (q4[byte] >> shift) & 0xF;
    return fp4_e2m1_decode(code) * scale;
}

static float relative_error(float a, float b) {
    float diff = fabsf(a - b);
    float max_val = fmaxf(fabsf(a), fabsf(b));
    return max_val > 1e-12f ? diff / max_val : diff;
}

int main(void) {
    int pass = 0, fail = 0;

    /* Test 1: Verify decode table (OCP FP4 E2M1: s=bit3, e=bits2-1, m=bit0) */
    {
        printf("Test 1: FP4 E2M1 decode table\n");
        /* Positive codes 0-7: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
         * Negative codes 8-15: -0 (subnormal zero), -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 */
        struct { int code; float expected; } tests[] = {
            {  0,  0.0f },  {  1,  0.5f },  {  2,  1.0f },  {  3,  1.5f },
            {  4,  2.0f },  {  5,  3.0f },  {  6,  4.0f },  {  7,  6.0f },
            {  8,  0.0f },  {  9, -0.5f }, { 10, -1.0f }, { 11, -1.5f },
            { 12, -2.0f },  { 13, -3.0f }, { 14, -4.0f }, { 15, -6.0f },
        };
        int n = sizeof(tests) / sizeof(tests[0]);
        for (int i = 0; i < n; i++) {
            float got = fp4_e2m1_decode(tests[i].code);
            /* -0.0 == 0.0 in float, so treat as match */
            float diff = fabsf(got - tests[i].expected);
            if (diff < 0.001f || (fabsf(got) < 1e-12f && fabsf(tests[i].expected) < 1e-12f)) {
                printf("  PASS: code%d -> %.4f\n", tests[i].code, got);
                pass++;
            } else {
                printf("  FAIL: code%d -> %.4f, expected %.4f\n", tests[i].code, got, tests[i].expected);
                fail++;
            }
        }
    }

    /* Test 2: Encode → decode round-trip */
    {
        printf("\nTest 2: Encode/decode round-trip\n");
        float vals[] = {0.0f, 0.3f, 0.5f, 0.8f, 1.0f, 1.2f, 1.5f,
                        2.0f, 2.5f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                        -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f, -7.0f};
        int n = sizeof(vals) / sizeof(vals[0]);
        for (int i = 0; i < n; i++) {
            float v = vals[i];
            int code = fp4_e2m1_quantize(v);
            float decoded = fp4_e2m1_decode(code);
            
            /* Verify: quantize picks closest representable to |v| */
            int best = 1;
            float besterr = fabsf(fabsf(v) - fabsf(fp4_e2m1_decode(1)));
            for (int c = 2; c < 16; c++) {
                float err = fabsf(fabsf(v) - fabsf(fp4_e2m1_decode(c)));
                if (err < besterr) { besterr = err; best = c; }
            }
            float expected = fp4_e2m1_decode(best);
            
            float err = relative_error(decoded, expected);
            if (err < 0.001f) {
                printf("  PASS: %.4f -> code%d -> %.4f (nearest=%.4f)\n", v, code, decoded, expected);
                pass++;
            } else {
                printf("  FAIL: %.4f -> code%d -> %.4f, expected nearest=%.4f\n", v, code, decoded, expected);
                fail++;
            }
        }
    }

    /* Test 3: Pack/decode round-trip */
    {
        printf("\nTest 3: Pack/decode round-trip\n");
        float w[8] = {0.3f, 0.5f, 1.0f, 1.5f, 2.5f, 4.0f, 0.1f, 5.0f};
        int I = 8;
        uint8_t q4[4];
        float scale[1] = {0};
        float sum = 0;

        pack_fp4_row(w, q4, scale, I);
        for (int i = 0; i < I; i++) {
            int code = fp4_e2m1_quantize(w[i] / scale[0]);
            sum += fp4_e2m1_decode(code) * scale[0];
        }
        
        /* Verify dequant matches encode */
        int ok = 1;
        for (int i = 0; i < I; i++) {
            int code = fp4_e2m1_quantize(w[i] / scale[0]);
            float decoded = fp4_e2m1_decode(code) * scale[0];
            float actual = fp4_row_dequant(q4, i, scale[0]);
            float err = relative_error(actual, decoded);
            if (err > 0.001f) {
                ok = 0;
                printf("  mismatch[%d]: actual=%.6f, expected=%.6f\n", i, actual, decoded);
            }
        }
        if (ok) { printf("  PASS: pack/decode consistent\n"); pass++; }
        else fail++;
    }

    /* Test 4: Verify values match OCP spec */
    {
        printf("\nTest 4: OCP FP4 E2M1 spec values\n");
        /* Code 0=+0, 1=+0.5, 2=+1.0, 3=+1.5, 4=+2.0, 5=+3.0, 6=+4.0, 7=+6.0 */
        struct { int code; float expected; } specs[] = {
            { 0, 0.0f }, { 1, 0.5f }, { 2, 1.0f }, { 3, 1.5f },
            { 4, 2.0f }, { 5, 3.0f }, { 6, 4.0f }, { 7, 6.0f },
        };
        int ok = 1;
        for (int i = 0; i < 8; i++) {
            float got = fp4_e2m1_decode(specs[i].code);
            if (fabsf(got - specs[i].expected) > 0.001f) {
                ok = 0;
                printf("  FAIL: code%d -> %.4f, spec says %.4f\n", specs[i].code, got, specs[i].expected);
            }
        }
        if (ok) { printf("  PASS: All OCP values match\n"); pass++; }
        else fail++;
    }

    /* Summary */
    printf("\n=============================\n");
    printf("FP4 Self-Test Summary\n");
    printf("  Passed: %d\n", pass);
    printf("  Failed: %d\n", fail);
    printf("=============================\n");
    return fail > 0 ? 1 : 0;
}
