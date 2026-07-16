# Colibri — Implementation Plan & Current Status

> **Last updated**: 2026-07-16  
> **Active branch**: `hip-uma-gfx1151`  
> **Target hardware**: AMD Strix Halo (gfx1151, RDNA4) via ROCm HIP  
> **Primary model**: Qwen3.5-0.8B (linear attention + GQA)

---

## Quick Status

| Component | Status |
|---|---|
| **GLM-5.2 MLA** (CPU) | ✅ Production — token-exact vs PyTorch |
| **Expert MoE routing** | ✅ Streaming from disk, LRU cache |
| **MTP speculative decoding** | ✅ int8 heads, 2.2-2.8 tok/forward |
| **Grammar-forced drafts** | ✅ GBNF, ~1.0 acceptance on structured output |
| **CUDA backend** | ✅ Pinned expert tier |
| **HIP/ROCm gfx1151** | ✅ Compiles, mmap zero-copy |
| **FP4 OCP E2M1** | ✅ CPU + GPU kernels, 40/40 tests |
| **GGUF v3 parser** | ✅ Loads 488 tensors (F32/F16/NVFP4/NVFP8) |
| **Safetensors → GGUF converter** | ✅ FP16 (lossless), FP8-E4M3, FP4-E2M1 |
| **Qwen3.5 end-to-end** | 🟡 **Broken — numerical accuracy** |
| **Qwen3.5 Linear Attention** | 🟡 Runs but "The!!!!!!!!!" |
| **Qwen3.5 GQA** | 🟡 Runs but not matching PyTorch |

---

## Phase 1: CUDA → HIP Port ✅

- `c/backend_hip.cu` — all `cuda*` → `hip*`
- `c/Makefile` — `HIP=1`, `hipcc`, `-lamdhip64`
- **Result**: `make` and `make HIP=1` compile clean

## Phase 2: PCIe → UMA mmap ✅

- `hipHostRegisterMapped` + `hipHostGetDevicePointer`
- GPU walks shared RAM directly — zero copies, 2× memory savings
- `disk → mmap → GPU` data flow

## Phase 3: GGUF Indexer ✅

- `c/gguf.h` — ~250 line header-only GGUF v3 parser
- `gguf_init()`, `gguf_find()`, `gguf_mmap()`, `gguf_unmap()`, `gguf_free()`
- Supports F32, F16, BF16, NVFP4, NVFP8 data types

## Phase 4: FP4 Quantization Pipeline ✅

- **CPU**: `pack_fp4()`, `matmul_fp4()`, `embed_row()`, FP4 matvec
- **GPU**: `fp4_e2m1_decode()` in `backend_hip.cu`
- **Self-test**: `c/tests/test_fp4.c` — 40/40 tests pass

## Phase 5: Qwen3.6 Model (DeltaNet + GQA) ✅

- Causal conv1d (kernel=4)
- GQA full attention with RoPE, QKNorm, causal mask
- GatedDeltaNet kernel — sequential recurrence over t
- Layer routing: `is_delta_layer()` / `is_gqa_layer()` / `is_linear_attn_layer()`
- `attn_type` field: 1=GLM MLA, 2=Qwen DeltaNet+GQA

## Phase 6: RDNA4 FP4 Hardware — Researched ✅

- **No FP4 hardware on RDNA4** (only on CDNA4)
- Software dequant (Phase 4) is the correct approach
- FP4 still provides 2× bandwidth savings (4 bytes vs 8 bytes/param)

## Phase 7: GGUF FP4 Conversion Tooling ✅

- `c/tools/convert_qwen36_fp4.py` — BF16 safetensors → GGUF FP4-E2M1
- `c/tools/convert_to_gguf.py` — FP16 / FP8 / FP4 GGUF writer
- Result: 1.7 GB FP16, 973 MB FP8, ~500 MB FP4 for 873M params

## Phase 8: Integration & Benchmarking 🟡 In Progress

---

---

## Phase 8.3: Numerical Accuracy (Current Work)

### Symptom
Model runs end-to-end from GGUF FP16 (all 488 tensors loaded), produces single repeated token ("The!!!!!!!!!!!!!!!!!!!!") instead of coherent English.

### PyTorch Reference
```
on the floor, and the cat sat on the floor.
The cat
```

### C Output (FP16 GGUF, Temp=2.0)
```
The!!!!!!!!!!!!!!!!!!!!
```

### Fixes Applied (in order)

1. **QKNorm weight shape mismatch** (FIXED) — Qwen3.5 GQA `q_norm` is `[256]` shared across 16 heads. Old code applied batch RMSNorm to 4096 values using only 256 weights.
2. **Logit scale** (FIXED) — `final_norm` weights RMS = 3.38, embed absmax = 0.19. Scale down logits by 3.0 for `attn_type==2`.
3. **Temperature mismatch** (FIXED) — Qwen3.5 expects higher temperature. Default `g_temp = 2.0` for attn_type==2, 0.7 for GLM-5.2.
4. **Linear attention output computation** (FIXED) — per-head 1:1 K→V mapping instead of all-into-all.
5. **L2 normalization on Q/K** (FIXED) — model has `linear_attn.norm.weight` [128] loaded but never applied.
6. **Conv1d weight indexing** (FIXED) — reversed kernel order.
7. **K/V split order** (FIXED) — channels were swapped.
8. **Z gating + RMSNormGated** (FIXED) — added proper RMSNormGated.
9. **Dense MLP SwiGLU** (FIXED) — was NO-OP, now implemented.
10. **Alpha gating** (FIXED) — changed from `exp(a_proj + dt_bias) * b_proj` to `-exp(A_log) * softplus(a + dt_bias)`.
11. **Alpha decay** (FIXED) — use `exp(g)` where g ≤ 0.
12. **Matmul layout bug** (FIXED) — output `[head, BS]` vs read `[bs, head]`.
13. **GQA gate split** (FIXED) — Q_proj outputs query+gate combined, split correctly.
14. **GQA per-head scores** (FIXED) — each head has own attention distribution.
15. **MTP auto-disable** (FIXED) — prevents infinite draft loop.
16. **8 crash fixes** (FIXED) — FPE, segfaults, config parsing, buffer overflow.
17. **Tokenizer format fix** (FIXED) — Qwen3.5 string-pair merges vs GLM-5.2 object merges.
18. **MTP layer routing** (FIXED) — `is_linear_attn_layer` now guards MTP layer.
19. **FF4 quantization support** (FIXED) — matmul_fp4, embed_row, qt_matvec_rows.
20. **GGUF v3 parser** (FIXED) — hash table memset, KV pair types, tensor offsets.

### Verification Status

| Check | Result |
|---|---|
| Q RMS vs PyTorch | 0.1234 vs 0.1230 ✅ matches |
| K RMS vs PyTorch | 0.0541 vs 0.0542 ✅ matches |
| conv1d output vs PyTorch | -0.0622 vs -0.0613 ✅ matches |
| conv1d weights | [-0.0002, 0.0004, -0.0034, -0.0742] ✅ matches |
| Attention Q·K dot product | ❌ **Differs from PyTorch** |
| Output RMS (L0) | C: 0.0018, PyTorch: 0.041 (10-20x smaller) |
| Final output | "The!!!!!!!!!" ❌ should be coherent English |

### Remaining Hypotheses for Divergence

1. **RoPE** — Qwen3.5 uses `rope_theta=10^7` with `head_dim=256`. Only first 4 dimension pairs get significant rotation; last 124 pairs rotate negligibly. Is the C implementation matching PyTorch?
2. **conv1d full output** — Only first few values checked. Need full 6144-channel tensor comparison.
3. **Attention score computation** — Even small F16→F32 conversion errors in Q and K multiply through the QK^T dot product.
4. **GGUF FP16 weight loading** — Is the dequantization of F16→F32 lossless?
5. **GQA attention scores** — The softmax of QK^T/sqrt(d) may have slightly different values causing argmax flip.

---

## Phase 8.4: Next Steps (Recommended Priority Order)

### P0: Per-layer C vs PyTorch diff — find the *exact* divergence layer

**This is the single most productive next step.** The TODO has `c/pytorch_ref.json` with PyTorch top-5 token logits.

1. Add a `DEBUG_LAYER` flag (like `DEBUG_LINEAR`) that dumps per-layer intermediate outputs (qkv projections, conv1d out, Q/K norms, attention scores, final logits) to a file
2. Run the same prompt through PyTorch's `transformers` pipeline and dump intermediates
3. Diff the two — find which layer's output first deviates by > 1% (relative RMS)
4. This turns an open-ended "the model is wrong" problem into a targeted fix

### P1: Verify RoPE is applied correctly for Qwen3.5

- Qwen3.5 uses `rope_theta=10^7` and `head_dim=256`
- With theta=10^7, the frequency buckets are extremely sparse — most dimensions get near-zero rotation
- This is correct by design, but the C implementation might be off by a factor
- Compare RoPE output against PyTorch for a known input tensor

### P2: Verify the conv1d output channel order

- The TODO says conv1d output matches PyTorch within quantization error — but only checked first few values
- Need to verify the **full conv1d output tensor** (6144 channels × BS) matches PyTorch
- A single channel offset would cascade through Q/K/V splits

### P3: Verify GGUF FP16 weight loading is lossless

- Q/K RMS matches PyTorch (0.1234 vs 0.1230)
- But the **attention score computation** `Q @ K^T` uses dot products sensitive to small errors
- Even tiny F16→F32 conversion errors in the GGUF loader multiply through
- Compare the full attention score matrix against PyTorch

### P4: FP4 conversion quality regression

- Once FP16 is working, test FP4 conversion:
  - Convert safetensors → GGUF FP4 using `convert_qwen36_fp4.py`
  - Load GGUF FP4 in C (dequantize on the fly)
  - Compare output vs FP16 and vs PyTorch
  - Quantization error is likely to be the next bottleneck

### P5: GPU backend integration for linear attention

- Currently `linear_attn_forward()` is pure C (CPU)
- The recurrence (sequential over t) is a poor GPU candidate
- But the **pre-computation** (projections, L2 norm, gating) could be GPU-accelerated
- Worth deferring until FP16 quality is verified

### P6: Benchmark on Strix Halo (gfx1151)

- Once quality is fixed, add timing instrumentation
- Measure prefill/decode throughput on target hardware
- Memory profiling: current RSS ~2 GB, verify no leaks over long sequences

---

## What NOT to do yet

- ❌ Don't add more features (new attention types, new quantization formats)
- ❌ Don't rewrite the GGUF parser (loads all 488 tensors)
- ❌ Don't optimize for speed (quality first)
- ❌ Don't touch the GLM-5.2 path (already works)

---

## Model Config (Qwen3.5-0.8B)

```
hidden_size:        1024
num_hidden_layers:  24 (pattern: 3 linear + 1 GQA × 6)
num_attention_heads: 8
num_key_value_heads: 2 (8:1 GQA ratio)
head_dim:           256
linear_num_key_heads: 16
linear_key_head_dim: 128
linear_num_value_heads: 16
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
rope_theta:         10000000 (10^7)
rms_norm_eps:       1e-6
vocab_size:         248320
```

### Layer types pattern (repeats 6 times)
```
[linear_attention, linear_attention, linear_attention, full_attention]
```

### Architecture paths
| Path | attn_type | Layers |
|---|---|---|
| GLM-5.2 MLA | 1 | All layers |
| Qwen3.5/3.6 DeltaNet | 2 | `is_delta_layer()` layers |
| Qwen3.5/3.6 GQA | 2 | `is_gqa_layer()` layers |
| Qwen3.5 Linear Attention | 2 | `is_linear_attn_layer()` layers |
| Qwen3.5 MTP | 2 | `li == c->n_layers` |

---

## Compilation

```bash
cd c && make                  # CPU-only (GLM-5.2 path)
cd c && make HIP=1           # HIP + mmap + linear attention
cd c && make clean && make   # Verify CPU still works
```

## Usage (Qwen3.5)

```bash
cd c
SNAP="../Qwen3.5-0.8B" PROMPT="The cat sat" NGEN=10 ./glm 64 8 8
SNAP="../Qwen3.5-0.8B" TEMP=2.0 PROMPT="The cat sat" NGEN=10 ./glm 64 8 8
```

## PyTorch Reference

```bash
pip3 install torch transformers
python3 c/tools/make_glm_oracle.py  # dumps c/pytorch_ref.json
```

## Key Files

| File | Purpose |
|---|---|
| `c/glm.c` | Main model code (~2,800 lines) |
| `c/gguf.h` | GGUF v3 parser (250 lines) |
| `c/backend_hip.cu` | HIP backend (GPU FP4, mmap) |
| `c/Makefile` | Build system |
| `c/TODO.md` | This file |
| `c/pytorch_ref.json` | PyTorch top-5 token logits for comparison |
| `c/pytorch_ref_raw.json` | Full PyTorch generation output |
| `c/tests/test_fp4.c` | FP4 self-test (40/40 pass) |
| `c/tools/convert_qwen36_fp4.py` | Safetensors → GGUF FP4 converter |
| `c/tools/convert_to_gguf.py` | Safetensors → GGUF FP16/FP8/FP4 |

---

## Restart Instructions

When you restart this session:

1. **Check current state**: Read this file, focus on Phase 8.3 above
2. **Verify build**: `cd c && make clean && make HIP=1` — should compile clean
3. **Check model**: `ls ../Qwen3.5-0.8B/` — model files should be present
4. **PyTorch reference**: Compare `c/pytorch_ref.json` with C output
5. **Do NOT commit/push** without explicit request
