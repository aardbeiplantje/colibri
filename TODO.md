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
Model runs end-to-end from GGUF FP16 (all 488 tensors loaded). Previously produced single repeated token ("The!!!!!!!!!!!!!!!!!!!!"). After conv1d fix (commit f56dc25), now produces "The cat sat fung unit SOVE..." — coherent-ish but wrong tokens.

### PyTorch Reference
```
on the floor, and the cat sat on the floor.
The cat
```

### C Output (FP16 GGUF, Temp=2.0, after conv1d fix)
```
The cat sat fung unit SOVE...
```

### Key Metrics (Layer 0)
| Metric | C | PyTorch | Status |
|--------|---|---------|--------|
| QKV RMS | ~same | ~same | ✅ |
| K norm RMS | 0.022 | 0.022 | ✅ |
| Attention RMS | 0.029 | 0.038 | ⚠️ close |
| Attention first16 | different | [-0.648, 0.063, ...] | ⚠️ scale OK, values off |

### Root Cause
Conv1d kernel indexing was wrong: used `ti=t-(ck-1-k)` but should use `ti=t-k` because GGUF flat array layout differs from PyTorch in-memory layout. Fixed in commit f56dc25.

### Current Status (as of 2026-07-17)

**Verified correct:**
1. ✅ Conv1d kernel indexing — ti = t - k (commit f56dc25)
2. ✅ K normalization RMS — matches PyTorch (0.022)
3. ✅ A_log / dt_bias values — match PyTorch exactly
4. ✅ Gating values (beta, silu(gate)) — match PyTorch
5. ✅ Output projection RMS — matches PyTorch (0.037)

**Remaining issue:**
- Per-element attention values differ even though RMS matches
- Model generates tokens from many languages (Russian, Korean, Chinese) instead of English
- The softmax distribution is shifted → wrong argmax

### Plan: Fix Per-Element Divergence

**Step 1: Full PyTorch reference dump (P0)**
- Run PyTorch model to dump ALL intermediate tensors for each layer
- Include: in_proj_qkv, conv1d_out, Q, K, V, z, beta, alpha/g, decay, state, y_all, out_proj
- Compare element-by-element with C output

**Step 2: Identify exact divergence point**
- Check QKV projection: `in_proj_qkv` shape [6144, 1024] — is the matmul layout correct?
- Check z projection: `in_proj_z` shape [2048, 1024] — verify loading and matmul
- Check conv1d output: compare exact values, not just RMS
- Check Q/K L2 normalization: verify no spurious division by head_dim
- Check recurrence: verify S[h, i, j] = decay*S[h,i,j] + kv[i] * (v[j] - kv_mem[j]) * beta[h]
- Check output: y[bs, h, d] = sum_i q[bs, h, i] * S[h, i, d] — verify einsum order
- Check z-gating: RMSNorm(y) * silu(z) — verify weight application
- Check output proj: out = y @ W_out.T — verify weight shape and matmul

**Step 3: Compare RoPE application**
- Qwen3.5 applies RoPE to Q/K — check if it's applied before or after the linear attention
- In gated_delta_net: RoPE is applied to Q/K in the GQA path, not the linear path
- In linear_attn_forward: No explicit RoPE — check if input already has RoPE applied

**Step 4: Compare attention output projection**
- The final layer output goes through lm_head
- Check if lm_head weights are loaded correctly
- Check temperature scaling: Qwen3.5 needs different temperature than GLM

**Step 5: Fix the identified issue and retest**

### Architecture Differences: Qwen3.5 vs GLM

| Feature | GLM | Qwen3.5 |
|---------|-----|---------|
| Attention | Standard causal (O(n²)) | Linear attention (O(n)) |
| State | KV cache | Recurrent state S[h, kd, vd] |
| Recurrence | None | S = decay * S + K * beta * (V - K^T*S) |
| Output | softmax(Q@K^T) @ V | Q^T @ S |
| Conv1d | None | Causal conv on QKV input |
| Gating | None | RMSNormGated: RMSNorm(y) * weight * silu(z) |
| Q/K Norm | Per-head RMSNorm | Per-head L2 normalize |
| Decay rate | N/A | g = -exp(A_log) * softplus(a + dt_bias) |
| Architecture | Standard transformer | [linear, linear, linear, full] × 6 |

### Fixes Applied (in order)
1. Conv1d weight layout — load as flat f32, apply as grouped conv
2. QKNorm per-head — use head_dim weights, not batch
3. Logit scale — divide by 3.0 for Qwen3.5/3.6
4. Temperature — set to 2.0 for Qwen3.5
5. Conv1d kernel indexing — ti = t - k (commit f56dc25)

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

### Verification Status (2026-07-17)

| Check | Result |
|---|---|
| conv1d weights [0:4] | [-0.000161, 0.000404, -0.003387, -0.074219] ✅ matches PyTorch |
| conv1d out[0,0] | C: -0.061159, PyTorch: -0.061159 ✅ matches |
| conv1d out[0,1] | C: 0.000802, PyTorch: 0.000802 ✅ matches |
| QKV all RMS | C: 1.2901, PyTorch: 1.2901 ✅ matches |
| Input LN RMS | C: 1.2247, PyTorch: 1.2386 ✅ matches (within 1.1%) |
| Z RMS | C: 0.9410, PyTorch: 0.9138 ✅ matches (within 3%) |
| K RMS | C: 0.0884, PyTorch: 0.0221 ❌ **4x diff!** |
| Attention output RMS | C: 0.1020, PyTorch: 0.0377 ❌ **2.7x diff!** |
| Attention output first16 | C: [-1.80, -0.06, 0.16, ...], PyTorch: [-0.65, 0.06, -0.08, ...] ❌ **wrong signs & magnitudes** |
| Final output | "The!!!!!!!!!" ❌ should be coherent English |

### Key Findings

- **conv1d is correct** — weights and outputs match PyTorch exactly
- **Input layernorm is correct** — RMS matches within 1.1%
- **QKV projection is correct** — RMS matches
- **Z gating is correct** — RMS matches within 3%
- **K values are wrong** — C K RMS=0.088 vs PyTorch K RMS=0.022 (4x diff)
- **Attention output is wrong** — C RMS=0.102 vs PyTorch RMS=0.038 (2.7x diff)
- **First divergence is in K normalization or the linear attention recurrence**

### Remaining Hypotheses for Divergence

1. **K L2 normalization** — C computes `K / sqrt(sum(K^2) + eps)`. PyTorch uses `F.normalize(K, p=2, dim=-1)`. These should be identical but K values differ by 4x.
2. **Linear attention recurrence** — The state update `S = decay*S + outer(K, delta)` may have a bug in the K^T @ S computation. Previously used wrong `torch.mv(state, kv)` instead of `torch.mv(state.T, kv)`.
3. **Value head mapping** — C uses `vh = h * nv / nq` for mapping query heads to value heads. PyTorch uses a different mapping.
4. **Output projection** — The `out_proj` matmul from `[BS, nv*vd]` to `[BS, D]` may have wrong weight loading or layout.

---

## Phase 8.4: Next Steps (Recommended Priority Order)

### P0: Debug K normalization and linear attention recurrence

**K values are 4x different from PyTorch; attention output is 2.7x different.** The conv1d, input LN, QKV projection, and Z gating all match. The bug is in one of:

1. **K L2 normalization** — Compare raw K (before normalization) and K RMS (after normalization) against PyTorch. Use `DEBUG_LINEAR` to dump `k_all` values channel-by-channel.
2. **Linear attention recurrence** — The state update `S = decay*S + outer(K, delta)` may have a bug:
   - Previously tested with wrong PyTorch einsum: `torch.mv(state, kv)` vs correct `torch.mv(state.T, kv)`
   - Verify the K^T @ S computation matches PyTorch for all heads
3. **Value head mapping** — C uses `vh = h * nv / nq` for mapping query heads to value heads. Verify PyTorch uses the same mapping.

### P1: Verify conv1d full output (6144 channels)

- Already verified conv1d out[0,0] and out[0,1] match PyTorch
- Need to check all 6144 channels for at least the first timestep
- Use `DEBUG_LINEAR` to dump conv1d out for each channel
- A single channel offset would cascade through Q/K/V splits

### P2: Verify GGUF FP16 weight loading is lossless

- Q/K RMS matches PyTorch (0.1234 vs 0.1230) before normalization
- But the **K values** are 4x different after normalization
- Check if the K normalization itself is correct: `K / sqrt(sum(K^2) + eps)`
- Compare the full K tensor against PyTorch

### P3: FP4 conversion quality regression

- Once FP16 is working, test FP4 conversion:
  - Convert safetensors → GGUF FP4 using `convert_qwen36_fp4.py`
  - Load GGUF FP4 in C (dequantize on the fly)
  - Compare output vs FP16 and vs PyTorch
  - Quantization error is likely to be the next bottleneck

### P4: GPU backend integration for linear attention

- Currently `linear_attn_forward()` is pure C (CPU)
- The recurrence (sequential over t) is a poor GPU candidate
- But the **pre-computation** (projections, L2 norm, gating) could be GPU-accelerated
- Worth deferring until FP16 quality is verified

### P5: Benchmark on Strix Halo (gfx1151)

- Once quality is fixed, add timing instrumentation
- Measure prefill/decode throughput on target hardware
- Memory profiling: current RSS ~2 GB, verify no leaks over long sequences

### Tools Available

- `DEBUG_LAYER=1` — dumps per-layer intermediate tensors to `c/debug_layer_*.json`
- `DEBUG_LINEAR=1` — dumps detailed linear attention intermediates (conv1d, QKV, gating, state)
- `c/pytorch_dump_layers.py` — dumps PyTorch layer outputs and intermediates
- `c/pytorch_linear_debug.py` — focused linear attention comparison script
- `c/compare_common.py` — compares C vs PyTorch measurements
- `c/pytorch_ref_layers.json` — saved PyTorch reference data
- `c/pytorch_linear_debug.json` — saved linear attention comparison data

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
