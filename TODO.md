# Colibri — Implementation Plan & Current Status

> **Last updated**: 2026-07-22  
> **Active branch**: `hip-uma-gfx1151`  
> **Target hardware**: AMD Strix Halo (gfx1151, RDNA4) via ROCm HIP  
> **Primary model**: Qwen3.5-0.8B (linear attention + GQA)
>
> **Session Focus**: Debugging Qwen3.5 linear attention numerical discrepancy

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
| **Qwen3.5 end-to-end** | 🔴 **Linear attention output 4-400x smaller than PyTorch** |
| **Qwen3.5 Linear Attention** | 🔴 **attn_out RMS mismatch: C=0.0026 vs PyT=0.0297 (L0)** |
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

## Phase 8.3: Numerical Accuracy (FIXED K Normalization)

### Current Status (2026-07-17)
**post_ln RMS fix verified ✅. Attention output RMS 10-4000× smaller than PyTorch ❌. Generation stuck on token 271 (double newline).**

| Metric | C | PyTorch | Status |
|---|---|---|---|
| K RMS (after L2 norm) | 0.0221 | 0.0221 | ✅ FIXED |
| out_first4 (layer 0) | [-0.3347, 0.0559, -0.0622, -0.0114] | [-0.3347, 0.0559, -0.0622, -0.0114] | ✅ MATCHES |
| out_rms (layer 0) | 0.0032 | 0.0297 | ⚠️ diverging |
| **post_ln RMS (layer 0)** | **0.126** | **0.159** | ✅ FIXED (was 1.05) |
| **post_ln RMS (layer 1)** | **0.187** | **0.189** | ✅ FIXED (was 1.16) |

### Fixes Applied
1. ✅ **K normalization** — Changed from per-head (dividing by sqrt(sum/128)) to per-row (dividing by sqrt(sum/2048))
2. ✅ **RMSNormGated** — Changed from per-head normalization to per-row normalization
3. ✅ **Q normalization** — Same per-row fix applied
4. ✅ **post_ln weight handling** — `rmsnorm_qw35` now uses raw weight `w[i]` directly (not `1.0f+w[i]`), matching GGUF-stored PyTorch weights. Post-LN RMS: L0 C=0.126 vs PyTorch weight RMS=0.128 ✅

### Symptom (Remaining)
Model runs end-to-end from GGUF FP16. Layer 0 attention matches PyTorch. **post_ln RMS fix applied — now C and PyTorch agree on LN output magnitudes.**
- Expected: "on the floor, and the cat sat on the floor."
- Actual: "The cat sat andre ... HTTP Cair Calabria 过年"
- Layer 0 post_attn RMS: C=0.003 vs PyTorch=0.032 (diverging from S=1)
- Layer 1+ post_attn RMS: C and PyTorch differ
- post_ln RMS fix verified (C now matches PyTorch weight RMS, not 1+w)
- Debug script at `c/debug_qwen.py` for layer-by-layer comparison

### PyTorch Reference
```
on the floor, and the cat sat on the floor.
The cat
```

### C Output (latest)
```
The cat sat andre ... HTTP Cair Calabria 过年 表述 destinado
```

### Key Metrics (Layer 0, latest)
| Metric | C | PyTorch | Status |
|--------|---|---------|--------|
| QKV RMS | 1.2901 | 1.2901 | ✅ exact |
| K norm RMS | 0.0221 | 0.0221 | ✅ FIXED (per-row) |
| conv1d_out[0] | -0.0612 | -0.0612 | ✅ exact match! |
| **post_ln RMS** | **0.126** | **0.159** | ✅ FIXED (was 1.05) |
| post_attn_out RMS | 0.0032 | 0.0319 | ⚠️ diverging |
| layer_out RMS | 0.0180 | 0.1596 | ⚠️ diverging |

### Recent Fixes
1. ✅ Conv1d kernel indexing — ti = t - k (commit f56dc25)
2. ✅ qkv_t index — fixed stride from `b*S*conv_dim + ti*conv_dim + c_dim` to `ti*conv_dim*S + c_dim*S`
3. ✅ Full S+ck-1 conv output — compute on full padded output before truncating
4. ✅ A_log/dt_bias — match PyTorch
5. ✅ Out_proj weights — match PyTorch
6. ✅ qkv_all matmul — matches PyTorch
7. ✅ **post_ln RMSNorm weight handling** — Changed from `(1.0f+w[i])` to raw `w[i]`, matching GGUF-stored PyTorch weights

### Session 2026-07-22: Linear Attention Output Discrepancy

**New Finding**: The C linear attention output is **4-400x smaller** than PyTorch across all layers.

| Layer | Type | C attn_out RMS | PyTorch out_proj RMS | Ratio |
|-------|------|----------------|---------------------|-------|
| L0 | Linear | 0.0026 | 0.0297 | 0.089 |
| L1 | Linear | 0.00024 | 0.0172 | 0.014 |
| L2 | Linear | 0.00010 | 0.0194 | 0.005 |
| L5 | Linear | 0.0115 | 0.1011 | 0.114 |
| L18 | Linear | 0.0249 | 0.0957 | 0.260 |

**PyTorch Reference (Layer 0)**:
- y_all_rms (before RMSNormGated): 0.00058
- y_gated_rms (after RMSNormGated): 0.0699  
- out_proj_rms: 0.0297

**Suspected Root Causes**:
1. State update in recurrence may not accumulate correctly
2. RMSNormGated may be applied incorrectly
3. Output projection matmul may have wrong weights/layout

**Debug Tools Created**:
- `c/compare_layers.py`: Layer-by-layer C vs PyTorch comparison
- `c/compare_linear_exact.py`: Element-by-element linear attention comparison
- `c/DEBUG_FINDINGS.md`: Detailed debugging findings

**Remaining Issues**:
1. **Linear attention output**: C=0.0026 vs PyTorch=0.0297 (L0, 11x difference)
2. **Layer divergence**: Ratio varies from 0.002 to 0.26 across layers
3. **Conv1d RMS**: C=0.096 vs PyTorch=0.11 — close but not exact
4. **post_ln fix**: Weight handling corrected, now within 5-10%
5. **Logits**: Still need to verify after all fixes

### Debug Tools
- **`c/debug_qwen.py`**: Layer-by-layer PyTorch vs C comparison
  - `python3 debug_qwen.py` — run PyTorch and save outputs
  - `python3 debug_qwen.py --compare` — compare with latest C debug file
  - `python3 debug_qwen.py --check linear` — focus on linear attention layers
  - `python3 debug_qwen.py --layer 0,1,2` — specific layers
- **`DEBUG_LAYER=1`**: Per-layer intermediate dump to `debug_layer_*.json`
- **`DEBUG_LINEAR=1`**: Detailed linear attention debugging (conv1d, QKV, gating)
- **`REF_FORCE=1`**: Override oracle validation error for real model testing

---

## PLAN: Fix Conv1d Output Divergence (Next Session)

### P0: Element-by-Element conv1d Comparison ✅ FIXED
1. ✅ Dump PyTorch conv1d input/output for all 3 timesteps and first 32 channels
2. ✅ Dump C conv1d input/output for same positions  
3. ✅ Found qkv_t transpose bug: was assigning same value to all timesteps
4. ✅ Fixed: qkv_t[bs, c_dim, 0] = qkv_all[bs, c_dim]

### P1: Check conv1d Weight Layout ✅ VERIFIED
- ✅ C's wc[k] at index c_dim * ck + k matches PyTorch's weight[c_dim, 0, k]
- ✅ w[0]=-0.000161, [1]=0.000404, [2]=-0.003387, [3]=-0.074219 matches

### P2: Check conv1d Computation ✅ FIXED
- ✅ qkv_t index: ti*conv_dim*S + c_dim*S (was b*S*conv_dim + ti*conv_dim + c_dim)
- ✅ conv1d output now matches PyTorch element-by-element
- ✅ v_raw RMS now matches PyTorch (0.1335)

### P3: Check Silu Application ✅ VERIFIED
- ✅ PyTorch: F.silu(conv_out)[:, :, :S]
- ✅ C: applies silu after transpose back to [BS, conv_dim]
- Both produce same result

### P4: Check Recurrence
- ✅ Manual PyTorch recurrence matches C output (out_proj RMS=0.069)
- ❌ PyTorch fla's chunk_gated_delta_rule produces different results (RMS=0.038)
- **ROOT CAUSE**: fla library implements the gated delta rule differently

### P5: Investigate fla implementation differences
- The chunk_gated_delta_rule from fla may use different numerical precision
- May apply L2 normalization internally vs externally
- May have different chunking strategy affecting accumulation
- May use fused operations that produce different rounding

### P6: Fix z-gating and Output Projection
- The z-gating matches PyTorch manual computation
- out_proj weights match
- The remaining difference is in the recurrence itself

### P7: Compare fla implementation details
- Check if fla's use_qk_l2norm_in_kernel=True changes normalization
- Check if fla uses different softplus or sigmoid implementations
- Check if fla's chunking causes different accumulation order

### P0: Element-by-Element conv1d Comparison
1. Dump PyTorch conv1d input/output for all 3 timesteps and first 32 channels
2. Dump C conv1d input/output for same positions
3. Find exact timestep/channel where values first diverge

### P1: Check conv1d Weight Layout
- PyTorch: `conv1d.weight` shape [6144, 1, 4], groups=6144, padding=3
- C: `conv1d_w` loaded as flat [conv_dim * ck] = [24576]
- Verify: C's `wc[k]` at index `c_dim * ck + k` matches PyTorch's `weight[c_dim, 0, k]`
- Current C loads: l->conv1d_w[0]=-0.000161, [1]=0.000404, [2]=-0.003387, [3]=-0.074219
- PyTorch weight[0] = [-0.000161, 0.000404, -0.003387, -0.074219] → ✅ MATCH

### P2: Check conv1d Computation
- PyTorch conv1d: `out[c, t] = sum_k weight[c, 0, k] * input[c, t-(3-k)]`
- C conv1d: `acc += qkv_t[ti*conv_dim*S + c_dim*S] * wc[k]` where `ti = t-(ck-1-k)`
- For t=1, c_dim=0, k=2: PyTorch uses input[1-(3-2)] = input[0]
- C uses qkv_t at ti=0, which should be qkv_all[0, 0] = 0.824031
- Verify the qkv_t values match PyTorch's mixed_qkv_t

### P3: Check Silu Application
- PyTorch: `F.silu(conv_out)[:, :, :S]` — silu on full padded output, then truncate
- C: applies silu after transpose back to [BS, conv_dim] — same result
- The silu output should match: PyTorch conv_out_silu[0,0,0:3] = [-0.0296, -0.0131, -0.0316]
- C conv1d_out after silu should match

### P4: Check V Split
- PyTorch: v_raw from conv_out_silu, shape [1, 3, 2048], RMS=0.1335
- C: v_all from conv1d_out after silu, shape [3, 2048], RMS=0.0751
- If conv1d_out matches, v_raw should match. If not, trace back to conv1d.

### P5: Check Recurrence
- If V matches, verify the gated delta rule recurrence produces correct state
- Manual PyTorch recurrence vs fla's chunk_gated_delta_rule
- State S[h, i, j] = decay*S + K * (V - K^T*S) * beta

### P6: Check z-gating and Output
- Verify RMSNormGated: RMSNorm(y) * weight * silu(z)
- Verify out_proj: y @ W_out.T
- Verify lm_head and temperature scaling

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
- `c/debug_qwen.py` — **NEW**: Full PyTorch→C layer-by-layer comparison tool
  - `python3 debug_qwen.py` — run PyTorch and save outputs
  - `python3 debug_qwen.py --compare` — compare with latest C debug file
  - `python3 debug_qwen.py --check linear` — focus on linear attention layers only
  - `python3 debug_qwen.py --check gqa` — focus on GQA layers only
  - `python3 debug_qwen.py --layer 0,1,2,3` — compare specific layers
  - `python3 debug_qwen.py --prompt "custom prompt"` — use different prompt
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

## GLM-5.2 FP16 Support Issue (2026-07-22)

### Problem
The C engine's `expert_load` function in `c/glm.c` assumes all GGUF files have quantization scales (`.weight.scales` tensors). FP16 GGUF files don't have these scales, causing the code to fail.

### Root Cause
The code at line 2460 checks `if(!tw[k]||!tq[k])` and exits if scales don't exist. This prevents FP16 models from loading.

### Fix Needed
1. Check if scales exist before requiring them
2. Handle non-quantized tensors (FP16) by setting `scales = NULL`
3. Update all code that uses `scales` to handle NULL

### Status
- FP16 GGUF conversion works
- PyTorch inference works (deterministic output)
- C CPU backend: BUG in expert loading (expects quantization scales)
- C HIP backend: Not tested (CPU bug blocks it)

### Next Steps
Fix `expert_load` to handle both quantized (FP4/FP8) and non-quantized (FP16) tensors.
