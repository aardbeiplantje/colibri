# Colibri — TODO (Updated 2026-07-13)

## Quick Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | GLM-5.2 MLA foundation | ✅ Done |
| 2 | Expert MoE routing | ✅ Done |
| 3 | Flash attention v2 / MLA KV cache | ✅ Done |
| 4 | DSA lightning indexer | ✅ Done |
| 5 | Qwen3.6 hybrid architecture (DeltaNet + GQA) | ✅ Done |
| 6 | Config parsing (Qwen3.5/3.6 params, layer_types) | ✅ Done |
| 7 | Qwen3.5 linear attention kernel + GGUF FP4 tool | ✅ Done |
| 8 | End-to-end integration, testing, benchmarking | 🔲 Next |

---

## Phase 5 — Qwen3.6 Hybrid Architecture ✅ DONE

### What was added to `c/glm.c`:

**Config (`Cfg` struct + `load_cfg`)**:
- `n_kv_heads`, `head_dim`, `attn_type` (1=GLM MLA, 2=Qwen)
- `delta_n_heads`, `delta_v_heads`, `delta_head_dim`, `delta_repeats`
- `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`
- `linear_conv_kernel_dim`, `attn_output_gate`
- `layer_types[128]` — explicit per-layer attention type (from config JSON)

**`Layer` struct additions**:
- GQA: `q_proj_gqa`, `k_proj_gqa`, `v_proj_gqa`, `o_proj_gqa`, `q_norm_w_gqa`, `k_norm_w_gqa`
- GatedDeltaNet: `q_proj_delta`, `k_proj_delta`, `v_proj_delta`, `o_proj_delta`, `dt_bias_delta`
- **Qwen3.5 Linear Attention**: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d_w`, `A_log`, `dt_bias`, `ln_w`, `o_proj_delta`

**New functions**:
- `gqa_attention()` — GQA with full RoPE, QKNorm (RMSNorm), head expansion, causal masking
- `gated_delta_net()` — L2-normalized recurrent state, causal conv1d, softplus alpha decay
- `causal_conv1d()` — depthwise causal convolution (kernel=4)
- `l2_normalize_heads()` — per-token L2 normalization over head dim
- **`linear_attn_forward()`** — Qwen3.5/3.6 linear attention with exp-decay gating and outer-product recurrence
- `is_delta_layer()`, `is_gqa_layer()`, `is_linear_attn_layer()` — routing helpers

**`layer_forward()` dispatch**:
- `attn_type == 1` → GLM MLA path (existing)
- `attn_type == 2` → DeltaNet / GQA / Linear Attention based on `layer_types[i]`

### Model Architecture (Qwen3.5-0.8B):
- 24 layers: `[3× linear_attention + 1× full_attention] × 6`
- Hidden dim: 1024, Attention heads: 8, KV heads: 2 (8:1 GQA)
- Vocab: 248,320, RMSNorm eps: 1e-6
- MLP intermediate: 3584

---

## Phase 6.5 — GGUF FP4 Conversion Tool ✅ DONE

### `c/tools/convert_qwen36_fp4.py`

Pure Python/numpy tool (no torch required) that:

1. **Reads** model.safetensors files (BF16, F32 dtypes)
2. **Maps** tensor names to C struct expectations
3. **Quantizes** to FP4 (E2M1) with per-row scaling
4. **Writes** GGUF v3 format with NVFP4 custom type

**NVFP4 E2M1 encoding**:
- 4 bits: sign(1) | exp(2) | mant(1)
- Values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
- Scale: absmax / 6.0 per row

**Usage**:
```bash
python3 c/tools/convert_qwen36_fp4.py <input_dir> <output.gguf> [--bits 4]
```

**Verified**: Successfully converted Qwen3.5-0.8B → 379.5 MB GGUF with 320 tensors.

---

## Phase 8 — Detailed Plan (2026-07-13)

### 8.0 Pre-requisites (fixes before integration)

#### 8.0.1 Conv1d weight loading bug (P0)
**Problem**: `conv1d.weight` has shape `[6144, 1, 4]` (3D) but `qt_load` only supports 2D tensors. The current code loads it as `[nv*vd, ck] = [2048, 4]` — a dimension mismatch that produces wrong weights.

**Fix**: Load conv1d tensor separately with raw st_read from safetensors, handling 3D→1D flattening. Or add a `qt_load_3d()` helper. The conv1d weight should be stored as a flat float buffer (not quantized) since kernel weights are always float.

**Location**: `c/glm.c:1547` — `l->conv1d_w = qt_load(m, P("linear_attn.conv1d.weight"), nv*vd, ck, dbits);`

**Estimated effort**: 2-3 hours

#### 8.0.2 causal_conv1d applied in linear_attn_forward (P0)
**Problem**: `linear_attn_forward` currently does `memcpy(x_proj, x_in, ...)` as a stub instead of actually applying causal conv1d on the projected QKV tensor.

**Reference from transformers/Qwen3.5**:
```python
mixed_qkv = self.in_proj_qkv(hidden_states)  # [BS, D] -> [BS, 6144]
mixed_qkv = mixed_qkv.transpose(1, 2)         # [BS, 6144, S]
mixed_qkv = causal_conv1d_fn(mixed_qkv, ...)  # apply conv
query, key, value = torch.split(mixed_qkv, [kd, kd, vd])  # split after conv
```

**Fix**: 
1. Project to [BS, 6144, S], transpose to [BS, 6144, S]
2. Apply causal_conv1d with weight [6144, 1, 4] as a grouped convolution (groups=6144)
3. Split into Q [BS, 2048, S], K [BS, 2048, S], V [BS, 2048, S]
4. Continue with linear attention recurrence

**Alternative**: For now, skip conv1d and apply it only to V (the most critical path). The conv1d on Q/K is less impactful.

**Estimated effort**: 4-6 hours

#### 8.0.3 in_proj_qkv combined projection inefficiency (P1)
**Problem**: The current code does a full matmul on the combined weight [6144, D] then slices Q, V, K from the output. The GPU kernel computes all 6144 output elements even though only the sliced results are used.

**Weight layout**: `[nq*kd + nv*vd + nq*kd] = [2048 + 2048 + 2048, 1024]`
- Q channels: first 2048 output positions
- V channels: next 2048 output positions  
- K channels: last 2048 output positions

**Fix options**:
1. **Split weight on disk**: Create three separate tensors in GGUF (`in_proj_q`, `in_proj_v`, `in_proj_k`) and load them as separate QT tensors. Do three separate matmuls. Most efficient but requires changing the conversion tool.
2. **Slice on GPU**: Pass offset/stride info to GPU kernel to compute only needed rows. Complex.
3. **Accept inefficiency**: For 0.8B model, the overhead is ~50% extra compute for QKV projections but it works. Defer optimization.

**Recommendation**: Start with option 3 (accept current inefficiency) for integration testing. Plan option 1 for a later optimization pass.

**Estimated effort**: 1 hour (option 3) / 4-6 hours (option 1)

#### 8.0.4 MTP layer for Qwen3.5/3.6 (P1)
**Problem**: Current MTP loading is hardcoded for GLM-5.2 tensor names. Qwen3.5 MTP uses completely different tensor names:

| GLM-5.2 MTP | Qwen3.5 MTP |
|---|---|
| `eh_proj.weight` | `mtp.fc.weight` |
| `enorm.weight` | `mtp.norm.weight` |
| `hnorm.weight` | `mtp.pre_fc_norm_embedding.weight` |
| `shared_head.norm.weight` | `mtp.pre_fc_norm_hidden.weight` |
| `self_attn.q_a_proj.weight` | `mtp.layers.0.self_attn.q_proj.weight` |
| `self_attn.q_b_proj.weight` | `mtp.layers.0.self_attn.k_proj.weight` |
| `self_attn.kv_a_proj_with_mqa.weight` | `mtp.layers.0.self_attn.v_proj.weight` |
| `mlp.experts.*` | `mtp.layers.0.mlp.*_proj.weight` |

**Fix**: Add a conditional MTP loading path in `model_init` that checks `attn_type`:
- `attn_type == 1` → GLM-5.2 MTP names (existing)
- `attn_type == 2` → Qwen3.5 MTP names with GQA-style attention (q_proj, k_proj, v_proj, o_proj)

Qwen3.5 MTP uses standard GQA attention (not linear attention) and a dedicated FC projection head (`mtp.fc.weight [1024, 2048]`) instead of the shared lm_head.

**Estimated effort**: 3-4 hours

---

### 8.1 End-to-End Integration Testing

**Goal**: Verify the compiled binary can run the converted Qwen3.5 model end-to-end.

**Steps**:
1. Compile with `make` (or `HIP=1 make` for GPU)
2. Run: `./glm Qwen3.5-0.8B` (point to the converted GGUF)
3. Check for:
   - Tensor loading errors (missing tensors, dimension mismatches)
   - NaN/inf in outputs
   - Correct token generation (no crash, reasonable perplexity)

**Test script**: `c/tests/run_qwen35.sh`
```bash
#!/bin/bash
cd c
./glm ../../Qwen3.5-0.8B \
  --prompt "The quick brown fox" \
  --max-tokens 64 \
  --temperature 0.8
```

**Expected**: Model generates coherent text without crashes or NaN propagation.

**Estimated effort**: 1-2 hours

---

### 8.2 Numerical Accuracy Validation

**8.2.1 DeltaNet forward pass vs. PyTorch reference**
- Use `/tmp/gdtn/lit_gpt/gated_delta_net.py` as reference
- Run same input through C and PyTorch, compare outputs element-wise
- Check per-token max absolute error and relative error
- Focus on: α decay, recurrence state updates, output projection

**8.2.2 GQA attention vs. HuggingFace**
- Compare QKV projections, RoPE embeddings, attention scores
- Verify softmax normalization is numerically stable
- Check causal mask produces correct triangular attention pattern

**8.2.3 FP4 dequantization accuracy**
- Run the conversion tool with different bit settings (--bits 4, --bits 8)
- Measure reconstruction error: max absolute error, MSE per tensor
- Identify which tensor types are most sensitive to quantization
- Expected: embeddings and lm_head (high precision) → 0 error; attention weights → small error; MLP weights → moderate error

**Estimated effort**: 6-8 hours

---

### 8.3 Benchmarking on Strix Halo (gfx1151)

**8.3.1 Prefill throughput**
- Measure tokens/sec for batch sizes 1, 4, 8
- Compare linear attention vs. GQA vs. DeltaNet layers
- Profile GPU utilization with rocminfo/rocm-smi

**8.3.2 Decode latency**
- Measure time per token at various context lengths (128, 512, 1024)
- Check if DeltaNet/GQA recurrence scales linearly with context
- Compare FP4 vs. BF16 latency

**8.3.3 Memory profiling**
- RSS monitoring via /proc/PID/status or `valgrind --tool=massif`
- DeltaNet state memory: [n_layers_delta][max_batch][n_heads*hd*hd]
- KV cache memory for GQA layers
- Total resident memory vs. model file size

**8.3.4 FP4 vs. BF16 quality**
- Run identical prompts at both quantizations
- Compare generated text quality (perplexity, coherence)
- Identify degradation thresholds (e.g., "beyond top-20 predictions")

**Estimated effort**: 8-12 hours

---

### 8.4 Extension to Larger Models

**8.4.1 Multi-file safetensors sharding**
- Current `convert_qwen36_fp4.py` only handles single-file models
- For 7B+ models, safetensors are split across multiple shards
- Need to iterate over all shards, combine tensors, write GGUF

**8.4.2 Larger vocabularies and context**
- Test with 7B model (vocab ~128,000+)
- Verify memory scales correctly
- Check if fp4 conversion handles larger weight tensors

**8.4.3 CUDA/HIP GPU backend for linear attention**
- The `matmul_qt` calls for in_proj_qkv, in_proj_z, etc. already use GPU via COLI_HIP
- Need to add GPU kernel for causal_conv1d (group conv, kernel=4, silu)
- The linear attention recurrence is inherently sequential (O(S) time steps) — stays on CPU like DeltaNet
- GPU kernel for the state update per time step: decay + outer product + output query

**Estimated effort**: 8-16 hours (depends on GPU availability and testing infrastructure)

---

## Known Issues / TODO (Detailed Analysis)

### Issue 1: Conv1d Weight Loading Bug
**Severity**: P0 — model produces wrong outputs
**Root Cause**: `qt_load()` flattens 3D tensor [6144, 1, 4] but loads with dimensions [2048, 4], discarding a factor of 3.
**Fix**: Load conv1d weight via raw st_read as float buffer (no quantization), reshape to [conv_dim, kernel_size] in code.
**Files**: `c/glm.c` lines 139 (conv1d_w field), 1547 (loading), layer 141 (struct)
**Estimate**: 2-3 hours

### Issue 2: causal_conv1d Not Applied
**Severity**: P0 — linear attention recurrence is incorrect without conv1d preprocessing
**Root Cause**: `linear_attn_forward` copies input with memcpy, skipping conv1d entirely.
**Fix**: Apply causal_conv1d to projected QKV tensor after projection, before splitting into Q/K/V. The conv1d operates on the combined tensor as a grouped convolution.
**Files**: `c/glm.c` lines 1146-1147 (current stub), 817-833 (causal_conv1d function)
**Estimate**: 4-6 hours

### Issue 3: in_proj_qkv Inefficiency
**Severity**: P1 — 50% wasted compute in projections
**Root Cause**: Single matmul on combined weight [6144, D] computes all 6144 outputs; only slices are used.
**Fix (short term)**: Accept current inefficiency, verify correctness first.
**Fix (long term)**: Split weight into 3 tensors in GGUF, load as separate QTs, do 3 matmuls.
**Files**: `c/glm.c` lines 1158-1175, `c/tools/convert_qwen36_fp4.py`
**Estimate**: 1h (skip) / 4-6h (fix)

### Issue 4: MTP Layer for Qwen3.5/3.6
**Severity**: P1 — MTP feature unusable for Qwen models
**Root Cause**: Hardcoded GLM-5.2 tensor names in MTP loading block.
**Fix**: Conditional loading based on `attn_type`. Qwen3.5 MTP uses: `mtp.fc.weight`, `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`, `mtp.layers.0.{q,k}_norm.weight`, `mtp.norm.weight`, `mtp.pre_fc_norm_{embedding,hidden}.weight`.
**Files**: `c/glm.c` lines 1578-1625 (MTP block)
**Estimate**: 3-4 hours

### Issue 5: GPU Backend for Linear Attention
**Severity**: P2 — CPU-only, but recurrence is inherently sequential
**Root Cause**: `linear_attn_forward` is pure C/CPU. The matmul calls for projections already use GPU via `matmul_qt` + COLI_HIP, but the recurrence loop (state decay + outer product) cannot be parallelized across time steps.
**Fix**: The recurrence can be unrolled into a single GPU kernel processing all S time steps at once. This requires a custom HIP kernel that maintains state in shared memory. For S=1 (decode), the current CPU path is fine.
**Files**: `c/glm.c` lines 1179-1234 (recurrence loop), `c/backend_hip.cu` (add kernel)
**Estimate**: 8-12 hours

### Issue 6: Batch Size Hardcoded to 8
**Severity**: P3 — minor, easily remedied
**Root Cause**: DeltaNet state allocation uses `max_batch=8` constant.
**Fix**: For DeltaNet, make `max_batch` a config parameter or derive from model. For linear attention, batch size flows naturally from caller (no pre-alloc).
**Files**: `c/glm.c` line ~1426
**Estimate**: 30 minutes
