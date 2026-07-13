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
| 6.5 | GGUF FP4 conversion tool (`c/tools/convert_qwen36_fp4.py`) | ✅ Done |
| 7 | Qwen3.5 linear attention kernel | ✅ Done |
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

## Phase 8 — Next Steps (TODO)

### 8.1 End-to-End Integration
- [ ] Test the compiled binary against the converted GGUF model
- [ ] Verify token generation on Qwen3.5 model
- [ ] Check that linear attention forward pass produces valid outputs

### 8.2 Numerical Accuracy
- [ ] Compare DeltaNet forward pass against PyTorch reference
- [ ] Validate GQA attention scores match HuggingFace implementation
- [ ] Test FP4 dequantization accuracy (check reconstruction error)

### 8.3 Benchmarking (Strix Halo / gfx1151)
- [ ] Measure prefill throughput (tokens/sec) for linear attention layers
- [ ] Measure decode latency per token
- [ ] Profile memory/RSS vs. BF16 baseline
- [ ] Compare FP4 vs. BF16 accuracy degradation

### 8.4 Extension to Qwen3.6 (larger models)
- [ ] Support multi-file safetensors sharding
- [ ] Handle larger vocabularies and context lengths
- [ ] Add MTP (Multi-Token Prediction) support if applicable

---

## Known Issues / TODO

- [ ] The `in_proj_qkv` combined projection needs proper slicing in `linear_attn_forward` — currently does a full matmul then copies, which is inefficient
- [ ] `causal_conv1d` in `gated_delta_net` currently just copies input (kernel applied in weight projections)
- [ ] MTP layer support for Qwen3.5/3.6 models (currently only for GLM-5.2)
- [ ] CUDA/HIP GPU backend for linear attention (CPU-only for now)
- [ ] Batch processing support (max_batch=8 hardcoded in DeltaNet)
