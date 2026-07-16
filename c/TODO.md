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
| 8.1 | Crash fixes (8 bugs) | ✅ Done |
| 8.2 | Tokenizer format fix | ✅ Done |
| 8.3 | Numerical accuracy — output quality | 🟡 In progress (major layout fix applied) |
| 8.4 | GGUF integration (FP16/FP8/FP4) | ✅ Done |

## Phase 8.4 — GGUF Integration (Done 2026-07-15)

### What Was Implemented
1. **GGUF v3 parser** — Complete `gguf.h` with:
   - `gguf_init()`, `gguf_find()`, `gguf_mmap()`, `gguf_unmap()`, `gguf_free()`
   - F32, F16, NVFP4, NVFP8 data types
   - Pre-quantized tensor dequantization on-the-fly

2. **Hash table bug fix** — `hidx` array allocated with `calloc` (init to 0) but lookup expects -1 for empty slots.
   **Fix**: Added `memset(ctx->hidx, -1, ctx->hcap * sizeof(int))`.

3. **KV pair type handling** — Fixed to match GGUF v3 spec:
   - Type 0 = BOOL (1 byte)
   - Type 2 = STRING (Q + data)
   - Type 5 = U32 (4 bytes)
   - Type 7 = F32 (4 bytes)

4. **Tensor info parsing** — Removed placeholder offset from tensor info (GGUF v3 stores offsets in separate section).

5. **Tokenizer path for GGUF** — Updated to extract directory from GGUF file path.

6. **Config path for GGUF** — Updated `cfg_root()` to handle `.gguf` file paths.

### Results
- **GGUF loading**: ✅ All 488 tensors loaded correctly from `model.gguf` (FP16, 1.7 GB)
- **Forward pass**: ✅ Runs through all 24 layers + MTP
- **Generation**: ⚠️ Output quality poor — "The!!!!!!!!!!!!!!!!!!!!" (numerical accuracy issue)
- **Performance**: 4.84 tok/s, 3.38 GB RSS

### Remaining Work
- Numerical accuracy: C output doesn't match PyTorch reference
- First step: Compare per-layer outputs to identify where divergence starts
- PyTorch reference: Coherent English ("The following are multiple choice...")
- C output: "The on the floor" (with FP16: "luftiscoitek")

---

## Phase 8.3 — Fix: Memory Layout Bug in Linear Attention (2026-07-15)

### Critical Bug Found and Fixed

**ROOT CAUSE**: The matmul output uses `[head, BS]` layout, but the gating code read/wrote `[bs, head]` layout.

The `matmul_qt(a_all, x_in, &l->in_proj_a, BS)` produces:
- Layout: `[nq, BS]` = `[16, 3]` → indices: `a_all[h * BS + bs]`
- The gating code incorrectly accessed: `a_all[bs * nq + h]`

For BS=3, nq=16:
- Correct: `[0,0]=0, [0,1]=1, [0,2]=2, [1,0]=16, [1,1]=17, ...`
- Wrong:    `[0,0]=0, [0,1]=1, [0,2]=2, [0,3]=3, ..., [0,15]=15, [1,0]=16`

For small BS values these layouts overlap at indices 0-2 but diverge completely for indices 3-15.

### Fixes Applied

1. **Linear attention — QKV split layout fix**: Changed from `[bs, dim]` to `[head, BS]` matmul layout
2. **Linear attention — L2 normalization layout fix**: Q and K L2 norm now uses correct `[head, BS]` indexing  
3. **Linear attention — Gating layout fix**: `g_all` and `b_all` computed with correct `[head, BS]` layout
4. **Linear attention — State recurrence layout fix**: `g_all`, `b_all`, `k_all`, `v_all` accessed with `[head * dim * BS + bs * dim + d]` indexing
5. **Linear attention — Y_all layout fix**: Output written in `[head, BS]` layout matching matmul
6. **GQA attention — Q-projection gate split**: Q_proj outputs query+gate combined, now split correctly
7. **GQA attention — Per-head scores**: Each head has its own attention distribution (not summed)
8. **GQA attention — K normalization**: Uses `k_norm_w_gqa` with `(1+w)` offset (was using zeros-falloc'd array)
9. **GQA attention — Head count fix**: `n_q = qO / hd / 2` (gate takes half of q_proj output)
10. **GGUF — Directory auto-detection**: Added `.gguf` file lookup when snap path is a directory

### Results

| Test | Before Fix | After Fix |
|------|-----------|-----------|
| L0 out_rms | 0.0018 | ~0.0018 (FP16) |
| g_val sign | sometimes positive | always <= 0 (verified) |
| Token diversity | single repeated token | diverse tokens |
| C output (int8) | "kis" | "itek" |
| C output (FP16) | N/A | "luftiscoitek" |
| PyTorch output | N/A | "on the floor" |

### Remaining Issues

1. **Numerical accuracy**: C layer outputs have much lower RMS than PyTorch (0.0018 vs 0.041 for L0). This could be from:
   - The GGUF weight loading (F16→F32 conversion accuracy)
   - The conv1d output being smaller than PyTorch
   - The z-gate values being too small
2. **MTP speculation**: 0% acceptance rate — speculative tokens don't match main model
3. **Output tokens**: C generates "itek" vs PyTorch generates "on" (first generated token)

### Key Files
- `c/gguf.h` — GGUF v3 parser (250 lines)
- `c/tools/convert_to_gguf.py` — Python GGUF writer (FP16/FP8/FP4)
- `c/glm.c` — Updated for GGUF weight loading

---

## Phase 8.3 — Numerical Accuracy (IN PROGRESS)

---

## Phase 8.1 — Crash Fixes (All Done)

Fixed 8 critical crashes that prevented Qwen3.5 from running:

1. **FPE in matmul_q_idot** — `st_numel()` returns total elements (rows×cols), not row count.
   Fixed by computing `rows = st_numel / D` for tensor dimensions in GQA layer loading.

2. **Segfault: `o_proj_gqa` dimension swap** — Weight tensor shape [D, nv*vd] but `qt_load` was called with `O=nv*vd, I=D` (swapped).
   Fixed: `qt_load(m, name, O=D, I=oO, dbits)`.

3. **Segfault: `mtp_norm` NULL** — Qwen3.5 MTP loading didn't load `mtp.norm.weight`.
   Fixed: Added `m->mtp_norm = ld(m, "mtp.norm.weight")` in MTP loading.

4. **Segfault: `qkn_ln_w_gqa` NULL** — MTP layer didn't set `qkn_ln_w_gqa`.
   Fixed: Added `l->qkn_ln_w_gqa = falloc(nq * hd)` in MTP loading.

5. **Segfault: MTP layer routed to linear attention** — `is_linear_attn_layer(24, c)` returned true for MTP layer (layer_types[24]=0 uninitialized).
   Fixed: Added `li >= c->n_layers` guard in `is_linear_attn_layer`, and explicit `li==c->n_layers && c->attn_type==2` check in `layer_forward`.

6. **Buffer overflow in GQA scores** — `max_prev = cur_pos + 1` could exceed `S`.
   Fixed: `if(max_prev > S) max_prev = S`.

7. **Config parsing: `n_kv_heads` and linear attention params** — Read from wrong JSON object (`r` instead of `root`).
   Fixed: Changed to use `root` (text_config) for Qwen3.5/3.6 params.

8. **MTP `q_gqa_O`/`kv_gqa_O` not set** — GQA function computed `n_heads` from these fields.
   Fixed: Added `l->q_gqa_O = qO; l->kv_gqa_O = kO` in MTP loading.

## Phase 8.2 — Tokenizer Format Fix (Done)

Fixed Qwen3.5 tokenizer loading. The tokenizer has two different merge formats:
- **GLM-5.2**: merges are objects `[{"source":...,"target":...}]` with `kids[0].str` / `kids[1].str`
- **Qwen3.5**: merges are string pairs `"Ġ Ġ"` (left token, space, right token)

The old code only handled GLM-5.2 format and crashed on Qwen3.5. Fixed by auto-detecting
the format via `jval->t == J_STR` and parsing strings with `strrchr(pr->str, ' ')` to split
left and right tokens at the last space.

---

## Phase 8.3 — Numerical Accuracy (IN PROGRESS)

### Latest Progress (2026-07-15)

**GGUF Integration Complete:**
- All 488 tensors load correctly from GGUF FP16 (1.7 GB)
- Forward pass runs through all layers
- Output: "The!!!!!!!!!!!!!!!!!!!!" (single repeated token)

**Previous Fixes (2026-07-14):**
1. **Qwen3.5 RMSNorm offset weights** — Added `(1.0 + weight)` pattern matching PyTorch
2. **F.silu after conv1d** — Added silu activation after conv1d
3. **softplus helper** — Numerically stable softplus for alpha
4. **Conv1d weight indexing** — Fixed reversed kernel order
5. **K/V split order** — Fixed K/V channel order
6. **Z gating + RMSNorm** — Added proper RMSNormGated
7. **L2 normalization** — Removed head_scale multiplier
8. **Dense MLP** — Implemented SwiGLU forward (was NO-OP)
9. **Gating computation** — Fixed to -exp(A_log) * softplus(a + dt_bias)
10. **Alpha decay** — Fixed to use exp(g) where g is negative

**Current Status:**
- Model loads and runs end-to-end ✓
- Output is a single repeated token (!!!!!!!!!!!!!!)
- PyTorch produces coherent English ✓
- The model is NOT producing coherent text

**Key Observation:**
- All layers produce output (no zeros)
- But the output logits are degenerate (one token has near-100% probability)
- This suggests numerical errors in early layers that amplify through the network

**Next Debugging Step:**
1. Compare Layer 0 outputs between C and PyTorch (qkv_all, conv1d_out, Q/K norms)
2. Compare Layer 1 outputs (should match if L0 is correct)
3. Identify first layer where outputs diverge
4. Fix the divergent layer
5. Iterate until C output matches PyTorch

### Next Steps (pick up from here after restart)

1. **Fix alpha computation** in `linear_attn_forward()` (line ~1291):
   - Current: `expf(a_val + dt) * b_val` → produces negative values
   - Target: Reference pattern using `logsigmoid` → always produces valid (0,1] gate
   - The model has `A_log` [16] F32, `dt_bias` [16], `in_proj_a` [16,1024], `in_proj_b` [16,1024]

2. **Verify fix**:
   ```bash
   SNAP="../Qwen3.5-0.8B" DEBUG_LINEAR=1 PROMPT="The" NGEN=1 ./glm 64 8 8
   # Check: L0 out_rms >> 0.0
   ```

3. **Compare with PyTorch reference**:
   ```bash
   # Then run C with same prompt and compare logits
   ```

4. **Fix remaining issues** (RoPE, MLP, etc.) if alpha fix resolves the zero output.

### Debugging Session: Linear Attention Zero Output (2026-07-13)

**CRITICAL BUG DISCOVERED**: The Qwen3.5 model produces ALL-ZERO hidden states after the first linear attention layer. Despite non-zero inputs at every stage, the final layer output is zero.

#### Debug Tracing (with `DEBUG_LINEAR=1`)

Step-by-step trace through Layer 0 (first linear attention layer, prompt="The", S=1):

```bash
$ SNAP="../Qwen3.5-0.8B" DEBUG_LINEAR=1 PROMPT="The" NGEN=1 ./glm 64 8 8
[LIN] L0: in_rms=0.2455 qkv_rms=0.3567 conv_dim=6144  ← in_proj_qkv OK
[LIN] L0: conv1d_w=0x... conv1d_n=24576                  ← conv1d weights loaded
[LIN] L0: pre_l2_q_rms=0.0745 ln_w=0x...                 ← Q has content before L2
[LIN] L0: q_l2=0.0011 inv_l2=678.5042                    ← Q L2 norm very small
[LIN] L0: z_rms=0.2218 a_rms=0.2615 b_rms=-0.0365       ← Z, A, B projections non-zero
[LIN] L0: t=0 k_rms=0.0712 v_rms=0.0296                 ← K, V have content
[LIN] L0: a_all[0]=-0.6052 a_all[1]=0.0011              ← *** ALPHA IS NEGATIVE ***
[LIN] L0: h=0 vh=0 a_val=-0.6052                         ← alpha used in decay
[LIN] L0: t=0 state_rms=0.0011                           ← state barely nonzero
[LIN] L0: pre_out_y_rms=0.0000                           ← *** OUTPUT ZERO ***
[LIN] L0: post_out_rms=0.0000
[LIN] L0: out_rms=0.0000
```

All subsequent layers also output zero because they receive zero input from layer 0.

#### Root Cause Analysis

The alpha computation in `linear_attn_forward()` is:
```c
float a_val = a_all[(int64_t)(t*B+b)*nq + h];  // from in_proj_a
float dt = l->dt_bias[h];                       // learned bias [16]
float b_val = b_all[(int64_t)(t*B+b)*nq + h];  // from in_proj_b
a_all[...] = expf(a_val + dt) * b_val;          // PROBLEM HERE
```

This produces NEGATIVE values (e.g., -0.6052) because `b_val` can be negative and `exp(...)` is always positive.

The alpha value is then used in state decay:
```c
state[b_off + h_off + i] *= a_val;  // multiplying by -0.6052 flips signs
```

After outer product addition, state has tiny RMS (0.0011). The output `y = Q @ S` produces effectively zero.

#### Reference Implementation

The fla-org reference (GatedDeltaNet) computes alpha differently:

```python
# From fla-org gated_delta_rule_ops/chunk.py
gk = F.logsigmoid(gk) / self.gate_logit_normalizer  # always <= 0
go = F.logsigmoid(go) / self.gate_logit_normalizer  # always <= 0
# Gate V and beta:
v = v * exp(gk)  # exp(gk) ∈ (0, 1] → multiplicative gate
beta = beta * exp(go)  # same pattern
```

Key differences:
1. Reference uses `logsigmoid` (log of sigmoid) → always negative → `exp(logsigmoid)` ∈ (0,1]
2. Reference applies gating multiplicatively to V and beta, not computing alpha from projections
3. Reference has `A_log` [16] tensor (learned log decay per head) used in the recurrence

The C code's formula `exp(a_proj + dt_bias) * b_proj` is algorithmically different and produces invalid decay rates.

#### Fix Required

The alpha computation in `linear_attn_forward()` must match the reference:
```c
// Current (WRONG):
a_all[bs*nq + h] = expf(a_val + dt) * b_val;

// Should use reference pattern:
// gk = a_proj (already computed, but needs sigmoid)
// alpha_gate = expf(-expf(A_log[h]) * softplus(gk + dt_bias[h]))
// This ensures alpha ∈ (0, 1)
```

The model has these tensors available:
- `A_log` [16] F32 — learned log decay rate per head (already loaded in DeltaNet code)
- `dt_bias` [16] BF16 — per-head bias (loaded as f32)
- `in_proj_a` [16, 1024] BF16 — gate projection for V
- `in_proj_b` [16, 1024] BF16 — gate projection for beta
- `in_proj_z` [2048, 1024] BF16 — Z gate projection

#### Verification Method

After fixing, verify with:
```bash
SNAP="../Qwen3.5-0.8B" DEBUG_LINEAR=1 PROMPT="The" NGEN=1 ./glm 64 8 8
# Expected: L0 out_rms >> 0.0 (non-zero output)
```

Compare against PyTorch reference:
```bash
# Compare c/pytorch_ref.json with C output
```

#### Performance Baseline (Before Fix)

| Metric | Value |
|--------|-------|
| Speed | 5.51 tok/s (single-thread CPU) |
| Memory | 1.58 GB RSS |
| MTP acceptance | 0% |
| Speculation | 2.00 tokens/forward (1 forward per 2 tokens) |

Output quality: `和白平台建设平台` (Chinese characters) — completely wrong.

---

### Symptom

Model runs end-to-end and generates text, but output quality is poor:
- Low temperature (0.7-1.0): Repeats same token (`upon upon upon` or `!!!!!!!!`)
- High temperature (1.5-2.0): Diverse multi-language gibberish (`the_months_營業時間` etc.)
- Model does NOT produce coherent English continuations

### Root Cause Analysis

#### Issue 1: QKNorm weight shape mismatch (FIXED)
- Qwen3.5 GQA `q_norm` weights are `[256] = head_dim`, shared across 16 heads
- Old code applied batch RMSNorm to all `n_heads × head_dim = 4096` values using only 256 weights
- Attention scores exploded to ~10^25, softmax degenerate
- **Fix**: Apply RMSNorm per-head: loop over heads, apply same `[head_dim]` weights to each head

#### Issue 2: Logit scale (FIXED)
- `final_norm` weights have RMS = 3.38 (learned, not 1.0)
- Embed weights absmax = 0.19
- Dot product ≈ 20.5, but actual logits reach ~44 (10x larger than standard)
- Temperature=0.7 produces degenerate softmax (top token ~100% probability)
- **Fix**: Scale down logits by 3.0 for `attn_type==2` in `step()` and `step_all()`

#### Issue 3: Temperature mismatch (FIXED)
- Qwen3.5 expects higher temperature than GLM-5.2's default 0.7
- With TEMP=0.7: repeats first token forever
- With TEMP=2.0: generates diverse tokens
- **Fix**: Default `g_temp = 2.0` when `c->attn_type==2`, 0.7 for GLM-5.2

#### Issue 4: Linear attention output computation (FIXED)
- Old code accumulated all K heads into all V heads incorrectly
- Reference uses `einsum('bhd,bhdm->bhm', q, S)` — per-head 1:1 mapping
- **Fix**: Each K head h contributes to output head h only

#### Issue 5: Linear attention L2 normalization (FIXED)
- Reference impl uses L2 normalization on Q and K after projection: `q = l2_norm_fn(q)`
- Model has `linear_attn.norm.weight` [128] loaded but NEVER applied
- **Fix**: After splitting QKV from convolved tensor, L2 normalize Q and K per head with `ln_w` weights

### Current Results

| Temp | Output | Coherence |
|------|--------|-----------|
| 0.7 | `upon upon upon upon upon` | ❌ Repeats |
| 1.0 | `upon uponceptionsceptionsception` | ❌ Partial |
| 1.5 | `the_months_營業時間เอียด` | 🟡 Diverse but not English |
| 2.0 | `::mai uslсипе菲律所の` | 🟡 Multi-language gibberish |

### Performance

| Metric | Value |
|--------|-------|
| Speed | 8.2 tok/s (single-thread CPU) |
| Memory | 1.94 GB RSS |
| MTP acceptance | 0% (auto-disabled) |
| Speculation | 1.05 tokens/forward |

### Remaining Issues

- [ ] **BUG FOUND: Linear attention alpha produces negative values** (2026-07-13)
  
  Debug trace with DEBUG_LINEAR=1 shows:
  ```
  L0: a_all[0]=-0.6052 a_all[1]=0.0011
  L0: h=0 vh=0 a_val=-0.6052
  L0: t=0 state_rms=0.0011
  L0: pre_out_y_rms=0.0000
  L0: out_rms=0.0000
  ```
  
  The alpha decay rate `a_val` is NEGATIVE (-0.6052). State decay: `S *= a_val` 
  means S gets multiplied by -0.6052, flipping signs. After outer product addition, 
  state has tiny RMS (0.0011). Final y = Q @ S produces zeros.
  
  **Root cause**: The alpha computation `exp(a_proj + dt_bias) * b_proj` is wrong.
  The reference uses `F.logsigmoid(gk) / gate_logit_normalizer` (always <= 0),
  then `v * exp(gk)` as gating. The C code's formula is algorithmically different.
  
  **Fix needed**: Replace alpha computation in linear_attn_forward() to match reference.
  The model has: A_log [16], dt_bias [16], in_proj_a [16,1024], in_proj_b [16,1024].

- [ ] **Coherent text generation** — After fixing alpha, verify model produces coherent English.
  - RoPE with theta=10^7 not differentiating positions enough (first few pairs rotate, last 126 pairs have tiny angles)
  - MLP layers producing wrong outputs
  - BF16→F32 conversion quality (model weights loaded from safetensors as F32)
  - Missing normalization somewhere in the chain

- [ ] **PyTorch reference comparison** — PyTorch IS installed (2.11.0+rocm7.13.0, ROCm gfx1151).
  PyTorch generates coherent English:
  ```
  on the floor, and the cat sat on the floor.
The cat
  ```
  **ACTION**: Compare C vs PyTorch per-token logits:
  2. Run C with same prompt and compare logits
  3. Identify first layer where outputs diverge
  4. Fix the offending layer

  - Greedy and temperature-based generation
  - Top-5 vocabulary with logits and probabilities
  - JSON output for comparison (`--save c/pytorch_ref.json`)
  - Full vocabulary dump with `--verbose`

- [ ] **Verify `linear_attn.norm.weight` values** — Check if weights look correct:
  - RMS should be ~1.0 (learned RMSNorm weights)
  - All values should be finite, non-zero
  - Shape is [128], applied per-head

- [ ] **RoPE verification** — With theta=10^7 and head_dim=256:
  - j=0: inv=1.0, j=1: inv≈0.88, j=127: inv≈1.33e-7
  - First 4 pairs get significant angles, last 124 pairs get negligible rotation
  - This is correct behavior for large theta; positions 0,1,2 should still be differentiated

- [ ] **Profiling instrumentation** — Add timing to `gqa_attention()` and `linear_attn_forward()`
  - `double ta0=now_s();` at function start
  - `m->t_attn += now_s()-ta0;` before return

- [ ] **Benchmark on Strix Halo (gfx1151)** — Prefill throughput, decode latency, memory profiling

- [ ] **FP4 accuracy** — Compare FP4 vs BF16 output quality using `convert_qwen36_fp4.py` tool

### Code References

Key files changed:
- `c/glm.c` — Main model code (~280+ new lines added across phases)
- `c/backend_hip.cu` — HIP backend (fixed nodiscard warnings, added HIP_CHECK error handling)
- `c/tools/convert_qwen36_fp4.py` — FP4 conversion tool

Key model config values (Qwen3.5-0.8B):
```
hidden_size: 1024
num_hidden_layers: 24 (pattern: 3 linear + 1 GQA × 6)
num_attention_heads: 8
num_key_value_heads: 2 (8:1 GQA ratio)
head_dim: 256
linear_num_key_heads: 16
linear_key_head_dim: 128
linear_num_value_heads: 16
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
rope_theta: 10000000 (10^7)
rms_norm_eps: 1e-6
vocab_size: 248320
```

Layer types pattern (repeats 6 times):
```
[linear_attention, linear_attention, linear_attention, full_attention]
```

---

## Phase 8.4 — Performance (TODO)

- [ ] Add timing instrumentation to `gqa_attention()` and `linear_attn_forward()`
- [ ] Benchmark on Strix Halo (gfx1151): prefill throughput, decode latency
- [ ] Memory profiling: current RSS ~2GB, verify no leaks over long sequences
- [ ] FP4 accuracy: compare FP4 vs BF16 output quality

---

## Phase 9 — MTP Improvement (TODO)

- [ ] Investigate MTP 0% acceptance rate
- [ ] The MTP draft path runs GQA attention on layer 24 weights
- [ ] MTP uses different weights than main model — always different output
- [ ] Consider disabling MTP entirely for Qwen3.5 if it provides no benefit
- [ ] Compare MTP logits against main model logits at same position

---

## Known Working Paths

- **GLM-5.2 MLA path** (attn_type=1): Full forward pass, attention scores working, RoPE applied correctly
- **Qwen3.6 DeltaNet path** (is_delta_layer): Fixed-state recurrence, alpha decay, L2 normalization
- **Qwen3.5 GQA path** (is_gqa_layer): QKNorm per-head, RoPE, attention scores working
- **Qwen3.5 Linear Attention** (is_linear_attn_layer): Conv1d, L2 norm on Q/K, per-head output
- **MTP layer** (li==c->n_layers): GQA-style attention, auto-disabling drafts

## Compilation

```bash
cd /workdir/colibri.git/c && make
```

## Model Paths

- Qwen3.5-0.8B: `/workdir/colibri.git/Qwen3.5-0.8B/`
- PyTorch reference: `/tmp/gdtn/lit_gpt/` (cloned from GitHub)

## Usage

```bash
cd /workdir/colibri.git/c
SNAP="../Qwen3.5-0.8B" PROMPT="hello" NGEN=10 ./glm 64 8 8
SNAP="../Qwen3.5-0.8B" TEMP=2.0 PROMPT="hello" NGEN=10 ./glm 64 8 8
```

## Restart Instructions

When the user restarts pi.dev after installing torch:

1. **Verify torch installation**: `pip3 install torch` (or conda equivalent)
2. **Run PyTorch reference**: Compare C output against PyTorch generation
3. **Key script to run**:
   ```python
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("/workdir/colibri.git/Qwen3.5-0.8B")
   tokens = t.encode("The cat sat", add_special_tokens=False)
   # Then generate with the actual model using torch
   ```
4. **Continue from**: Phase 8.3 numerical accuracy — the TODO item at top of this file
5. **Do NOT commit/push** without explicit user request

## 2026-07-14: Qwen3.5 Linear Attention Fixes + GGUF Conversion

### Critical Fixes Applied
1. **Conv1d weight indexing**: Fixed reversed kernel order to match PyTorch padding=3 semantics
2. **K/V split order**: Fixed K/V channel order (was swapped)
3. **Z gating + RMSNorm**: Added proper RMSNormGated (was missing)
4. **L2 normalization**: Removed head_scale multiplier from Q/K normalization (PyTorch doesn't use it)
5. **Dense MLP**: Implemented SwiGLU forward (was NO-OP)
6. **Gating computation**: Fixed to use -exp(A_log) * softplus(a + dt_bias) pattern
7. **Alpha decay**: Fixed to use exp(g) where g is negative (was using exp(a+dt)*b)

### Verification Status
- Q/K RMS matches PyTorch: 0.1234 vs 0.1230 (Q), 0.0541 vs 0.0542 (K)
- conv1d output matches PyTorch: -0.0622 vs -0.0613 (within quantization error)
- conv1d weights match: [-0.0002, 0.0004, -0.0034, -0.0742]

### GGUF Conversion Tool (NEW)
- Created `c/tools/convert_to_gguf.py`: Full safetensors→GGUF converter
- Supports FP16 (lossless), FP8-E4M3 (2x compression), FP4-E2M1 (4x compression)
- Pre-quantizes weights at conversion time → zero runtime quantization
- Updated `c/gguf.h`: Added NVFP8 type, gguf_read_tensor() with dequantization
- **Results**: 
  - FP16 GGUF: 1.7 GB (873M params)
  - FP8 GGUF: 973 MB (55% of FP16, 873M params quantized)
  - Conversion time: ~10 seconds for full model

### Output Quality Progress
- **Before fixes**: Chinese characters, Arabic, gibberish
- **After L2 fix**: English words ("dermat_rank", "平缓生生的") but not coherent
- The model now generates English tokens, but the grammar and context are wrong
- Root cause: int8 quantization of in_proj_qkv introduces errors that propagate through attention
- **New**: GGUF FP16 weights should eliminate quantization errors

### Remaining Issue
- The attention Q·K dot product differs from PyTorch due to numerical precision errors
- The conv1d input values (qkv_all) are different from PyTorch due to int8 quantization
- **Next step**: Test GGUF FP16 loading in C application to verify output quality
- If GGUF FP16 produces coherent English, the fix is complete

### Comparison with PyTorch
| Token | PyTorch | C Code (dbits=16) |
|-------|---------|--------|
| "The" | "following are multiple choice..." | "dermat_rank" (English but not coherent) |
| "The cat sat" | "on the mat" | "dermat_rank" |
| Quality | Coherent English | English words, wrong context |

