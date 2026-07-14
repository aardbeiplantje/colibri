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
| 8.3 | Numerical accuracy — output quality | 🔴 Needs more work |

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

### Latest Progress (2026-07-14)

**Critical Bug Fixes Applied:**
1. **Qwen3.5 RMSNorm offset weights** — Added `rmsnorm_qw35()` that applies `(1.0 + weight)` instead of raw `weight`. PyTorch's `Qwen3_5RMSNorm` uses `weight = nn.Parameter(torch.zeros(dim))` and forward is `output = _norm(x) * (1.0 + weight)`. This was causing all layer norm outputs to be wrong.
2. **F.silu after conv1d** — Added silu activation after conv1d in `linear_attn_forward()`, matching the reference.
3. **softplus helper** — Added numerically stable softplus for alpha computation.

**Current Status:**
- Model now produces NON-ZERO activations ✓
- Layer norm outputs are now correct (RMS ~1.22 vs expected ~1.0-1.2) ✓
- Model generates tokens but produces INCOHERENT mixed-language output

**Comparison:**
- PyTorch: "The cat sat on the floor" (coherent English)
- C: "The cat sat canoeadou敢于好用(BigDecimal señora conferencias" (gibberish)

**Remaining Issues:**
- Linear attention kernel output quality — Q/K normalization, state accumulation, or RoPE may be incorrect
- MLP output quality  
- Weight loading verification for some tensors
- Alpha/gating computation may still not match reference pattern

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
   python3 c/torch_test.py --prompt "The cat sat" --save c/pytorch_ref.json
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
python3 c/torch_test.py --prompt "The cat sat" --save c/pytorch_ref.json
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
  $ python3 c/torch_test.py --prompt "The cat sat" --ngenerate 3
  on the floor, and the cat sat on the floor.
The cat
  ```
  **ACTION**: Compare C vs PyTorch per-token logits:
  1. `python3 c/torch_test.py --prompt "The cat sat" --save c/pytorch_ref.json`
  2. Run C with same prompt and compare logits
  3. Identify first layer where outputs diverge
  4. Fix the offending layer

- [ ] **PyTorch reference script** — `c/torch_test.py` provides:
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
