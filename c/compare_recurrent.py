"""
Step-by-step comparison of gated delta rule recurrence: C vs PyTorch.
Compares state S[h,i,j], Q, K, V, and y_all at each timestep.
"""
import os
import json
import torch
import numpy as np
import torch.nn.functional as F

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
device = torch.device('cuda')

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    '/workdir/colibri.git/Qwen3.5-0.8B',
    torch_dtype=torch.float32,
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
config = model.config.get_text_config()
tokenizer = AutoTokenizer.from_pretrained(
    '/workdir/colibri.git/Qwen3.5-0.8B', trust_remote_code=True)

# Tokenize
tokens = tokenizer("The cat sat", return_tensors="pt").input_ids.to(device)
BS = 1
S = tokens.shape[1]  # 3

# Get config
nq = config.linear_num_key_heads      # 16
nv = config.linear_num_value_heads    # 16
kd = config.linear_key_head_dim       # 128
vd = config.linear_value_head_dim     # 128
ck = config.linear_conv_kernel_dim    # 4
D = config.hidden_size                # 1024

print(f"\nConfig: nq={nq} nv={nv} kd={kd} vd={vd} ck={ck} D={D} S={S}")

# Extract parameters
layer0 = model.model.layers[0]

def to_np(t):
    return t.detach().cpu().float().numpy()

# Run forward pass step by step
with torch.no_grad():
    # 1. Embed + LayerNorm
    hidden = model.model.embed_tokens(tokens)
    nrm = layer0.input_layernorm(hidden)
    
    print("\n=== Step 1: Input to linear_attn (after LN) ===")
    print(f"nrm shape: {nrm.shape}")
    print(f"nrm[0] first 8: {to_np(nrm)[0, :8]}")
    
    # 2. Causal conv1d on QKV
    mixed_qkv = layer0.linear_attn.in_proj_qkv(nrm)
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [BS, conv_dim, S]
    
    # Apply conv1d with silu (matching the PyTorch code)
    conv_out = layer0.linear_attn.conv1d(mixed_qkv)
    conv_out_silu = F.silu(conv_out)[:, :, :S]  # truncate to S
    conv_out_silu = conv_out_silu.transpose(1, 2)  # [BS, S, conv_dim]
    
    print("\n=== Step 2: After conv1d ===")
    print(f"conv_out_silu shape: {conv_out_silu.shape}")
    print(f"conv_out_silu[0] first 8: {to_np(conv_out_silu)[0, :8]}")
    
    # 3. Split QKV
    key_dim = nq * kd
    value_dim = nv * vd
    q_raw, k_raw, v_raw = torch.split(conv_out_silu, [key_dim, key_dim, value_dim], dim=-1)
    z_raw = layer0.linear_attn.in_proj_z(nrm)
    a_raw = layer0.linear_attn.in_proj_a(nrm)
    b_raw = layer0.linear_attn.in_proj_b(nrm)
    
    print("\n=== Step 3: Split QKV ===")
    print(f"q_raw shape: {q_raw.shape}")
    print(f"z_raw[0] first 8: {to_np(z_raw)[0, :8]}")
    
    # 4. Reshape Q, K, V
    q = q_raw.reshape(BS, S, nq, kd)
    k = k_raw.reshape(BS, S, nq, kd)
    v = v_raw.reshape(BS, S, nv, vd)
    
    # 5. L2 normalize Q, K
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    
    print("\n=== Step 4: After L2 normalize ===")
    print(f"q[0,0] RMS: {torch.sqrt(torch.mean(q**2)).item():.6f}")
    print(f"q[0,0] first 8: {to_np(q)[0, 0, :8]}")
    print(f"k[0,0] first 8: {to_np(k)[0, 0, :8]}")
    
    # 6. Compute gating
    A_log = layer0.linear_attn.A_log.float()
    dt_bias = layer0.linear_attn.dt_bias
    g = -A_log.exp() * F.softplus(a_raw.float() + dt_bias)
    beta = torch.sigmoid(b_raw.float())
    
    print("\n=== Step 5: Gating ===")
    print(f"g[0] first 4: {to_np(g)[0, :4]}")
    print(f"beta[0] first 4: {to_np(beta)[0, :4]}")
    print(f"A_log first 4: {to_np(A_log)[:4]}")
    print(f"dt_bias first 4: {to_np(dt_bias)[:4]}")
    
    # 7. GQA repeat (if applicable)
    if nv // nq > 1:
        q = q.repeat_interleave(nv // nq, dim=2)
        k = k.repeat_interleave(nv // nq, dim=2)
        print(f"\nGQA: repeated {nv//nq}x, q shape: {q.shape}")
    
    # 8. Manual gated delta rule
    print("\n=== Step 6: Gated Delta Rule Recurrence ===")
    
    # Initialize state S[h, kd, vd] = 0
    state = torch.zeros(nq, kd, vd, device=device)
    
    results = {
        'pytorch': {},
        'c_reference': {},
    }
    
    for t in range(S):
        print(f"\n--- Timestep {t} ---")
        
        q_t = q[0, t]  # [nq, kd] = [16, 128]
        k_t = k[0, t]  # [nq, kd] = [16, 128]
        v_t = v[0, t]  # [nv, vd] = [16, 128]
        g_t = g[0, t]  # [nq] = [16]
        beta_t = beta[0, t]  # [nv] = [16]
        
        # Decay: S = exp(g) * S
        decay = torch.exp(torch.clamp(g_t, max=0))  # [nq] = [16]
        for h in range(nq):
            state[h] = state[h] * decay[h]
        
        print(f"  decay[{t % nq}] = {decay[t % nq].item():.6f}")
        
        # Compute kv_mem[h] = sum_j S[h,i] * k[h,i] for each v-channel j
        # kv_mem[h] has shape [nv, vd] but we need [nq, vd]
        kv_mem = torch.zeros(nq, vd, device=device)
        for h in range(nq):
            # kv_mem[h, j] = sum_i S[h, i, j] * k[h, i]
            kv_mem[h] = torch.einsum('i,hj->j', k_t[h], state[h])
        
        # Innovation: delta[h] = (v[h] - kv_mem[h/2]) * beta[h]
        delta = torch.zeros(nv, vd, device=device)
        for h in range(nv):
            kh = h * nq // nv  # GQA mapping
            delta[h] = (v_t[h] - kv_mem[kh]) * beta_t[h]
        
        print(f"  delta[0] RMS: {torch.sqrt(torch.mean(delta**2)).item():.6f}")
        print(f"  delta[0] first 4: {to_np(delta)[0, :4]}")
        
        # Update state: S[h] += outer(k[h], delta[h])
        for h in range(nq):
            state[h] += torch.einsum('i,j->ij', k_t[h], delta[h])
        
        # Output: y[h, j] = sum_i q[h, i] * S[h, i, j]
        # state[h] has shape [kd, vd], q_t[kh] has shape [kd]
        y_t = torch.zeros(nv, vd, device=device)
        for h in range(nv):
            kh = h * nq // nv
            y_t[h] = torch.einsum('i,ij->j', q_t[kh], state[h])
        
        print(f"  y[{t}] RMS: {torch.sqrt(torch.mean(y_t**2)).item():.6f}")
        print(f"  y[0] first 4: {to_np(y_t)[0, :4]}")
        
        # Store results
        results['pytorch'][f't{t}'] = {
            'decay': to_np(decay)[:4].tolist(),
            'delta_rms': torch.sqrt(torch.mean(delta**2)).item(),
            'delta_first4': to_np(delta)[0, :4].tolist(),
            'y_rms': torch.sqrt(torch.mean(y_t**2)).item(),
            'y_first4': to_np(y_t)[0, :4].tolist(),
            'state_rms': torch.sqrt(torch.mean(state**2)).item(),
        }
        
        # Store state for C comparison
        for h in range(min(nq, 2)):
            results['pytorch'][f'state_t{t}_h{h}'] = to_np(state[h])[:4, :4].tolist()
    
    # 9. Get the norm output to see the actual PyTorch pre-out_proj values
    print("\n=== Step 7: Norm Output ===")
    
    class Hook:
        def __init__(self):
            self.value = None
        def __call__(self, module, inp, outp):
            if isinstance(outp, torch.Tensor):
                self.value = outp.detach().cpu().numpy()
    
    hook = Hook()
    layer0.linear_attn.norm.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(tokens)
    norm_out = hook.value
    
    print(f"norm_out shape: {norm_out.shape}")
    print(f"norm_out RMS: {np.sqrt(np.mean(norm_out**2)):.6f}")
    print(f"norm_out[0] first 16: {norm_out[:16].flatten()}")
    print(f"norm_out[1] first 16: {norm_out[16:32].flatten()}")
    print(f"norm_out[2] first 16: {norm_out[32:48].flatten()}")
    
    # Compare with our manual recurrence
    print("\n=== Comparison ===")
    print(f"Manual y RMS at t=0: {results['pytorch']['t0']['y_rms']:.6f}")
    print(f"Manual y RMS at t=1: {results['pytorch']['t1']['y_rms']:.6f}")
    print(f"Manual y RMS at t=2: {results['pytorch']['t2']['y_rms']:.6f}")
    print(f"Actual norm_out RMS: {np.sqrt(np.mean(norm_out**2)):.6f}")
    
    # The norm output includes z-gating. We need to check if the pre-gating y values match.
    # pyTorch: out = RMSNorm(y) * weight * silu(z)
    # We can get pre-gating y by dividing out the gating
    z_flat = z_raw[0].reshape(-1, 128)
    z_silu = F.silu(z_flat).cpu().numpy()
    ln_w = to_np(layer0.linear_attn.norm.weight)
    
    # pre_gate = norm_out / (RMSNorm_weight * silu(z))
    # But RMSNorm normalizes per-channel, so we need to compute it properly
    var = np.mean(norm_out**2, axis=1, keepdims=True)
    rms_norm = np.sqrt(var + 1e-6)
    pre_gate = norm_out / rms_norm / ln_w / z_silu  # This gives the raw y values
    print(f"\nPre-gating y first 16: {pre_gate[:16].flatten()}")
    print(f"Pre-gating y RMS: {np.sqrt(np.mean(pre_gate**2)):.6f}")

# Save results
with open('c/recurring_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to c/recurring_comparison.json")
