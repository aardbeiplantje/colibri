#!/usr/bin/env python3
"""
Dump linear attention intermediates from PyTorch model for C engine comparison.
Matches the C code's DEBUG_LINEAR points exactly.
"""
import torch
import json
import os
import math

def rms(tensor):
    """Compute RMS of a tensor."""
    return float(torch.sqrt(torch.mean(tensor * tensor)))

def first16(tensor):
    """Return first 16 values of a tensor as list."""
    return tensor.flatten()[:16].tolist()

def main():
    model_path = os.environ.get("SNAP", "../Qwen3.5-0.8B")
    prompt = "The cat sat"
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"Loading model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    S = input_ids.shape[1]
    B = input_ids.shape[0]
    
    # Get config
    config = model.config.get_text_config()
    D = config.hidden_size
    nq = config.linear_num_key_heads  # 16
    nv = config.linear_num_value_heads  # 16
    kd = config.linear_key_head_dim  # 128
    vd = config.linear_value_head_dim  # 128
    ck = config.linear_conv_kernel_dim  # 4
    conv_dim = nq*kd + nv*vd + nq*kd  # 6144
    
    print(f"Linear attn params: D={D}, nq={nq}, nv={nv}, kd={kd}, vd={vd}, ck={ck}, conv_dim={conv_dim}")
    
    results = {}
    
    # Process layer 0
    layer0 = model.model.layers[0]
    
    # 1) Embedding + Input layernorm (Qwen3.5: (1+weight)/RMSNorm)
    embed = model.model.embed_tokens(input_ids)
    x = embed[0]  # [S, D]
    
    in_ln_w = layer0.input_layernorm.weight
    x_rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    x_ln = (1 + in_ln_w) * x / x_rms  # [S, D]
    
    results["in_rms"] = rms(x_ln)
    results["in_ln_first16"] = first16(x_ln[0])
    
    # 2) Combined QKV projection: in_proj_qkv
    in_proj_qkv_w = layer0.linear_attn.in_proj_qkv.weight  # [conv_dim, D]
    qkv_all = torch.matmul(x_ln, in_proj_qkv_w.T)  # [S, conv_dim]
    
    results["qkv_rms"] = rms(qkv_all)
    results["qkv_first16"] = first16(qkv_all[0])
    
    # 3) Causal conv1d
    conv1d_w = layer0.linear_attn.conv1d.weight  # [conv_dim, 1, ck]
    results["conv1d_w_first4"] = first16(conv1d_w[0, 0])
    
    # Apply conv1d manually per-channel
    # conv1d_w shape: [conv_dim, 1, ck] — each channel convolved independently
    S = qkv_all.shape[0]
    conv_out_raw = torch.zeros(S, conv_dim, device=device, dtype=dtype)
    for c in range(conv_dim):
        # Convolve channel c with kernel
        kernel = conv1d_w[c, 0]  # [ck]
        # Pad input: add (ck-1) zeros at the beginning
        padded = torch.nn.functional.pad(qkv_all[:, c], (ck-1, 0))  # [S+ck-1]
        for t in range(S):
            conv_out_raw[t, c] = torch.dot(padded[t:t+ck], kernel)
    conv_out = torch.nn.functional.silu(conv_out_raw)
    
    results["conv1d_rms"] = rms(conv_out)
    results["conv1d_first16"] = first16(conv_out[0])
    
    # 4) Split QKV
    q_all = conv_out[:, :nq*kd]  # [S, nq*kd]
    k_all = conv_out[:, nq*kd:2*nq*kd]  # [S, nq*kd]
    v_all = conv_out[:, 2*nq*kd:]  # [S, nv*vd]
    
    # L2 normalize Q and K (matches PyTorch F.normalize)
    q_norm = torch.sqrt(q_all.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    q_all = q_all / q_norm
    k_norm = torch.sqrt(k_all.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    k_all = k_all / k_norm
    
    results["k_rms"] = rms(k_all)
    results["k_first16"] = first16(k_all[0])
    results["q_rms"] = rms(q_all)
    results["v_rms"] = rms(v_all)
    
    # 5) Z, A, B projections
    z_all = torch.matmul(x_ln, layer0.linear_attn.in_proj_z.weight.T)  # [S, nv*vd]
    a_all = torch.matmul(x_ln, layer0.linear_attn.in_proj_a.weight.T)  # [S, nq]
    b_all = torch.matmul(x_ln, layer0.linear_attn.in_proj_b.weight.T)  # [S, nq]
    
    results["z_rms"] = rms(z_all)
    results["z_first16"] = first16(z_all[0])
    results["a_first16"] = first16(a_all[0])
    results["b_first16"] = first16(b_all[0])
    
    # 6) Gating
    A_log = layer0.linear_attn.A_log  # [nq]
    dt_bias = layer0.linear_attn.dt_bias  # [nq]
    
    g_all = torch.zeros(S, nq, device=device, dtype=dtype)
    beta_all = torch.zeros(S, nq, device=device, dtype=dtype)
    for t in range(S):
        for h in range(nq):
            A_val = torch.exp(A_log[h])
            dt_val = dt_bias[h]
            a_val = a_all[t, h]
            b_val = b_all[t, h]
            # softplus(a_val + dt)
            soft_a = torch.nn.functional.softplus(a_val + dt_val)
            g_all[t, h] = -A_val * soft_a
            beta_all[t, h] = torch.sigmoid(b_val)
    
    results["g_first8"] = first16(g_all[0])[:8]
    results["beta_first8"] = first16(beta_all[0])[:8]
    results["decay_first8"] = (torch.exp(g_all.clamp(min=-100))).tolist()[:8]
    
    # 7) Linear attention recurrence
    state = torch.zeros(nq, kd, vd, device=device, dtype=dtype)
    y_all = torch.zeros(S, nv*vd, device=device, dtype=dtype)
    
    for t in range(S):
        bs = t
        for h in range(nq):
            decay = torch.exp(g_all[bs, h])
            state[h] *= decay
            kv = k_all[bs, h*kd:(h+1)*kd].squeeze(0)  # [kd]
            vh = h * nv // nq  # value head
            vv = v_all[bs, vh*vd:(vh+1)*vd].squeeze(0)  # [vd]
            beta = beta_all[bs, h]
            kv_mem = torch.mv(state[h].T, kv)  # [vd]
            delta = (vv - kv_mem) * beta
            state[h] += torch.ger(kv, delta)  # outer product
        
        # Output: y[bs, h, j] = sum_i q[bs, h, i] * S[bs, h, i, j]
        for h in range(nv):
            qh = q_all[bs, h*kd:(h+1)*kd].squeeze(0)  # [kd]
            y_all[bs, h*vd:(h+1)*vd] = torch.mv(state[h].T, qh)  # [vd]
    results["y_rms"] = rms(y_all)
    results["y_first16"] = first16(y_all[0])
    
    # 8) Z gating + RMSNorm: RMSNorm(core_attn) * weight * silu(z)
    # ln_w shape is [vd], applied per value-head channel
    ln_w = layer0.linear_attn.norm.weight  # [vd]
    ln_w_expanded = ln_w.unsqueeze(0).expand(nv, -1).reshape(-1)  # [nv*vd]
    silu_z = torch.nn.functional.silu(z_all)
    y_rms = torch.sqrt(y_all.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    y_gated = (y_all / y_rms) * ln_w_expanded * silu_z  # [S, nv*vd]
    
    results["y_gated_rms"] = rms(y_gated)
    results["y_gated_first16"] = first16(y_gated[0])
    
    # 9) Output projection
    o_proj_w = layer0.linear_attn.out_proj.weight  # [D, nv*vd]
    out_final = torch.matmul(y_gated, o_proj_w.T)  # [S, D]
    
    results["out_rms"] = rms(out_final)
    results["out_first16"] = first16(out_final[0])
    results["out_first8"] = first16(out_final[0])[:8]
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_linear_debug.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    print(f"\nKey values:")
    print(f"  in_rms: {results['in_rms']:.6f}")
    print(f"  qkv_rms: {results['qkv_rms']:.6f}")
    print(f"  conv1d_rms: {results['conv1d_rms']:.6f}")
    print(f"  k_rms: {results['k_rms']:.6f}")
    print(f"  y_rms: {results['y_rms']:.6f}")
    print(f"  y_gated_rms: {results['y_gated_rms']:.6f}")
    print(f"  out_rms: {results['out_rms']:.6f}")
    print(f"\n  out_first8: {results['out_first8']}")
    print(f"\n  conv1d_w_first4: {results['conv1d_w_first4']}")
    
    # Compare with C values from DEBUG_LINEAR output
    print(f"\n=== Comparison with C ===")
    print(f"  conv1d_w[0:4]: C=-0.000161,0.000404,-0.003387,-0.074219  PyTorch={results['conv1d_w_first4']}")
    print(f"  out_rms: C=0.062670  PyTorch={results['out_rms']:.6f}")
    print(f"  out_first8: C=[-0.988341, 0.148112, -0.153579, -0.032761, -0.023323, -0.050638, 0.075717, -0.061789]")
    print(f"  out_first8: PyTorch={results['out_first8']}")

if __name__ == "__main__":
    main()
