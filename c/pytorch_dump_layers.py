#!/usr/bin/env python3
"""
Dump per-layer intermediate tensors from PyTorch model for C engine comparison.
Saves to c/pytorch_ref_layers.json.

Usage:
    python3 pytorch_dump_layers.py "The cat sat" 3
"""
import torch
import json
import sys
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The cat sat"
    ngen = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model_path = os.environ.get("SNAP", "../Qwen3.5-0.8B")
    
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
    print(f"Prompt: {prompt}")
    print(f"Input shape: {input_ids.shape}")
    
    n_layers = model.config.get_text_config().num_hidden_layers
    layer_types = model.config.get_text_config().layer_types
    
    # Store per-layer measurements
    layers = {}
    
    # Register hooks per layer, capturing layer index in closure
    for i in range(n_layers):
        layer_name = f"l{i}_"
        lt = layer_types[i]
        layer = model.model.layers[i]
        
        # === Attention output ===
        if lt == "full_attention":
            attn_mod = layer.self_attn
        elif lt == "linear_attention":
            attn_mod = layer.linear_attn
        else:
            continue
        
        def make_attn_hook(idx, lname):
            def hook(mod, inp, out):
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    return
                layers[f"{lname}attn_out"] = {
                    "first16": out.flatten()[:16].tolist(),
                    "rms": float(out.pow(2).mean().sqrt())
                }
            return hook
        
        h1 = attn_mod.register_forward_hook(make_attn_hook(i, layer_name))
        
        # === Q/K projections (for full attention layers) ===
        if lt == "full_attention" and hasattr(attn_mod, 'q_proj'):
            def make_q_hook(idx, lname):
                def hook(mod, inp, out):
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if not isinstance(out, torch.Tensor):
                        return
                    layers[f"{lname}q_proj"] = {
                        "first16": out.flatten()[:16].tolist(),
                        "rms": float(out.pow(2).mean().sqrt())
                    }
                return hook
            h_q = attn_mod.q_proj.register_forward_hook(make_q_hook(i, layer_name))
            
            def make_k_hook(idx, lname):
                def hook(mod, inp, out):
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if not isinstance(out, torch.Tensor):
                        return
                    layers[f"{lname}k_proj"] = {
                        "first16": out.flatten()[:16].tolist(),
                        "rms": float(out.pow(2).mean().sqrt())
                    }
                return hook
            h_k = attn_mod.k_proj.register_forward_hook(make_k_hook(i, layer_name))
        
        # === Conv1d input projection (for linear attention) ===
        if lt == "linear_attention" and hasattr(attn_mod, 'in_proj_qkv'):
            def make_conv_hook(idx, lname):
                def hook(mod, inp, out):
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    if not isinstance(out, torch.Tensor):
                        return
                    layers[f"{lname}in_proj_qkv"] = {
                        "first16": out.flatten()[:16].tolist(),
                        "rms": float(out.pow(2).mean().sqrt())
                    }
                return hook
            h_c = attn_mod.in_proj_qkv.register_forward_hook(make_conv_hook(i, layer_name))
        
        # === MLP output ===
        def make_mlp_hook(idx, lname):
            def hook(mod, inp, out):
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    return
                layers[f"{lname}mlp_out"] = {
                    "first16": out.flatten()[:16].tolist(),
                    "rms": float(out.pow(2).mean().sqrt())
                }
            return hook
        h_mlp = layer.mlp.register_forward_hook(make_mlp_hook(i, layer_name))
        
        # === Post-layer norm ===
        def make_postln_hook(idx, lname):
            def hook(mod, inp, out):
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    return
                layers[f"{lname}post_ln"] = {
                    "first16": out.flatten()[:16].tolist(),
                    "rms": float(out.pow(2).mean().sqrt())
                }
            return hook
        h_pn = layer.post_attention_layernorm.register_forward_hook(make_postln_hook(i, layer_name))
        
        # === Input layer norm (captures post-residual signal from previous layer) ===
        def make_resid_hook(idx, lname):
            def hook(mod, inp, out):
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if not isinstance(out, torch.Tensor):
                    return
                # inp[0] is the signal going into this layer (i.e., output from previous layer)
                layers[f"{lname}post_resid"] = {
                    "first16": out.flatten()[:16].tolist(),
                    "rms": float(out.pow(2).mean().sqrt())
                }
            return hook
        h_iln = layer.input_layernorm.register_forward_hook(make_resid_hook(i, layer_name))
        
        print(f"  Layer {i} ({lt}): registered 6 hooks")
    
    # === Final model output (after last norm) ===
    def make_final_hook():
        def hook(mod, inp, out):
            if isinstance(out, (list, tuple)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return
            layers["final_output"] = {
                "first16": out.flatten()[:16].tolist(),
                "rms": float(out.pow(2).mean().sqrt())
            }
        return hook
    h_final = model.model.norm.register_forward_hook(make_final_hook())
    
    # Forward pass (prefill)
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits
    
    # Generate tokens
    generated = []
    for step in range(ngen):
        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits
            next_token = torch.argmax(logits[0, -1, :], dim=-1)
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nGenerated ({len(generated)} tokens): {generated_text}")
    
    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_ref_layers.json")
    with open(output_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "generated_ids": generated,
            "generated_text": generated_text,
            "layers": layers,
        }, f, indent=2)
    
    print(f"\nSaved {len(layers)} layer measurements to {output_path}")
    print(f"\nLayer keys ({len(layers)}):")
    for k in sorted(layers.keys()):
        print(f"  {k}")

if __name__ == "__main__":
    main()
