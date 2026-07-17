"""
Full per-element PyTorch reference dump for Qwen3.5 linear attention.
Runs the PyTorch model and dumps all intermediate tensors for comparison with C.
"""
import os
import json
import torch
import numpy as np

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
device = torch.device('cuda')

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/workdir/colibri.git/Qwen3.5-0.8B",
    torch_dtype=torch.float32,
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
config = model.config.get_text_config()

# Tokenize prompt
tokenizer = AutoTokenizer.from_pretrained("/workdir/colibri.git/Qwen3.5-0.8B", trust_remote_code=True)
prompt = "The cat sat"
tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"Prompt tokens: {tokens.shape} -> {tokenizer.decode(tokens[0])}")

# Hook to capture intermediate tensors
captured = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            captured[name] = output.detach().cpu().float().numpy()
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    captured[f"{name}_{i}"] = o.detach().cpu().float().numpy()
        else:
            captured[name] = str(type(output))
    return hook

# Hook into Qwen3.5GatedDeltaNet forward
layer0 = model.model.layers[0]
linear_attn = layer0.linear_attn

# Register hooks on key operations
linear_attn.in_proj_qkv.register_forward_hook(make_hook("in_proj_qkv"))
linear_attn.in_proj_z.register_forward_hook(make_hook("in_proj_z"))
linear_attn.in_proj_a.register_forward_hook(make_hook("in_proj_a"))
linear_attn.in_proj_b.register_forward_hook(make_hook("in_proj_b"))
linear_attn.conv1d.register_forward_hook(make_hook("conv1d"))
linear_attn.norm.register_forward_hook(make_hook("norm"))
linear_attn.out_proj.register_forward_hook(make_hook("out_proj"))

# Also hook the internal forward
orig_forward = linear_attn.forward
def hooked_forward(hidden_states, **kwargs):
    result = orig_forward(hidden_states, **kwargs)
    return result
linear_attn.forward = hooked_forward

# Run inference
print("Running forward pass...")
with torch.no_grad():
    outputs = model(tokens, use_cache=False)
    logits = outputs.logits
    preds = torch.argmax(logits[0, -1], dim=-1)
    print(f"Next token: {tokenizer.decode(preds)}")

# Dump all captured tensors
print(f"\nCaptured {len(captured)} tensors:")

def tensor_to_dict(t):
    """Convert numpy array to JSON-serializable dict"""
    if t.dtype == np.float64:
        t = t.astype(np.float32)
    d = {
        "shape": list(t.shape),
        "rms": float(np.sqrt(np.mean(t**2))),
        "first16": t.flatten()[:16].tolist(),
    }
    return d

result = {}
for name, t in captured.items():
    if isinstance(t, np.ndarray):
        result[name] = tensor_to_dict(t)
        print(f"  {name:30s} shape={str(t.shape):25s} rms={result[name]['rms']:.6f} first4={t.flatten()[:4]}")
    else:
        result[name] = {"type": str(t)}
        print(f"  {name:30s} type={t}")

# Save to JSON
with open("c/pytorch_qwen35_full.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to c/pytorch_qwen35_full.json")

# Now manually compute each step to see the full pipeline
print("\n" + "="*80)
print("MANUAL STEP-BY-STEP COMPUTATION")
print("="*80)

# Re-run with manual intermediate captures
model2 = AutoModelForCausalLM.from_pretrained(
    "/workdir/colibri.git/Qwen3.5-0.8B",
    torch_dtype=torch.float32,
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

manual = {}

# Get hidden states from embed_tokens
hidden = model2.model.embed_tokens(tokens)
manual["embed"] = tensor_to_dict(hidden[0].cpu().float().numpy())
print(f"\n1. Embed: {hidden.shape}")

# Get layer norm input
nrm = model2.model.norm(hidden)
manual["final_norm"] = tensor_to_dict(nrm[0].cpu().float().numpy())
print(f"   final_norm rms={manual['final_norm']['rms']:.6f}")

# Forward through layer 0
print(f"\n2. Layer 0 forward...")
layer_out = model2.model.layers[0](hidden, use_cache=False).hidden_states
manual["layer0_out"] = tensor_to_dict(layer_out[0].cpu().float().numpy())
print(f"   layer_out rms={manual['layer0_out']['rms']:.6f}")

# Save manual results
with open("c/pytorch_qwen35_manual.json", "w") as f:
    json.dump(manual, f, indent=2)
print(f"\nSaved manual results to c/pytorch_qwen35_manual.json")
