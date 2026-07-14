"""
PyTorch reference for Qwen3.5-0.8B numerical accuracy testing.
Compares per-token logits and top-5 vocabulary against C implementation.

Usage:
    python3 torch_test.py                    # Full comparison (greedy + top-5)
    python3 torch_test.py --top 10           # Show top-10 instead of top-5
    python3 torch_test.py --prompt "Once upon a time"  # Custom prompt
    python3 torch_test.py --verbose          # Print ALL vocabulary logits
"""
import os
import sys
import json
import argparse

os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
os.environ['LLVM_PATH'] = os.environ.get('ROCM_PATH', '/opt/rocm')
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.5.1'
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1151'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Qwen3.5-0.8B")

def main():
    parser = argparse.ArgumentParser(description="PyTorch reference for Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="The cat sat", help="Input prompt")
    parser.add_argument("--top", type=int, default=5, help="Show top-N tokens")
    parser.add_argument("--ngenerate", type=int, default=3, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature (0 = greedy)")
    parser.add_argument("--verbose", action="store_true", help="Print all vocabulary logits")
    parser.add_argument("--save", type=str, default=None, help="Save logits to JSON file")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    torch.set_default_device('cuda')
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # Encode prompt
    tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long).to("cuda")
    
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")
    print()

    all_logits = []
    generated_ids = []
    current_ids = input_ids.clone()

    for step in range(args.ngenerate):
        with torch.no_grad():
            outputs = model(current_ids, use_cache=False)
            logits = outputs.logits[0, -1]  # [vocab_size]
        
        # Convert to CPU for printing
        logits_cpu = logits.float()
        
        # Get top-K
        probs = torch.softmax(logits_cpu, dim=-1)
        
        if args.verbose:
            print(f"Step {step+1}: ALL vocab logits (first 20):")
            top_full = torch.topk(logits_cpu, min(20, len(logits_cpu)))
            for i, (logit, idx) in enumerate(zip(top_full.values, top_full.indices)):
                tok = tokenizer.decode([int(idx)])
                print(f"  {int(idx):6d} = {tok!r:20s} logit={logit.item():8.4f} prob={probs[idx].item():8.4f}")
        else:
            topk = torch.topk(logits_cpu, args.top)
            print(f"Step {step+1} logits (after '{tokenizer.decode(current_ids[0].tolist())}'):")
            for i, (logit, idx) in enumerate(zip(topk.values, topk.indices)):
                tok = tokenizer.decode([int(idx)])
                print(f"  {int(idx):6d} = {tok!r:20s} logit={logit.item():8.4f} prob={probs[idx].item():8.4f}")
        
        all_logits.append(logits_cpu.cpu().numpy())
        
        # Sample or greedy
        if args.temp > 0:
            logits_scaled = logits_cpu / args.temp
            probs_scaled = torch.softmax(logits_scaled, dim=-1)
            next_id = torch.multinomial(probs_scaled, 1)
        else:
            next_id = torch.argmax(logits_cpu, dim=-1, keepdim=True)
        
        generated_ids.append(int(next_id.item()))
        current_ids = torch.cat([current_ids, next_id.unsqueeze(0)], dim=1)
        print(f"  -> Generated: {int(next_id.item())} = {tokenizer.decode([int(next_id.item())])!r}")
        print()

    # Summary
    print("=== Generation Summary ===")
    full_ids = tokens + generated_ids
    full_text = tokenizer.decode(full_ids)
    print(f"Full text: {full_text!r}")
    print(f"Generated: {tokenizer.decode(generated_ids)!r}")
    print()

    # Save logits if requested
    if args.save:
        # Convert numpy arrays to lists
        save_data = []
        for i, logit_arr in enumerate(all_logits):
            top5_indices = logit_arr.argsort()[-args.top:][::-1]
            top5_logits = logit_arr[top5_indices]
            top5_tokens = [tokenizer.decode([int(x)]) for x in top5_indices]
            save_data.append({
                "step": i + 1,
                "prompt_context": tokenizer.decode(current_ids[0, :-(args.ngenerate-i)].tolist()),
                "top_tokens": [
                    {"id": int(idx), "token": tok, "logit": float(log)}
                    for idx, tok, log in zip(top5_indices, top5_tokens, top5_logits)
                ]
            })
        
        with open(args.save, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved logits to {args.save}")
        
        # Also save raw logits for direct comparison
        raw_data = {
            "prompt": args.prompt,
            "tokens": tokens,
            "generated": generated_ids,
            "generation": save_data,
            "all_logits": [arr.tolist() for arr in all_logits]
        }
        raw_path = args.save.replace('.json', '_raw.json')
        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        print(f"Saved raw logits to {raw_path}")

if __name__ == "__main__":
    main()
