#!/usr/bin/env python3
"""
Compare common layer measurements between C engine and PyTorch reference.
Filters to only the shared measurements: attn_out, post_resid, post_ln, mlp_out.
"""
import json
import sys
import glob

def load_json(path):
    return json.load(open(path))

def rms_relative_diff(a_rms, b_rms):
    if a_rms < 1e-10 and b_rms < 1e-10:
        return 0.0
    denom = max(a_rms, b_rms)
    return abs(a_rms - b_rms) / denom

def first16_relative_diff(a_first16, b_first16):
    if not a_first16 or not b_first16:
        return 0.0
    a_rms = (sum(x*x for x in a_first16) / len(a_first16)) ** 0.5
    if a_rms < 1e-10:
        return 0.0
    diff = sum(abs(a - b) for a, b in zip(a_first16, b_first16))
    return diff / (len(a_first16) * a_rms)

def main():
    c_files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "debug_layer_*.json"))
    pytorch_path = sys.argv[2] if len(sys.argv) > 2 else "pytorch_ref_layers.json"
    
    threshold = 0.01  # 1% relative diff
    
    print("=" * 80)
    print("C vs PyTorch Layer Comparison (common measurements only)")
    print("=" * 80)
    
    for c_file in c_files:
        c_data = load_json(c_file)
        pytorch_data = load_json(pytorch_path)
        
        c_layers = c_data["layers"]
        py_layers = pytorch_data["layers"]
        
        # Common measurement types
        common_types = ["attn_out", "post_resid", "post_ln", "mlp_out"]
        
        mismatches = []
        
        # Group by layer
        for layer in range(25):  # layers 0-24
            for meas_type in common_types:
                c_key = f"l{layer}_{meas_type}"
                py_key = f"l{layer}_{meas_type}"
                
                if c_key not in c_layers or py_key not in py_layers:
                    continue
                
                c_val = c_layers[c_key]
                py_val = py_layers[py_key]
                
                c_rms = c_val["rms"]
                py_rms = py_val["rms"]
                rms_diff = rms_relative_diff(c_rms, py_rms)
                
                f16_diff = first16_relative_diff(c_val["first16"], py_val["first16"])
                
                max_diff = max(rms_diff, f16_diff)
                
                if max_diff > threshold:
                    mismatches.append((layer, meas_type, c_rms, py_rms, rms_diff, f16_diff))
        
        print(f"\n--- {c_file} ---")
        print(f"Prompt: {c_data.get('prompt', 'N/A')}")
        print(f"Generated: {len(mismatches)} mismatches out of 100 common measurements")
        
        if mismatches:
            print(f"\nMISMATCHES (>{threshold*100}% diff):")
            for layer, meas, c_r, py_r, rd, fd in mismatches[:20]:
                print(f"  l{layer}_{meas:12s}  C_rms={c_r:.6f}  Py_rms={py_r:.6f}  "
                      f"RMS_diff={rd:.4f}  F16_diff={fd:.4f}")
            
            # Find the FIRST divergent layer
            first_layer = mismatches[0][0]
            first_meas = mismatches[0][1]
            print(f"\n*** FIRST DIVERGENCE: l{first_layer}_{first_meas} ***")
            
            # Show detailed comparison for first divergence
            c_key = f"l{first_layer}_{first_meas}"
            py_key = f"l{first_layer}_{first_meas}"
            c_first16 = c_layers[c_key]["first16"]
            py_first16 = py_layers[py_key]["first16"]
            print(f"  C:         {c_first16[:8]}")
            print(f"  PyTorch:   {py_first16[:8]}")
        else:
            print("\n  All common measurements match within threshold!")

if __name__ == "__main__":
    main()
