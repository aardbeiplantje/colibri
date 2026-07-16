#!/usr/bin/env python3
"""
Convert any safetensors model to GGUF with pre-quantized weights.
Supports FP16 (lossless), FP8-E4M3, and FP4-E2M1 quantization.

Usage:
    python3 convert_to_gguf.py <input_dir> <output.gguf> --fmt fp16  # FP16 (recommended for accuracy)
    python3 convert_to_gguf.py <input_dir> <output.gguf> --fmt fp8   # FP8-E4M3 (good accuracy, 2x compression)
    python3 convert_to_gguf.py <input_dir> <output.gguf> --fmt fp4   # FP4-E2M1 (max compression, lower accuracy)
    python3 convert_to_gguf.py <input_dir> <output.gguf> --fmt mixed  # Auto: FP16 for norms/biases, FP8 for weights

Dependencies: numpy, transformers (optional, for config extraction)
"""
import argparse
import glob
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np


# ── GGUF constants ───────────────────────────────────────────────────────────
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGML types (aligned with c/gguf.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 3
GGML_TYPE_Q4_0 = 2
GGML_TYPE_NVFP4 = 40  # OCP FP4 E2M1
GGML_TYPE_NVFP8 = 41  # OCP FP8 E4M3 (custom)


# ── FP4 E2M1 quantization ───────────────────────────────────────────────────
"""
OCP FP4 E2M1 encoding (4 bits per value):
  bit 3: sign
  bits 2-1: exponent (2 bits, bias=1)
  bit 0: mantissa (1 bit)

  Value = (-1)^sign × (1 + mant/2) × 2^(exp - 1)
  exp=0: subnormal → mant/2 × 2^(-1) = 0, 0.5
  exp=1: (1+0/2)×2^0=1, (1+1/2)×2^0=1.5
  exp=2: (1+0/2)×2^1=2, (1+1/2)×2^1=3
  exp=3: (1+0/2)×2^2=4, (1+1/2)×2^2=6

  Positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
  Max representable: ±6.0
  Scale: absmax / 6.0
"""
FP4_E2M1_REPS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def fp4_encode_scalar(x):
    """Encode a single float to FP4 E2M1 code (0-7 unsigned)."""
    sign = 0
    if x < 0:
        x = -x
        sign = 8

    if x <= 0.0:
        return 0

    # Find nearest representable value
    reps = FP4_E2M1_REPS
    diffs = np.abs(reps - x)
    best = int(np.argmin(diffs))
    return sign | best


def fp4_quantize_row(row):
    """Quantize a 1D row to FP4 E2M1, return (packed_uint8, scale)."""
    absmax = np.max(np.abs(row))
    if absmax < 1e-8:
        scale = 1.0
    else:
        scale = absmax / 6.0

    scaled = row / scale

    # Encode each value
    codes = np.zeros(len(row), dtype=np.uint8)
    for i, v in enumerate(scaled):
        codes[i] = fp4_encode_scalar(float(v))

    # Pack 2 values per byte (LSB first)
    n = len(row)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    for i in range(n):
        byte_idx = i // 2
        bit_pos = (i % 2) * 4
        packed[byte_idx] |= codes[i] << bit_pos

    return packed, scale


def fp4_quantize_2d(w2d):
    """
    Quantize a 2D weight matrix row-by-row to FP4 E2M1.
    Returns (packed_uint8, per_row_scales).
    """
    O, I = w2d.shape
    nvals = O * I
    npacked = (nvals + 1) // 2
    packed = np.zeros(npacked, dtype=np.uint8)
    scales = np.zeros(O, dtype=np.float32)

    for o in range(O):
        row = w2d[o].flatten()
        absmax = np.max(np.abs(row))
        if absmax < 1e-8:
            scales[o] = 1.0
        else:
            scales[o] = absmax / 6.0
        scaled = row / scales[o]

        for i in range(I):
            code = fp4_encode_scalar(float(scaled[i]))
            byte_idx = o * I + i
            packed_idx = byte_idx // 2
            bit_pos = (byte_idx % 2) * 4
            packed[packed_idx] |= code << bit_pos

    return packed, scales


# ── FP8 E4M3 quantization ────────────────────────────────────────────────────
"""
OCP FP8 E4M3 encoding (8 bits per value):
  bit 7: sign
  bits 6-4: exponent (3 bits, bias=7)
  bits 3-0: mantissa (4 bits)

  Value = (-1)^sign × (1 + mant/8) × 2^(exp - 7)
  Max representable: 448.0
  Min normal: 0.0625
  Subnormals: down to ~0.000061
  Scale: absmax / 448.0
"""
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 0.0625


def fp8_encode_scalar(x):
    """Encode a single float to FP8 E4M3 code (0-255 unsigned)."""
    sign = 0
    if x < 0:
        x = -x
        sign = 0x80

    if x == 0.0:
        return 0

    if x < FP8_E4M3_MIN_NORMAL:
        # Subnormal: exp=0, mant > 0
        # val = mant * 2^(-8) => mant = val * 256
        m = int(round(x * 256.0))
        m = max(1, min(m, 7))  # subnormals: mant in [1, 7]
        return sign | m
    else:
        # Normal: val = 2^(exp-7) * (1 + mant/8)
        # exp range: 1..14 (biased by 7), mantissa 0..7
        if x >= FP8_E4M3_MAX:
            # Clip to max representable: sign=0, exp=1110, mant=111 = 0x7E
            return 0x7E if sign == 0 else 0xFE
        import math
        exp_bits = min(int(math.log2(max(x, FP8_E4M3_MIN_NORMAL))) + 7, 14)
        if exp_bits < 1:
            exp_bits = 1
        mant = int(round((x / (2 ** (exp_bits - 7)) - 1.0) * 8))
        mant = max(0, min(mant, 7))
        return sign | (exp_bits << 3) | mant


def fp8_quantize_2d_fast(w2d):
    """
    Fully vectorized FP8 E4M3 quantization.
    Returns (fp8_uint8, per_row_scales).
    """
    O, I = w2d.shape
    scales = np.zeros(O, dtype=np.float32)
    fp8 = np.zeros((O, I), dtype=np.uint8)

    for o in range(O):
        row = w2d[o].flatten()
        absmax = np.max(np.abs(row))
        if absmax < 1e-8:
            scales[o] = 1.0
        else:
            scales[o] = absmax / FP8_E4M3_MAX

        if scales[o] == 1.0:
            continue

        scaled = row / scales[o]
        abs_scaled = np.abs(scaled)
        signs = (scaled < 0).astype(np.uint8)

        # Handle zero values
        zero_mask = abs_scaled < 1e-8

        # Handle subnormal values (abs_scaled < FP8_E4M3_MIN_NORMAL)
        subnormal_mask = (~zero_mask) & (abs_scaled < FP8_E4M3_MIN_NORMAL)
        # Subnormal: mant = round(val * 256), clamp to [1, 7]
        subnormal_mant = np.round(abs_scaled[subnormal_mask] * 256.0).astype(np.int32)
        subnormal_mant = np.clip(subnormal_mant, 1, 7)

        # Handle normal values (abs_scaled >= FP8_E4M3_MIN_NORMAL)
        normal_mask = ~zero_mask & ~subnormal_mask
        if np.any(normal_mask):
            normal_vals = abs_scaled[normal_mask]
            import math
            # exp = floor(log2(val)) + 7, clamped to [1, 14]
            normal_exp = np.floor(np.log2(np.maximum(normal_vals, FP8_E4M3_MIN_NORMAL))).astype(np.int32) + 7
            normal_exp = np.clip(normal_exp, 1, 14)
            # mant = round((val / 2^(exp-7) - 1) * 8), clamped to [0, 7]
            normal_mant = np.round((normal_vals / (2.0 ** (normal_exp - 7)) - 1.0) * 8.0).astype(np.int32)
            normal_mant = np.clip(normal_mant, 0, 7)

        # Handle overflow (abs_scaled >= FP8_E4M3_MAX)
        overflow_mask = ~zero_mask & ~subnormal_mask & ~normal_mask
            # Clip to max representable: 0x7E (sign=0) or 0xFE (sign=1)

        # Build codes
        codes = np.zeros(I, dtype=np.uint8)
        codes[zero_mask] = 0
        codes[subnormal_mask] = subnormal_mant & 0x07  # subnormals: sign|mant
        codes[normal_mask] = (normal_exp << 3) | normal_mant
        # Add sign bits
        codes |= (signs << 7)

        fp8[o] = codes

    return fp8, scales


def fp8_quantize_row(row):
    """Quantize a 1D row to FP8 E4M3, return (packed_uint8, scale)."""
    absmax = np.max(np.abs(row))
    if absmax < 1e-8:
        scale = 1.0
    else:
        scale = absmax / FP8_E4M3_MAX

    scaled = row / scale

    # Encode each value
    codes = np.array([fp8_encode_scalar(float(v)) for v in scaled], dtype=np.uint8)
    return codes, scale


def fp8_quantize_2d(w2d):
    """
    Quantize a 2D weight matrix row-by-row to FP8 E4M3.
    Returns (fp8_uint8, per_row_scales).
    """
    O, I = w2d.shape
    fp8 = np.zeros((O, I), dtype=np.uint8)
    scales = np.zeros(O, dtype=np.float32)

    for o in range(O):
        row = w2d[o].flatten()
        absmax = np.max(np.abs(row))
        if absmax < 1e-8:
            scales[o] = 1.0
        else:
            scales[o] = absmax / FP8_E4M3_MAX
        scaled = row / scales[o]

        for i in range(I):
            fp8[o, i] = fp8_encode_scalar(float(scaled[i]))

    return fp8, scales


# ── BF16 ↔ FP32 conversion ─────────────────────────────────────────────────
def bf16_to_f32(bf16_arr):
    """Convert BF16 numpy array to FP32."""
    # BF16: 1 sign, 8 exp, 7 mantissa
    # FP32: 1 sign, 8 exp, 23 mantissa
    # Simply shift left by 16 bits
    u32 = bf16_arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def f16_to_f32(f16_arr):
    """Convert FP16 numpy array to FP32."""
    return f16_arr.astype(np.float32)


# ── Safetensors reader ───────────────────────────────────────────────────────
def read_safetensors_file(filepath):
    """Read a single safetensors file and return dict of {name: numpy_array (FP32)}."""
    f = open(filepath, 'rb')

    # Read header size
    header_size_bytes = f.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]

    # Read header JSON
    header_bytes = f.read(header_size)
    header = json.loads(header_bytes)

    # Read tensor data
    tensors = {}
    offset = 8 + header_size

    for name, meta in header.items():
        if not isinstance(meta, dict) or 'dtype' not in meta:
            continue

        dtype_str = meta['dtype']
        shape = meta['shape']
        data_off = meta['data_offsets']

        f.seek(offset + data_off[0])
        raw = f.read(data_off[1] - data_off[0])

        if dtype_str == 'BF16':
            arr_u16 = np.frombuffer(raw, dtype='<u2')
            tensors[name] = bf16_to_f32(arr_u16).reshape(shape)
        elif dtype_str == 'F16':
            tensors[name] = f16_to_f32(np.frombuffer(raw, dtype='<u2').view(np.float16).reshape(shape))
        elif dtype_str == 'F32':
            tensors[name] = np.frombuffer(raw, dtype='<f4').reshape(shape)
        elif dtype_str == 'I64':
            tensors[name] = np.frombuffer(raw, dtype='<u8').reshape(shape)
        elif dtype_str == 'I32':
            tensors[name] = np.frombuffer(raw, dtype='<u4').reshape(shape)
        elif dtype_str == 'U8':
            tensors[name] = np.frombuffer(raw, dtype='<u1').reshape(shape)
        else:
            print(f"  WARNING: unsupported dtype {dtype_str} for {name}")

    f.close()
    return tensors


def read_all_safetensors(input_dir):
    """Read all safetensors files in input_dir and merge tensors."""
    safetensors_files = sorted(glob.glob(os.path.join(input_dir, '*.safetensors')))
    if not safetensors_files:
        print(f"Error: No .safetensors files found in {input_dir}")
        sys.exit(1)

    all_tensors = {}
    for sf_file in safetensors_files:
        print(f"Reading {os.path.basename(sf_file)}...")
        tensors = read_safetensors_file(sf_file)
        all_tensors.update(tensors)

    print(f"Total tensors: {len(all_tensors)}")
    return all_tensors


# ── GGUF writer ──────────────────────────────────────────────────────────────
class GGUFWriter:
    """Write GGUF v3 file with pre-quantized tensors."""

    def __init__(self, filepath):
        self.fp = open(filepath, 'wb')
        self.tensors = []  # (name, data_bytes, scales_bytes, dtype, shape)
        self.kv_pairs = []

    def add_kv(self, key, value):
        """Add a key-value pair."""
        self.kv_pairs.append((key, value))

    def add_tensor(self, name, data_bytes, scales_bytes, dtype, shape):
        """Add a pre-quantized tensor. data_bytes and scales_bytes are already serialized."""
        self.tensors.append((name, data_bytes, scales_bytes, dtype, shape))

    def write(self):
        """Write the complete GGUF file."""
        n_tensors = len(self.tensors)
        n_kv = len(self.kv_pairs)

        # ── Header ──
        self.fp.write(GGUF_MAGIC)
        self.fp.write(struct.pack('<I', GGUF_VERSION))
        self.fp.write(struct.pack('<Q', n_tensors))
        self.fp.write(struct.pack('<Q', n_kv))

        # ── KV pairs ──
        for key, value in self.kv_pairs:
            self._write_str(key)
            self._write_value(value)

        # ── Tensor info ──
        for name, data_bytes, scales_bytes, dtype, shape in self.tensors:
            self._write_str(name)
            self.fp.write(struct.pack('<I', dtype))
            self.fp.write(struct.pack('<I', len(shape)))
            for dim in shape:
                self.fp.write(struct.pack('<Q', dim))

        # ── Tensor data offsets ──
        # Compute actual offsets
        # GGUF v3 header: 4(magic) + 4(version) + 8(tensor_count) + 8(kv_count) = 24
        kv_data_size = self._compute_kv_data_size()
        tensor_info_size = sum(8 + len(name.encode()) + 4 + 4 + len(shape)*8 for name, _, _, _, shape in self.tensors)
        # Per-tensor info: name(8+len) + dtype(4) + ndim(4) + shape(ndim*8)
        offset = 24 + kv_data_size + tensor_info_size + n_tensors * 8  # header(24) + kv + info + offsets section

        # Compute all data offsets first (offsets must point to actual data, AFTER padding)
        offsets = []
        data_pos = offset
        for name, data_bytes, scales_bytes, dtype, shape in self.tensors:
            # Pad to 8-byte alignment BEFORE recording offset
            if data_pos % 8 != 0:
                pad = 8 - (data_pos % 8)
                data_pos += pad
            offsets.append(data_pos)  # Now points to actual data
            data_len = len(data_bytes)
            data_pos += data_len
            if scales_bytes:
                scales_len = len(scales_bytes)
                if data_pos % 8 != 0:
                    pad = 8 - (data_pos % 8)
                    data_pos += pad
                data_pos += scales_len

        # Write offsets (contiguous uint64, no padding)
        for off in offsets:
            self.fp.write(struct.pack('<Q', off))

        # ── Tensor data ──
        data_pos = offset
        for name, data_bytes, scales_bytes, dtype, shape in self.tensors:
            # Align to 8-byte boundary
            if data_pos % 8 != 0:
                pad = 8 - (data_pos % 8)
                self.fp.write(b'\x00' * pad)
                data_pos += pad
            self.fp.write(data_bytes)
            data_pos += len(data_bytes)
            if scales_bytes:
                if data_pos % 8 != 0:
                    pad = 8 - (data_pos % 8)
                    self.fp.write(b'\x00' * pad)
                    data_pos += pad
                self.fp.write(scales_bytes)
                data_pos += len(scales_bytes)

        self.fp.close()

    def _write_str(self, s):
        data = s.encode('utf-8')
        self.fp.write(struct.pack('<Q', len(data)))
        self.fp.write(data)

    def _write_value(self, v):
        if isinstance(v, str):
            self.fp.write(struct.pack('<I', 2))  # GGUF_TYPE_STRING (v3 spec)
            self._write_str(v)
        elif isinstance(v, int):
            self.fp.write(struct.pack('<I', 5))  # GGUF_TYPE_U32 (v3 spec)
            self.fp.write(struct.pack('<I', v))
        elif isinstance(v, float):
            self.fp.write(struct.pack('<I', 7))  # GGUF_TYPE_F32 (v3 spec)
            self.fp.write(struct.pack('<f', v))
        elif isinstance(v, bool):
            self.fp.write(struct.pack('<I', 0))  # GGUF_TYPE_BOOL (v3 spec)
            self.fp.write(struct.pack('<B', 1 if v else 0))
        else:
            raise ValueError(f"Unsupported KV value type: {type(v)}")

    def _compute_kv_data_size(self):
        size = 0
        for key, value in self.kv_pairs:
            size += 8 + len(key.encode('utf-8'))  # key: Q length + data
            size += 4  # type: I 4 bytes
            if isinstance(value, str):
                size += 8 + len(value.encode('utf-8'))  # value: Q length + data
            elif isinstance(value, (int,)):
                size += 4  # value: I 4 bytes
            elif isinstance(value, float):
                size += 4  # value: F 4 bytes
            elif isinstance(value, bool):
                size += 1  # value: B 1 byte
        return size


# ── Model config extraction ─────────────────────────────────────────────────
def extract_config(input_dir):
    """Extract model config from safetensors directory."""
    config_path = os.path.join(input_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {input_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get('text_config', config)
    return config, text_config


# ── Main conversion function ─────────────────────────────────────────────────
def convert_model(input_dir, output_path, fmt='fp16'):
    """
    Convert safetensors to GGUF with pre-quantized weights.

    Args:
        input_dir: Directory containing safetensors files and config.json
        output_path: Output GGUF file path
        fmt: Quantization format ('fp16', 'fp8', 'fp4', 'mixed')
    """
    config, text_config = extract_config(input_dir)

    hidden_size = text_config['hidden_size']
    num_hidden_layers = text_config['num_hidden_layers']
    num_attention_heads = text_config['num_attention_heads']
    num_key_value_heads = text_config['num_key_value_heads']
    head_dim = text_config.get('head_dim', hidden_size // num_attention_heads)
    vocab_size = text_config['vocab_size']
    rms_norm_eps = text_config.get('rms_norm_eps', text_config.get('attention_dropout', 1e-6))
    rope_theta = text_config.get('rope_theta', text_config.get('rope_parameters', {}).get('rope_theta', 10000000))
    layer_types = text_config.get('layer_types', [])
    linear_num_key_heads = text_config.get('linear_num_key_heads', num_attention_heads)
    linear_num_value_heads = text_config.get('linear_num_value_heads', num_attention_heads)
    linear_key_head_dim = text_config.get('linear_key_head_dim', head_dim)
    linear_value_head_dim = text_config.get('linear_value_head_dim', head_dim)
    linear_conv_kernel_dim = text_config.get('linear_conv_kernel_dim', 4)
    mlp_intermediate_size = text_config['intermediate_size']
    tie_word_embeddings = config.get('tie_word_embeddings', True)
    attn_output_gate = text_config.get('attn_output_gate', True)
    architectures = config.get('architectures', ['Qwen3_5ForConditionalGeneration'])

    print(f"\nModel: {architectures[0]}")
    print(f"  hidden={hidden_size}, layers={num_hidden_layers}, "
          f"heads={num_attention_heads}, kv_heads={num_key_value_heads}, "
          f"vocab={vocab_size}, fmt={fmt}")

    # Read all safetensors
    all_tensors = read_all_safetensors(input_dir)

    # ── Determine quantization for each tensor ──
    def should_quantize(name):
        """Decide if a tensor should be quantized (vs kept in FP16)."""
        # Keep in FP16: norms, biases, small tensors, A_log, dt_bias
        if '.norm.weight' in name or '.ln' in name:
            return False
        if '.bias' in name or '.bias.' in name:
            return False
        if name.endswith('.A_log') or name.endswith('.dt_bias'):
            return False
        if 'ln_w' in name or 'ln_b' in name:
            return False
        # Keep in FP16: embeddings and lm_head for accuracy
        if 'embed_tokens' in name or 'lm_head' in name:
            return fmt != 'fp16'  # Quantize only if not fp16 mode
        # Quantize large weight matrices
        return True

    # ── Build GGUF writer ──
    writer = GGUFWriter(output_path)

    # KV pairs
    writer.add_kv("general.architecture", "qwen3_5_text")
    writer.add_kv("general.name", config.get('name', 'Qwen3.5'))
    writer.add_kv("general.file_type", {
        'fp16': 1, 'fp8': 2, 'fp4': 3, 'mixed': 1
    }[fmt])
    writer.add_kv("qwen3_5_text.block_count", num_hidden_layers)
    writer.add_kv("qwen3_5_text.context_length", text_config.get('max_position_embeddings', 4096))
    writer.add_kv("qwen3_5_text.embedding_length", hidden_size)
    writer.add_kv("qwen3_5_text.attention.head_count", num_attention_heads)
    writer.add_kv("qwen3_5_text.attention.head_count_kv", num_key_value_heads)
    writer.add_kv("qwen3_5_text.attention.layer_norm_rms_epsilon", rms_norm_eps)
    writer.add_kv("qwen3_5_text.rope.freq_base", float(rope_theta))
    writer.add_kv("qwen3_5_text.attention.head_dim", float(head_dim))
    writer.add_kv("qwen3_5_text.attention.attn_output_gate", 1 if attn_output_gate else 0)
    writer.add_kv("qwen3_5_text.padded_vocab_size", vocab_size)
    writer.add_kv("qwen3_5_text.vocab_size", vocab_size)
    writer.add_kv("tokenizer.ggml.pre", "default")
    writer.add_kv("tokenizer.ggml.model", "bpe")

    # Linear attention KV pairs (if present)
    if linear_num_key_heads > 0:
        writer.add_kv("qwen3_5_text.linear_attention.num_key_heads", linear_num_key_heads)
        writer.add_kv("qwen3_5_text.linear_attention.num_value_heads", linear_num_value_heads)
        writer.add_kv("qwen3_5_text.linear_attention.key_head_dim", linear_key_head_dim)
        writer.add_kv("qwen3_5_text.linear_attention.value_head_dim", linear_value_head_dim)
        writer.add_kv("qwen3_5_text.linear_attention.conv_kernel_dim", linear_conv_kernel_dim)

    # MLP KV pair
    writer.add_kv("qwen3_5_text.feed_forward_length", mlp_intermediate_size)

    # ── Process tensors ──
    total_params = 0
    total_params_quantized = 0

    for name, tensor in sorted(all_tensors.items()):
        if not should_quantize(name):
            # Keep in FP16
            # The safetensors reader already converts all dtypes (BF16, F16, F32) to F32 numpy arrays.
            # Converting F32 -> F16 via astype(np.float16) preserves precision correctly.
            data_f16 = tensor.astype(np.float16).tobytes()
            scales = None
            dtype = GGML_TYPE_F16
            params = int(np.prod(tensor.shape))
            total_params += params
            print(f"  FP16:  {name:60s} {str(tensor.shape):20s} {params:>10,} params")
        elif fmt in ('fp8', 'mixed'):
            # Quantize to FP8-E4M3
            if tensor.ndim == 1:
                # 1D tensor: use per-element quantization
                fp8_data, scales = fp8_quantize_row(tensor.flatten())
                data_bytes = fp8_data.tobytes()
                scales_bytes = scales.astype(np.float32).tobytes()
                dtype = GGML_TYPE_NVFP8
                params = tensor.size
                total_params += params
                total_params_quantized += params
                print(f"  FP8:   {name:60s} {str(tensor.shape):20s} {params:>10,} params -> FP8")
            elif tensor.ndim == 2:
                # 2D+ tensor: row-by-row quantization (use fast version)
                fp8_data, scales = fp8_quantize_2d_fast(tensor)
                data_bytes = fp8_data.tobytes()
                scales_bytes = scales.astype(np.float32).tobytes()
                dtype = GGML_TYPE_NVFP8
                params = int(np.prod(tensor.shape))
                total_params += params
                total_params_quantized += params
                print(f"  FP8:   {name:60s} {str(tensor.shape):20s} {params:>10,} params -> FP8")
            else:
                # Higher dimensions: flatten and quantize
                flat = tensor.flatten()
                fp8_data, scales = fp8_quantize_row(flat)
                data_bytes = fp8_data.tobytes()
                scales_bytes = scales.astype(np.float32).tobytes()
                dtype = GGML_TYPE_NVFP8
                params = flat.size
                total_params += params
                total_params_quantized += params
                print(f"  FP8:   {name:60s} {str(tensor.shape):20s} {params:>10,} params -> FP8")
        elif fmt == 'fp4':
            # Quantize to FP4-E2M1
            if tensor.ndim == 1:
                packed, scale = fp4_quantize_row(tensor.flatten())
                data_bytes = packed.tobytes()
                scales_bytes = np.array([scale], dtype=np.float32).tobytes()
                dtype = GGML_TYPE_NVFP4
                params = tensor.size
                total_params += params
                total_params_quantized += params
                print(f"  FP4:   {name:60s} {str(tensor.shape):20s} {params:>10,} params -> FP4")
            elif tensor.ndim == 2:
                packed, scales = fp4_quantize_2d(tensor)
                data_bytes = packed.tobytes()
                scales_bytes = scales.astype(np.float32).tobytes()
                dtype = GGML_TYPE_NVFP4
                params = int(np.prod(tensor.shape))
                total_params += params
                total_params_quantized += params
                print(f"  FP4:   {name:60s} {str(tensor.shape):20s} {params:>10,} params -> FP4")
            else:
                flat = tensor.flatten()
                packed, scale = fp4_quantize_row(flat)
                data_bytes = packed.tobytes()
                scales_bytes = np.array([scale], dtype=np.float32).tobytes()
                dtype = GGML_TYPE_NVFP4
                params = flat.size
                total_params += params
                total_params_quantized += params
                print(f"  FP4:   {name:60s} {str(tensor.shape):20s} {params:>10,} params -> FP4")
        else:
            # FP16 mode
            data_f16 = tensor.astype(np.float16).tobytes()
            scales = None
            dtype = GGML_TYPE_F16
            params = int(np.prod(tensor.shape))
            total_params += params
            print(f"  FP16:  {name:60s} {str(tensor.shape):20s} {params:>10,} params")

        # Add tensor to writer
        if fmt in ('fp8', 'fp4', 'mixed'):
            writer.add_tensor(name, data_bytes, scales_bytes, dtype, list(tensor.shape))
        else:
            writer.add_tensor(name, data_f16, None, dtype, list(tensor.shape))

    # ── Write GGUF ──
    print(f"\nWriting {output_path}...")
    writer.write()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  Total params: {total_params:,} ({total_params * 2 / 1e6:.1f} MB in FP16)")
    print(f"  Quantized: {total_params_quantized:,}")
    print(f"  Format: {fmt}")
    print(f"{'='*70}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Convert safetensors to GGUF with pre-quantized weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert_to_gguf.py ./model ./model.gguf --fmt fp16
  python3 convert_to_gguf.py ./model ./model.gguf --fmt fp8
  python3 convert_to_gguf.py ./model ./model.gguf --fmt fp4
  python3 convert_to_gguf.py ./model ./model.gguf --fmt mixed
        """
    )
    parser.add_argument('input_dir', help='Directory containing safetensors files and config.json')
    parser.add_argument('output_gguf', help='Output GGUF file path')
    parser.add_argument('--fmt', choices=['fp16', 'fp8', 'fp4', 'mixed'], default='fp16',
                       help='Quantization format (default: fp16)')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        sys.exit(1)

    if not os.path.exists(os.path.join(args.input_dir, 'config.json')):
        print(f"Error: config.json not found in '{args.input_dir}'")
        sys.exit(1)

    convert_model(args.input_dir, args.output_gguf, args.fmt)


if __name__ == '__main__':
    main()
