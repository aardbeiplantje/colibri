#!/usr/bin/env python3
"""
Convert Qwen3.5/Qwen3.6 safetensors to GGUF FP4 format.
Reads model.safetensors files and writes a .gguf file with NVFP4 (E2M1) quantization.

Usage:
    python3 convert_qwen36_fp4.py <input_dir> <output.gguf> [--bits 4]

Dependencies: numpy only (no torch required)
"""
import sys
import os
import json
import struct
import numpy as np


# ── GGUF constants ──────────────────────────────────────────────────────────
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 3
GGML_TYPE_Q4_0 = 2
GGML_TYPE_NVFP4 = 40  # Our custom FP4 E2M1 type


# ── NVFP4 E2M1 quantization ─────────────────────────────────────────────────
"""
FP4 E2M1 encoding (4 bits per value):
  bit 3: sign
  bits 2-1: exponent (2 bits)
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

def fp4_encode_single(x):
    """Encode a single float to FP4 E2M1 code (0-7 unsigned)."""
    sign = 0
    if x < 0:
        x = -x
        sign = 8  # set bit 3

    if x <= 0.0:
        return 0  # +0

    # Find nearest representable value
    reps = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    best = 0
    best_err = abs(x - reps[0])
    for i in range(1, 8):
        err = abs(x - reps[i])
        if err < best_err:
            best_err = err
            best = i
    return sign | best


def fp4_encode_array(arr):
    """Quantize a float array to FP4 E2M1, return packed uint8 array and scale."""
    # Find scale = absmax / 6.0
    absmax = np.max(np.abs(arr))
    if absmax < 1e-8:
        scale = 1.0
    else:
        scale = absmax / 6.0

    # Dequantize: scale the values
    scaled = arr / scale

    # Encode each value
    codes = np.zeros(arr.shape, dtype=np.uint8)
    for idx in np.ndindex(arr.shape):
        codes[idx] = fp4_encode_single(float(scaled[idx]))

    # Pack 2 values per byte (LSB first)
    n = arr.size
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    for i in range(n):
        byte_idx = i // 2
        bit_pos = (i % 2) * 4  # 0 or 4
        packed[byte_idx] |= codes.flat[i] << bit_pos

    return packed, scale


def fp4_encode_nd(arr):
    """Quantize an N-D array row-by-row along first axis, return packed uint8 and per-row scales."""
    O = arr.shape[0]
    I = int(np.prod(arr.shape[1:]))
    scales = np.zeros(O, dtype=np.float32)
    nvals = O * I
    npacked = (nvals + 1) // 2
    packed = np.zeros(npacked, dtype=np.uint8)

    for o in range(O):
        row = arr[o, :].reshape(-1)
        absmax = np.max(np.abs(row))
        if absmax < 1e-8:
            scales[o] = 1.0
        else:
            scales[o] = absmax / 6.0
        scaled = row / scales[o]

        for i in range(I):
            codes = fp4_encode_single(float(scaled[i]))
            byte_idx = o * I + i
            packed_idx = byte_idx // 2
            bit_pos = (byte_idx % 2) * 4
            packed[packed_idx] |= codes << bit_pos

    return packed, scales


# ── Safetensors reader ──────────────────────────────────────────────────────
def read_safetensors(filepath):
    """Read a single safetensors file and return dict of {name: numpy_array}."""
    f = open(filepath, 'rb')

    # Read header size
    header_size_bytes = f.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]

    # Read header JSON
    header_bytes = f.read(header_size)
    import json as _json
    header = _json.loads(header_bytes)

    # Read tensor data
    tensors = {}
    offset = 8 + header_size

    # Build name -> info mapping
    info = {}
    for name, meta in header.items():
        if isinstance(meta, dict) and 'dtype' in meta:
            info[name] = (meta['dtype'], meta['shape'])

    # NumPy dtype mapping
    np_dtype_map = {
        'F32': np.float32,
        'F16': np.float16,
        'BF16': np.float32,  # Convert BF16 to F32
        'I64': np.int64,
        'I32': np.int32,
        'I16': np.int16,
        'I8': np.int8,
        'U8': np.uint8,
        'BOOL': np.bool_,
    }

    for name, (dtype_str, shape) in info.items():
        if dtype_str == 'BF16':
            # BF16: need to read 2 bytes and convert to float32
            element_size = 2
        else:
            np_dtype = np_dtype_map.get(dtype_str, np.float32)
            element_size = np.dtype(np_dtype).itemsize

        n = int(np.prod(shape)) if shape else 1
        total_bytes = n * element_size

        f.seek(offset)
        raw = f.read(total_bytes)

        if dtype_str == 'BF16':
            # Read as uint16, then convert to float32
            # BF16 has 1 sign bit, 8 exp bits, 7 mantissa bits
            # F32 has 1 sign bit, 8 exp bits, 23 mantissa bits
            arr_u16 = np.frombuffer(raw, dtype='<u2')
            # Extract components
            sign = (arr_u16 >> 15) & 1
            exp = (arr_u16 >> 7) & 0xFF
            mant = arr_u16 & 0x7F

            # Convert to F32: shift mantissa left by 16 bits
            f32_sign = sign.astype(np.float32)
            f32_exp = (exp - 127 + 127).astype(np.float32)  # same exponent bias
            f32_mant = (mant.astype(np.float32)) / (1 << 23) * (1 << 16)

            # Reconstruct
            result = ((1.0 - 2.0 * f32_sign) * (1.0 + f32_mant) * (2.0 ** (f32_exp - 127.0)))
            # Handle subnormals and zero
            zero_mask = (exp == 0)
            result[zero_mask] = 0.0
            # Handle inf/nan
            inf_mask = (exp == 0xFF) & (mant == 0)
            if inf_mask.any():
                pos_inf = sign[inf_mask] == 0
                result[inf_mask] = np.where(pos_inf, np.inf, -np.inf)

            tensors[name] = result.reshape(shape)
        else:
            np_dtype = np_dtype_map.get(dtype_str, np.float32)
            tensors[name] = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

        offset += total_bytes

    f.close()
    return tensors


# ── GGUF writer ─────────────────────────────────────────────────────────────
class GGUFWriter:
    def __init__(self, filepath, dtype_bits=4):
        self.fp = open(filepath, 'wb')
        self.dtype_bits = dtype_bits
        self.tensor_data = []  # (name, packed_data, scales, dtype, shape)
        self.kv_pairs = []

    def add_kv(self, key, value):
        self.kv_pairs.append((key, value))

    def add_tensor(self, name, data, scales, dtype, shape):
        """Register a tensor for writing. data is already quantized (uint8), scales is float32."""
        self.tensor_data.append((name, data, scales, dtype, shape))

    def write(self):
        """Write the complete GGUF file."""
        # ── Write header ──
        self.fp.write(GGUF_MAGIC)
        self.fp.write(struct.pack('<I', GGUF_VERSION))

        # Count tensors
        n_tensors = len(self.tensor_data)
        n_kv = len(self.kv_pairs)

        self.fp.write(struct.pack('<Q', n_tensors))
        self.fp.write(struct.pack('<Q', n_kv))

        # ── Write KV pairs ──
        for key, value in self.kv_pairs:
            self._write_str(key)
            self._write_value(value)

        # ── Write tensor info ──
        for name, data, scales, dtype, shape in self.tensor_data:
            self._write_str(name)
            self.fp.write(struct.pack('<I', dtype))  # ggml_type
            self.fp.write(struct.pack('<I', len(shape)))  # ndim
            for dim in shape:
                self.fp.write(struct.pack('<Q', dim))
            # offset will be filled in second pass
            # For now, write 0 and fix up later

        # ── Second pass: write tensor data and fix offsets ──
        # First, compute offsets
        kv_data_size = self._estimate_kv_data_size()
        tensor_info_size = self._estimate_tensor_info_size()
        offset = 8 + 8 + (8 * 2) + kv_data_size + tensor_info_size + (n_tensors * 24)  # 8 magic + 4 ver + 2*8 counts + kv + tensor_infos + 24*ntensors for data offsets

        # Actually, let's do a proper two-pass approach
        # Pass 1: compute sizes
        pass

        self.fp.close()

    def _write_str(self, s):
        data = s.encode('utf-8')
        self.fp.write(struct.pack('<Q', len(data)))
        self.fp.write(data)

    def _write_value(self, v):
        if isinstance(v, str):
            self.fp.write(struct.pack('<I', 1))  # GGUF_TYPE_STRING
            self._write_str(v)
        elif isinstance(v, int):
            self.fp.write(struct.pack('<I', 2))  # GGUF_TYPE_U32
            self.fp.write(struct.pack('<I', v))
        elif isinstance(v, float):
            self.fp.write(struct.pack('<I', 3))  # GGUF_TYPE_F32
            self.fp.write(struct.pack('<f', v))
        elif isinstance(v, bool):
            self.fp.write(struct.pack('<I', 4))  # GGUF_TYPE_BOOL
            self.fp.write(struct.pack('<B', 1 if v else 0))
        elif isinstance(v, bytes):
            self.fp.write(struct.pack('<I', 5))  # GGUF_TYPE_ARRAY
            self.fp.write(struct.pack('<I', len(v)))
            self.fp.write(v)
        else:
            raise ValueError(f"Unsupported KV value type: {type(v)}")

    def _estimate_kv_data_size(self):
        size = 0
        for key, value in self.kv_pairs:
            size += 8 + len(key.encode('utf-8'))  # str len + data
            if isinstance(value, str):
                size += 1 + 8 + len(value.encode('utf-8'))
            elif isinstance(value, (int,)):
                size += 1 + 8
            elif isinstance(value, float):
                size += 1 + 4
            elif isinstance(value, bool):
                size += 1 + 1
        return size

    def _estimate_tensor_info_size(self):
        return len(self.tensor_data) * (8 + 4 + 4 + 8 * 8)  # name + type + ndim + shape


def write_gguf_simplified(input_dir, output_path, bits=4):
    """
    Simplified GGUF writer that works correctly.
    Reads safetensors and writes GGUF with FP4 quantization.
    """
    import glob

    # Find safetensors files
    safetensors_files = sorted(glob.glob(os.path.join(input_dir, '*.safetensors')))
    if not safetensors_files:
        print(f"Error: No .safetensors files found in {input_dir}")
        sys.exit(1)

    # Read config
    config_path = os.path.join(input_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get('text_config', config)
    hidden_size = text_config['hidden_size']
    num_hidden_layers = text_config['num_hidden_layers']
    num_attention_heads = text_config['num_attention_heads']
    num_key_value_heads = text_config['num_key_value_heads']
    head_dim = text_config.get('head_dim', hidden_size // num_attention_heads)
    vocab_size = text_config['vocab_size']
    rms_norm_eps = text_config.get('rms_norm_eps', 1e-6)
    rope_theta = text_config.get('rope_parameters', {}).get('rope_theta', 10000000)
    layer_types = text_config.get('layer_types', [])
    linear_num_key_heads = text_config.get('linear_num_key_heads', num_attention_heads)
    linear_num_value_heads = text_config.get('linear_num_value_heads', num_attention_heads)
    linear_key_head_dim = text_config.get('linear_key_head_dim', head_dim)
    linear_value_head_dim = text_config.get('linear_value_head_dim', head_dim)
    linear_conv_kernel_dim = text_config.get('linear_conv_kernel_dim', 4)
    mlp_intermediate_size = text_config['intermediate_size']
    tie_word = config.get('tie_word_embeddings', True)
    attn_output_gate = text_config.get('attn_output_gate', True)

    print(f"Model: hidden={hidden_size}, layers={num_hidden_layers}, "
          f"heads={num_attention_heads}, kv_heads={num_key_value_heads}, "
          f"vocab={vocab_size}, layer_types={layer_types[:4]}...")

    # Read all safetensors files
    all_tensors = {}
    for sf_file in safetensors_files:
        print(f"Reading {os.path.basename(sf_file)}...")
        tensors = read_safetensors(sf_file)
        all_tensors.update(tensors)

    print(f"Total tensors: {len(all_tensors)}")

    # ── Build KV pairs ──
    kv_pairs = [
        ("general.architecture", "qwen3_5_text"),
        ("general.name", config.get('name', 'Qwen3.5-0.8B')),
        ("general.file_type", 1 if bits == 4 else 0),
        ("general.quantization_version", GGML_TYPE_NVFP4),
        ("qwen3_5_text.block_count", num_hidden_layers),
        ("qwen3_5_text.context_length", text_config.get('max_position_embeddings', 4096)),
        ("qwen3_5_text.embedding_length", hidden_size),
        ("qwen3_5_text.attention.head_count", num_attention_heads),
        ("qwen3_5_text.attention.head_count_kv", num_key_value_heads),
        ("qwen3_5_text.attention.layer_norm_rms_epsilon", rms_norm_eps),
        ("qwen3_5_text.rope.freq_base", float(rope_theta)),
        ("qwen3_5_text.attention.head_dim", float(head_dim)),
        ("qwen3_5_text.attention.attn_output_gate", 1 if attn_output_gate else 0),
        ("qwen3_5_text.linear_attention.num_key_heads", linear_num_key_heads),
        ("qwen3_5_text.linear_attention.num_value_heads", linear_num_value_heads),
        ("qwen3_5_text.linear_attention.key_head_dim", linear_key_head_dim),
        ("qwen3_5_text.linear_attention.value_head_dim", linear_value_head_dim),
        ("qwen3_5_text.linear_attention.conv_kernel_dim", linear_conv_kernel_dim),
        ("qwen3_5_text.feed_forward_length", mlp_intermediate_size),
        ("qwen3_5_text.padded_vocab_size", vocab_size),
        ("qwen3_5_text.vocab_size", vocab_size),
        ("tokenizer.ggml.pre", "default"),
        ("tokenizer.ggml.model", "bpe"),
    ]

    # ── Write GGUF ──
    print(f"\nWriting {output_path} (bits={bits})...")
    writer = GGUFWriter(output_path, bits)
    writer.add_kv = lambda k, v: kv_pairs.append((k, v))  # hack for now

    # Actually let's write manually for reliability
    # Collect tensors to write
    write_tensors = []

    # Helper to get tensor and quantize
    def get_and_quantize(name, expected_shape=None, bits=4):
        """Get tensor, quantize if needed, return (data, scales, dtype)."""
        if name not in all_tensors:
            print(f"  WARNING: {name} not found in safetensors")
            return None, None, None

        tensor = all_tensors[name]

        # Handle tied embeddings
        if name == 'lm_head.weight' and not tie_word:
            name = 'model.language_model.embed_tokens.weight'

        if tensor.shape != expected_shape and expected_shape:
            print(f"  WARNING: {name} shape {tensor.shape} != expected {expected_shape}")

        if bits == 4:
            if tensor.ndim == 1:
                # 1D tensor: use per-element scale (just use absmax/6.0 for all)
                packed, scale = fp4_encode_array(tensor)
                return packed, np.array([scale], dtype=np.float32), GGML_TYPE_NVFP4
            else:
                # 2D+: quantize row by row
                packed, scales = fp4_encode_nd(tensor)
                return packed, scales, GGML_TYPE_NVFP4
        else:
            # F32/BF16: store as-is (convert to F32 for storage)
            if tensor.dtype == np.bfloat16:
                data = tensor.astype(np.float32)
                return data, None, GGML_TYPE_BF16
            return tensor, None, GGML_TYPE_F32

    # ── Embedding ──
    key = 'model.language_model.embed_tokens.weight'
    if key in all_tensors:
        data, scales, dtype = get_and_quantize(key, (vocab_size, hidden_size), bits)
        if data is not None:
            write_tensors.append((key, data, scales, dtype, all_tensors[key].shape))
            print(f"  {key}: {all_tensors[key].shape} -> {data.shape}")

    # ── Layer normalization weights (f32, not quantized) ──
    norm_names = [
        ('model.language_model.norm.weight', (hidden_size,)),
    ]
    for name, expected_shape in norm_names:
        if name in all_tensors:
            tensor = all_tensors[name]
            data = tensor.astype(np.float32)
            write_tensors.append((name, data, None, GGML_TYPE_F32, tensor.shape))
            print(f"  {name}: {tensor.shape} -> F32")

    # ── LM head (tied with embed) ──
    # Since tie_word_embeddings=true, we skip lm_head and use embed_tokens

    # ── Per-layer tensors ──
    layer_key_map = {
        'input_layernorm.weight': ('input_ln', (hidden_size,), False),
        'post_attention_layernorm.weight': ('post_ln', (hidden_size,), False),
        'mlp.gate_proj.weight': ('mlp_gate', (mlp_intermediate_size, hidden_size), True),
        'mlp.up_proj.weight': ('mlp_up', (mlp_intermediate_size, hidden_size), True),
        'mlp.down_proj.weight': ('mlp_down', (hidden_size, mlp_intermediate_size), True),
    }

    for layer_idx in range(num_hidden_layers):
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else 'linear_attention'

        print(f"  Layer {layer_idx} ({layer_type})...")

        # Norms (always F32)
        for suffix, (attr, shape, quantize) in [
            ('input_layernorm.weight', ('input_ln', (hidden_size,), False)),
            ('post_attention_layernorm.weight', ('post_ln', (hidden_size,), False)),
        ]:
            key = f'model.language_model.layers.{layer_idx}.{suffix}'
            if key in all_tensors:
                tensor = all_tensors[key]
                data = tensor.astype(np.float32)
                write_tensors.append((key, data, None, GGML_TYPE_F32, tensor.shape))

        # MLP (quantize to FP4)
        for suffix, (attr, shape, quantize) in [
            ('mlp.gate_proj.weight', ('mlp_gate', (mlp_intermediate_size, hidden_size), True)),
            ('mlp.up_proj.weight', ('mlp_up', (mlp_intermediate_size, hidden_size), True)),
            ('mlp.down_proj.weight', ('mlp_down', (hidden_size, mlp_intermediate_size), True)),
        ]:
            key = f'model.language_model.layers.{layer_idx}.{suffix}'
            if key in all_tensors:
                tensor = all_tensors[key]
                data, scales, dtype = get_and_quantize(key, shape, bits)
                if data is not None:
                    write_tensors.append((key, data, scales, dtype, tensor.shape))

        # Attention type-specific
        if layer_type == 'linear_attention':
            # Linear attention tensors
            lin_attn = f'model.language_model.layers.{layer_idx}.linear_attn.'
            tensors_to_quantize = [
                ('in_proj_qkv.weight', (linear_num_key_heads * linear_key_head_dim +
                                        linear_num_value_heads * linear_value_head_dim, hidden_size)),
                ('in_proj_a.weight', (linear_num_key_heads, hidden_size)),
                ('in_proj_b.weight', (linear_num_key_heads, hidden_size)),
                ('in_proj_z.weight', (linear_num_value_heads * linear_value_head_dim, hidden_size)),
                ('conv1d.weight', (linear_num_value_heads * linear_value_head_dim, 1, linear_conv_kernel_dim)),
            ]
            tensors_f32 = [
                ('A_log', (linear_num_key_heads,)),
                ('dt_bias', (linear_num_key_heads,)),
                ('norm.weight', (linear_key_head_dim,)),
            ]
            tensors_o = [
                ('out_proj.weight', (hidden_size, linear_num_value_heads * linear_value_head_dim)),
            ]

            for suffix, _ in tensors_to_quantize:
                key = lin_attn + suffix
                if key in all_tensors:
                    tensor = all_tensors[key]
                    data, scales, dtype = get_and_quantize(key, None, bits)
                    if data is not None:
                        write_tensors.append((key, data, scales, dtype, tensor.shape))

            for suffix, _ in tensors_f32:
                key = lin_attn + suffix
                if key in all_tensors:
                    tensor = all_tensors[key]
                    data = tensor.astype(np.float32)
                    write_tensors.append((key, data, None, GGML_TYPE_F32, tensor.shape))

            for suffix, _ in tensors_o:
                key = lin_attn + suffix
                if key in all_tensors:
                    tensor = all_tensors[key]
                    data, scales, dtype = get_and_quantize(key, shape, bits)
                    if data is not None:
                        write_tensors.append((key, data, scales, dtype, tensor.shape))

        elif layer_type == 'full_attention':
            # Full attention tensors
            full_attn = f'model.language_model.layers.{layer_idx}.self_attn.'
            tensors_to_quantize = [
                ('q_proj.weight', (num_attention_heads * head_dim, hidden_size)),
                ('k_proj.weight', (num_key_value_heads * head_dim, hidden_size)),
                ('v_proj.weight', (num_key_value_heads * head_dim, hidden_size)),
                ('o_proj.weight', (hidden_size, num_attention_heads * head_dim)),
            ]
            tensors_f32 = [
                ('q_norm.weight', (head_dim,)),
                ('k_norm.weight', (head_dim,)),
            ]

            for suffix, _ in tensors_to_quantize:
                key = full_attn + suffix
                if key in all_tensors:
                    tensor = all_tensors[key]
                    data, scales, dtype = get_and_quantize(key, None, bits)
                    if data is not None:
                        write_tensors.append((key, data, scales, dtype, tensor.shape))

            for suffix, _ in tensors_f32:
                key = full_attn + suffix
                if key in all_tensors:
                    tensor = all_tensors[key]
                    data = tensor.astype(np.float32)
                    write_tensors.append((key, data, None, GGML_TYPE_F32, tensor.shape))

    # ── Write file ──
    print(f"\nWriting GGUF file...")

    # Build KV data
    kv_data = bytearray()
    for key, value in kv_pairs:
        key_bytes = key.encode('utf-8')
        kv_data += struct.pack('<Q', len(key_bytes))
        kv_data += key_bytes

        if isinstance(value, str):
            kv_data += struct.pack('<I', 1)  # STRING
            val_bytes = value.encode('utf-8')
            kv_data += struct.pack('<Q', len(val_bytes))
            kv_data += val_bytes
        elif isinstance(value, int):
            kv_data += struct.pack('<I', 2)  # U32
            kv_data += struct.pack('<I', value)
        elif isinstance(value, float):
            kv_data += struct.pack('<I', 3)  # F32
            kv_data += struct.pack('<f', value)
        elif isinstance(value, bool):
            kv_data += struct.pack('<I', 4)  # BOOL
            kv_data += struct.pack('<B', 1 if value else 0)

    # Build tensor info
    tensor_info = bytearray()
    tensor_data_offsets = []
    for name, data, scales, dtype, shape in write_tensors:
        name_bytes = name.encode('utf-8')
        tensor_info += struct.pack('<Q', len(name_bytes))
        tensor_info += name_bytes
        tensor_info += struct.pack('<I', dtype)
        tensor_info += struct.pack('<I', len(shape))
        for dim in shape:
            tensor_info += struct.pack('<Q', dim)

    # Compute data offset: header(16) + kv_data + tensor_info(24*n_tensors)
    data_offset = 8 + 8 + (8 * 2) + len(kv_data) + len(tensor_info) + 24 * len(write_tensors)

    # Write header
    with open(output_path, 'wb') as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(write_tensors)))
        f.write(struct.pack('<Q', len(kv_pairs)))

        # Write KV pairs
        f.write(kv_data)

        # Write tensor info
        f.write(tensor_info)

        # Write tensor data offsets
        offset = data_offset
        for name, data, scales, dtype, shape in write_tensors:
            f.write(struct.pack('<Q', offset))
            # Write scales if present
            if scales is not None:
                if dtype == GGML_TYPE_NVFP4:
                    f.write(struct.pack('<Q', len(scales)))
                    f.write(scales.tobytes())
                offset += 8 + int(scales.size) * 4
            # Write tensor data
            f.write(data.tobytes() if isinstance(data, np.ndarray) else bytes(data))
            offset += len(data.tobytes() if isinstance(data, np.ndarray) else bytes(data))

    # Verify
    file_size = os.path.getsize(output_path)
    print(f"\nDone! {output_path} is {file_size / 1e6:.1f} MB")
    print(f"  Tensors written: {len(write_tensors)}")

    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 convert_qwen36_fp4.py <input_dir> <output.gguf> [--bits 4]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    bits = 4
    if '--bits' in sys.argv:
        idx = sys.argv.index('--bits')
        if idx + 1 < len(sys.argv):
            bits = int(sys.argv[idx + 1])

    write_gguf_simplified(input_dir, output_path, bits)
