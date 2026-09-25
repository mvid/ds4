#!/usr/bin/env python3
"""Convert the pinned MiMo V2.6 Flash checkpoint to a text-only mimo2 GGUF."""

import argparse
import concurrent.futures
import ctypes
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import threading
import time

import numpy as np

ALIGNMENT = 32
F32, Q8_0, Q2_K, Q4_K, IQ2_XXS, MXFP4 = 0, 8, 10, 12, 16, 39
LAYOUT = {F32: (1, 4), Q8_0: (32, 34), Q2_K: (256, 84),
          Q4_K: (256, 144), IQ2_XXS: (256, 66), MXFP4: (32, 17)}
NAMES = {F32: 'F32', Q8_0: 'Q8_0', Q2_K: 'Q2_K', Q4_K: 'Q4_K',
         IQ2_XXS: 'IQ2_XXS', MXFP4: 'MXFP4'}
GA = {0, 5, 11, 17, 23, 29, 35, 41, 47}
PATTERN = [int(i not in GA) for i in range(51)]
REVISION = '5711b268169967567844e1e560e8a3966da959b1'


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def pack_string(s):
    b = s.encode() if isinstance(s, str) else s
    return struct.pack('<Q', len(b)) + b


def kv(key, kind, data):
    return pack_string(key) + struct.pack('<I', kind) + data


def u32(key, n):
    return kv(key, 4, struct.pack('<I', n))


def u64(key, n):
    return kv(key, 10, struct.pack('<Q', n))


def f32(key, n):
    return kv(key, 6, struct.pack('<f', n))


def string(key, s):
    return kv(key, 8, pack_string(s))


def array(key, kind, values):
    body = b''.join(pack_string(s) for s in values) if kind == 8 else struct.pack(
        f'<{len(values)}{ {4: "I", 5: "i"}[kind]}', *values)
    return kv(key, 9, struct.pack('<IQ', kind, len(values)) + body)


def tensor_bytes(qtype, shape):
    block, size = LAYOUT[qtype]
    if shape[0] % block:
        raise ValueError(f'block size {block} does not divide {shape}')
    return math.prod(shape) // block * size


@dataclasses.dataclass
class Tensor:
    name: str
    shape: tuple
    qtype: int
    role: str
    source: str
    experts: int = 0
    offset: int = 0
    nbytes: int = 0

    def header(self):
        return pack_string(self.name) + struct.pack('<I', len(self.shape)) + \
            struct.pack(f'<{len(self.shape)}Q', *self.shape) + struct.pack('<IQ', self.qtype, self.offset)


class Source:
    def __init__(self, directory):
        self.directory = Path(directory)
        index = json.loads((self.directory / 'model.safetensors.index.json').read_text())
        self.weight_map = index['weight_map']
        self.tensors = {}
        self.files = {}
        self._fd_lock = threading.Lock()
        # Audio/vision shards need not be downloaded. A source index can also
        # assign text tensors to shards containing multimodal payloads.
        required = {shard for name, shard in self.weight_map.items()
                    if name.startswith(('model.', 'lm_head.'))}
        for shard in sorted(required):
            path = self.directory / shard
            with path.open('rb') as fp:
                length = struct.unpack('<Q', fp.read(8))[0]
                header = json.loads(fp.read(length))
            for name, info in header.items():
                if name == '__metadata__':
                    continue
                if self.weight_map.get(name) != shard:
                    raise ValueError(f'{name}: inconsistent index')
                lo, hi = info['data_offsets']
                self.tensors[name] = (info['dtype'], info['shape'], shard, 8 + length + lo, hi - lo)
        wanted = {name for name in self.weight_map if name.startswith(('model.', 'lm_head.'))}
        if wanted - self.tensors.keys():
            raise ValueError(f'missing text tensors: {sorted(wanted - self.tensors.keys())[:3]}')

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            raise ValueError(f'missing tensor: {name}') from None

    def read(self, name):
        _, _, shard, offset, size = self.info(name)
        fd = self.files.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self.files.get(shard)
                if fd is None:
                    fd = os.open(self.directory / shard, os.O_RDONLY)
                    self.files[shard] = fd
        data = os.pread(fd, size, offset)
        if len(data) != size:
            raise ValueError(f'{name}: short read')
        return data

    def close(self):
        for fd in self.files.values():
            os.close(fd)


def plan_model(source, config, quant, attn_quant):
    expected = dict(model_type='mimo_v2', hidden_size=4096, vocab_size=152576,
                    num_hidden_layers=48, num_nextn_predict_layers=3,
                    num_attention_heads=64, num_key_value_heads=4,
                    swa_num_key_value_heads=8, head_dim=192, v_head_dim=128,
                    n_routed_experts=256, num_experts_per_tok=8,
                    moe_intermediate_size=2048, attention_value_scale=0.707,
                    scoring_func='sigmoid', topk_method='noaux_tc')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'unsupported {key}: {config.get(key)!r}')
    if config['hybrid_layer_pattern'] != PATTERN[:48]:
        raise ValueError('unknown global/SWA layer pattern')
    if config['quantization_config']['weight_block_size'] != [128, 128] or \
            config['quantization_config']['mxfp4_block_size'] != 32:
        raise ValueError('unsupported checkpoint encoding')
    plan = []

    def regular(target, origin, dims, qtype, role):
        dtype, shape, *_ = source.info(origin)
        if shape != list(dims) or dtype not in ('BF16', 'F32', 'F8_E4M3'):
            raise ValueError(f'{origin}: expected {dims}, got {dtype} {shape}')
        if dtype == 'F8_E4M3':
            scale = source.info(origin + '_scale_inv')
            expected_rows = (4 * ((dims[0] // 4 + 127) // 128)
                             if origin.startswith('model.layers.') and origin.endswith('.self_attn.qkv_proj.weight')
                             else (dims[0] + 127) // 128)
            if scale[:2] != ('F32', [expected_rows, (dims[1] + 127) // 128]):
                raise ValueError(f'{origin}: invalid FP8 scale')
        item = Tensor(target, tuple(reversed(dims)), qtype, role, origin)
        item.nbytes = tensor_bytes(qtype, item.shape)
        plan.append(item)

    regular('token_embd.weight', 'model.embed_tokens.weight', (152576, 4096), Q8_0, 'embedding')
    regular('output_norm.weight', 'model.norm.weight', (4096,), F32, 'norm')
    regular('output.weight', 'lm_head.weight', (152576, 4096), Q8_0, 'output')
    for layer in range(51):
        mtp = layer >= 48
        p = f'model.mtp.layers.{layer - 48}' if mtp else f'model.layers.{layer}'
        b = f'blk.{layer}'
        kv_heads = 8 if PATTERN[layer] else 4
        qkv = 64 * 192 + kv_heads * (192 + 128)
        if mtp:
            for target, origin, dims in (
                ('nextn.eh_proj', 'eh_proj', (4096, 8192)),
                ('nextn.enorm', 'enorm', (4096,)),
                ('nextn.hnorm', 'hnorm', (4096,)),
                ('layer_output_norm', 'final_layernorm', (4096,))):
                regular(f'{b}.{target}.weight', f'{p}.{origin}.weight', dims,
                        Q8_0 if target == 'nextn.eh_proj' else F32, 'mtp')
        regular(f'{b}.attn_norm.weight', f'{p}.input_layernorm.weight', (4096,), F32, 'norm')
        regular(f'{b}.attn_qkv.weight', f'{p}.self_attn.qkv_proj.weight', (qkv, 4096),
                Q4_K if attn_quant == 'q4' else Q8_0, 'attention')
        regular(f'{b}.attn_output.weight', f'{p}.self_attn.o_proj.weight', (4096, 8192),
                Q4_K if attn_quant == 'q4' else Q8_0, 'attention')
        if PATTERN[layer]:
            regular(f'{b}.attn_sinks.weight', f'{p}.self_attn.attention_sink_bias',
                    (64,), F32, 'sink')
        regular(f'{b}.ffn_norm.weight', f'{p}.pre_mlp_layernorm.weight' if mtp else
                f'{p}.post_attention_layernorm.weight', (4096,), F32, 'norm')
        if layer == 0 or mtp:
            for part in ('gate', 'up', 'down'):
                dims = (16384, 4096) if part != 'down' else (4096, 16384)
                regular(f'{b}.ffn_{part}.weight', f'{p}.mlp.{part}_proj.weight',
                        dims, Q8_0, 'dense_ffn')
        else:
            regular(f'{b}.ffn_gate_inp.weight', f'{p}.mlp.gate.weight',
                    (256, 4096), F32, 'router')
            regular(f'{b}.exp_probs_b.bias', f'{p}.mlp.gate.e_score_correction_bias',
                    (256,), F32, 'router')
            for part in ('gate', 'up', 'down'):
                output, width = (2048, 4096) if part != 'down' else (4096, 2048)
                pattern = f'{p}.mlp.experts.{{expert}}.{part}_proj.weight'
                for expert in range(256):
                    name = pattern.format(expert=expert)
                    dtype, shape, *_ = source.info(name)
                    if (dtype, shape) != ('U8', [output, width // 2]):
                        raise ValueError(f'{name}: expected packed [{output}, {width // 2}]')
                    scale = source.info(name + '_scale')
                    if scale[:2] != ('U8', [output, width // 32]):
                        raise ValueError(f'{name}: invalid MXFP4 scale')
                qtype = MXFP4 if quant == 'mxfp4' else (Q2_K if part == 'down' else IQ2_XXS)
                item = Tensor(f'{b}.ffn_{part}_exps.weight', (width, output, 256),
                              qtype, 'experts', pattern, experts=256)
                item.nbytes = tensor_bytes(qtype, item.shape)
                plan.append(item)
    offset = 0
    for item in plan:
        item.offset = offset
        offset += align(item.nbytes)
    return plan


def tokenizer_records(directory, config):
    tok = json.loads((Path(directory) / 'tokenizer.json').read_text())
    vocab = tok['model']['vocab']
    tokens = [None] * config['vocab_size']
    for word, token_id in vocab.items():
        tokens[token_id] = word
    special = set()
    for entry in tok['added_tokens']:
        token_id = entry['id']
        tokens[token_id] = entry['content']
        if entry.get('special'):
            special.add(token_id)
    for token_id, word in enumerate(tokens):
        if word is None:
            tokens[token_id] = f'[PAD{token_id}]'
    merges = tok['model']['merges']
    merges = [' '.join(m) if isinstance(m, list) else m for m in merges]
    return [string('tokenizer.ggml.model', 'gpt2'),
            string('tokenizer.ggml.pre', 'qwen2'),
            array('tokenizer.ggml.tokens', 8, tokens),
            array('tokenizer.ggml.token_type', 5, [3 if i in special else 1 for i in range(len(tokens))]),
            array('tokenizer.ggml.merges', 8, merges),
            u32('tokenizer.ggml.eos_token_id', 151645),
            u32('tokenizer.ggml.padding_token_id', 151643),
            kv('tokenizer.ggml.add_bos_token', 7, b'\x00'),
            string('tokenizer.chat_template', (Path(directory) / 'chat_template.jinja').read_text())]


def model_records(directory, revision, config, quant, attn_quant):
    return [string('general.architecture', 'mimo2'),
            string('general.name', 'MiMo V2.6 Flash'),
            string('general.source.revision', revision),
            u32('general.alignment', ALIGNMENT),
            u32('mimo2.block_count', 51),
            u64('mimo2.context_length', config['max_position_embeddings']),
            u32('mimo2.embedding_length', 4096),
            u32('mimo2.feed_forward_length', 16384),
            u32('mimo2.expert_feed_forward_length', 2048),
            u32('mimo2.expert_count', 256),
            u32('mimo2.expert_used_count', 8),
            u32('mimo2.expert_gating_func', 2),
            u32('mimo2.leading_dense_block_count', 1),
            u32('mimo2.nextn_predict_layers', 3),
            u32('mimo2.attention.head_count', 64),
            array('mimo2.attention.head_count_kv', 4, [8 if x else 4 for x in PATTERN]),
            u32('mimo2.attention.key_length', 192),
            u32('mimo2.attention.value_length', 128),
            u32('mimo2.attention.sliding_window', 128),
            array('mimo2.attention.sliding_window_pattern', 4, PATTERN),
            f32('mimo2.attention.layer_norm_rms_epsilon', 1e-6),
            f32('mimo2.attention.value_scale', 0.707),
            u32('mimo2.rope.dimension_count', 64),
            f32('mimo2.rope.freq_base', 1e7),
            f32('mimo2.rope.freq_base_swa', 1e4),
            string('mimo2.quantization', quant + ('-AttnQ4' if attn_quant == 'q4' else '-Q8Attn')),
            *tokenizer_records(directory, config)]


class Quantizer:
    def __init__(self, library):
        self.lib = ctypes.CDLL(library)
        self.lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
        self.lib.ds4q_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                                  ctypes.c_void_p, ctypes.c_int64,
                                                  ctypes.c_int64, ctypes.c_int64,
                                                  ctypes.POINTER(ctypes.c_float)]
        self.lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ds4q_quantize_init(IQ2_XXS)
        self.fp8_lut = np.empty(256, np.float32)
        for code in range(256):
            absolute = code & 127
            if absolute == 127:
                self.fp8_lut[code] = np.nan
                continue
            exponent, mantissa = (code >> 3) & 15, code & 7
            value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
            self.fp8_lut[code] = -value if code & 128 else value

    def regular(self, source, name):
        dtype, shape, *_ = source.info(name)
        raw = source.read(name)
        if dtype == 'F32':
            result = np.frombuffer(raw, '<f4').reshape(shape)
        elif dtype == 'BF16':
            result = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif dtype == 'F8_E4M3':
            codes = np.frombuffer(raw, np.uint8).reshape(shape)
            scales = np.frombuffer(source.read(name + '_scale_inv'), '<f4').reshape(
                source.info(name + '_scale_inv')[1])
            if name.startswith('model.layers.') and name.endswith('.self_attn.qkv_proj.weight'):
                kv = {13568: 4, 14848: 8}.get(shape[0])
                if kv is None or shape[1] != 4096:
                    raise ValueError(f'{name}: unsupported fused QKV shape {shape}')
                q, k, v = 64 * 192, kv * 192, kv * 128
                rpr = shape[0] // 4
                bpr = (rpr + 127) // 128
                if scales.shape != (4 * bpr, 32):
                    raise ValueError(f'{name}: incompatible rank scale layout {scales.shape}')
                result = np.empty(shape, np.float32)
                for rank in range(4):
                    src = codes[rank * rpr:(rank + 1) * rpr]
                    rank_scales = np.repeat(scales[rank * bpr:(rank + 1) * bpr], 128, axis=0)[:rpr]
                    values = self.fp8_lut[src] * np.repeat(rank_scales, 128, axis=1)
                    for dest, start, count in ((rank * (q // 4), 0, q // 4),
                                               (q + rank * (k // 4), q // 4, k // 4),
                                               (q + k + rank * (v // 4), (q + k) // 4, v // 4)):
                        result[dest:dest + count] = values[start:start + count]
            else:
                result = self.fp8_lut[codes] * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)[
                    :shape[0], :shape[1]]
        else:
            raise ValueError(f'{name}: unsupported dtype {dtype}')
        if not np.all(np.isfinite(result)):
            raise ValueError(f'{name}: nonfinite weights')
        return np.ascontiguousarray(result, np.float32)

    def expert(self, source, name):
        shape = source.info(name)[1]
        packed = np.frombuffer(source.read(name), np.uint8).reshape(shape)
        scale = np.frombuffer(source.read(name + '_scale'), np.uint8).reshape(shape[0], shape[1] // 16)
        if np.any(scale == 255):
            raise ValueError(f'{name}: nonfinite MXFP4 scale')
        values = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        out = np.empty((shape[0], shape[1] * 2), np.float32)
        out[:, 0::2], out[:, 1::2] = values[packed & 15], values[packed >> 4]
        out = (out.reshape(shape[0], -1, 32) *
               np.ldexp(np.ones_like(scale, np.float32), scale.astype(np.int32) - 127)[:, :, None])
        return out.reshape(shape[0], -1)

    def encode(self, values, qtype, importance=None):
        if qtype == F32:
            return np.asarray(values, '<f4').tobytes()
        ncols = values.shape[-1]
        matrix = np.ascontiguousarray(values.reshape(-1, ncols), np.float32)
        if importance is None and qtype in (IQ2_XXS, Q4_K):
            importance = np.square(matrix).sum(axis=0, dtype=np.float32)
        if importance is not None:
            importance = np.ascontiguousarray(importance, np.float32)
            if importance.size != ncols:
                raise ValueError('imatrix width differs from matrix width')
        out = np.empty(tensor_bytes(qtype, (ncols, matrix.shape[0])), np.uint8)
        count = self.lib.ds4q_quantize_chunk(qtype,
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.ctypes.data,
            0, matrix.shape[0], ncols,
            importance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if importance is not None else None)
        if count != out.size:
            raise ValueError(f'quantizer wrote {count}, expected {out.size}')
        return out.tobytes()


def read_imatrix(path):
    if not path:
        return {}
    entries = {}
    with open(path, 'rb') as fp:
        def i32():
            return struct.unpack('<i', fp.read(4))[0]
        for _ in range(i32()):
            length = i32()
            if not 0 < length < 4096:
                raise ValueError('invalid imatrix tensor name length')
            name = fp.read(length).decode()
            i32()  # number of calls
            n = i32()
            if n <= 0:
                raise ValueError(f'{name}: invalid imatrix length')
            values = np.frombuffer(fp.read(4 * n), '<f4').copy()
            if values.size != n or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f'{name}: invalid imatrix')
            entries[name] = values
    return entries


def repack_expert(source, name):
    raw = np.frombuffer(source.read(name), np.uint8).reshape(-1, 16)
    scales = np.frombuffer(source.read(name + '_scale'), np.uint8)
    if len(raw) != len(scales) or np.any(scales == 255):
        raise ValueError(f'{name}: invalid MXFP4 blocks')
    low = np.empty((len(raw), 16), np.uint8)
    low[:, 0::2], low[:, 1::2] = raw[:, :8] & 15, raw[:, :8] >> 4
    high = np.empty((len(raw), 16), np.uint8)
    high[:, 0::2], high[:, 1::2] = raw[:, 8:] & 15, raw[:, 8:] >> 4
    packed = np.empty((len(raw), 17), np.uint8)
    packed[:, 0] = scales
    packed[:, 1:] = low | (high << 4)
    return packed.tobytes()


def save_journal(path, signature, completed):
    temporary = path + '.tmp'
    with open(temporary, 'w') as fp:
        json.dump({'signature': signature, 'completed': completed}, fp)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(temporary, path)


def write_gguf(args, source, plan, records):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    quantizer = Quantizer(args.quants_library)
    imatrix = read_imatrix(args.imatrix)
    for item in plan:
        if item.experts and args.imatrix and item.qtype != MXFP4 and item.name not in imatrix:
            raise ValueError(f'{item.name}: missing calibration')
    headers = b''.join(item.header() for item in plan)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(records)) + b''.join(records) + headers
    header += bytes(align(len(header)) - len(header))
    partial, journal = args.out + '.partial', args.out + '.partial.json'
    digest = hashlib.sha256(header + args.source_revision.encode())
    digest.update((source.directory / 'model.safetensors.index.json').read_bytes())
    if args.imatrix:
        with open(args.imatrix, 'rb') as fp:
            for chunk in iter(lambda: fp.read(4 << 20), b''):
                digest.update(chunk)
    with open(args.quants_library, 'rb') as fp:
        for chunk in iter(lambda: fp.read(4 << 20), b''):
            digest.update(chunk)
    with open(__file__, 'rb') as fp:
        for chunk in iter(lambda: fp.read(4 << 20), b''):
            digest.update(chunk)
    signature = digest.hexdigest()
    if os.path.exists(args.out):
        raise ValueError(f'refusing to overwrite {args.out}')
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not args.resume or not (os.path.isfile(partial) and os.path.isfile(journal)):
            raise ValueError('partial output requires --resume')
        state = json.loads(Path(journal).read_text())
        if state['signature'] != signature:
            raise ValueError('resume header/recipe mismatch')
        completed = state['completed']
        if type(completed) is not int or not 0 <= completed <= len(plan):
            raise ValueError('resume journal has an invalid tensor index')
        with open(partial, 'rb') as fp:
            if fp.read(len(header)) != header:
                raise ValueError('resume GGUF header mismatch')
    else:
        with open(partial, 'xb') as fp:
            fp.write(header)
        save_journal(journal, signature, 0)
    remaining = sum(align(item.nbytes) for item in plan[completed:])
    if shutil.disk_usage(Path(args.out).resolve().parent).free < remaining + (32 << 30):
        raise ValueError('insufficient disk space for GGUF plus 32 GiB reserve')
    with open(partial, 'r+b') as fp, concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        for index in range(completed, len(plan)):
            item = plan[index]
            expected = len(header) + item.offset
            if os.fstat(fp.fileno()).st_size < expected:
                raise ValueError('resume output truncated')
            fp.truncate(expected)
            fp.seek(expected)
            started = time.monotonic()
            if item.experts:
                def convert(expert):
                    name = item.source.format(expert=expert)
                    if item.qtype == MXFP4:
                        return repack_expert(source, name)
                    values = quantizer.expert(source, name)
                    importance = imatrix.get(item.name)
                    if importance is not None:
                        importance = importance[expert * item.shape[0]:(expert + 1) * item.shape[0]]
                        if not np.any(importance > 0):
                            importance = None
                    return quantizer.encode(values, item.qtype, importance)
                for start in range(0, item.experts, args.threads):
                    for chunk in pool.map(convert, range(start, min(start + args.threads, item.experts))):
                        if len(chunk) != item.nbytes // item.experts:
                            raise ValueError(f'{item.name}: expert byte count mismatch')
                        fp.write(chunk)
            else:
                importance = imatrix.get(item.name) if item.role == 'attention' else None
                if importance is not None and not np.any(importance > 0):
                    importance = None
                fp.write(quantizer.encode(quantizer.regular(source, item.source),
                                          item.qtype, importance))
            if fp.tell() != expected + item.nbytes:
                raise ValueError(f'{item.name}: payload byte count mismatch')
            fp.write(bytes(align(item.nbytes) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            save_journal(journal, signature, index + 1)
            print(f'[{index + 1}/{len(plan)}] {item.name}: {item.nbytes / 2**30:.3f} GiB, '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    os.rename(partial, args.out)
    os.unlink(journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hf', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--quant', choices=('q2', 'mxfp4'), default='q2')
    parser.add_argument('--attn-quant', choices=('q8', 'q4'), default='q8')
    parser.add_argument('--imatrix')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--quants-library', default=str(Path(__file__).with_name(f'libds4quants.{suffix}')))
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_revision):
        parser.error('source revision must be a 40-character commit hash')
    if not 1 <= args.threads <= 32:
        parser.error('threads must be between 1 and 32')
    if args.attn_quant == 'q4' and not args.out.endswith('-AttnQ4.gguf'):
        if not args.out.endswith('.gguf'):
            parser.error('--attn-quant q4 requires a .gguf output path')
        args.out = args.out[:-5] + '-AttnQ4.gguf'
    config = json.loads((Path(args.hf) / 'config.json').read_text())
    source = Source(args.hf)
    try:
        plan = plan_model(source, config, args.quant, args.attn_quant)
        records = model_records(args.hf, args.source_revision, config, args.quant, args.attn_quant)
        data_start = align(24 + sum(map(len, records)) + sum(len(item.header()) for item in plan))
        by_role = {}
        for item in plan:
            by_role[item.role] = by_role.get(item.role, 0) + item.nbytes
        print(f'output: {args.out}')
        print('layers: 51 (9 global, 42 sliding-window including MTP)')
        print(f'tensors: {len(plan)}, file_bytes: {data_start + sum(align(i.nbytes) for i in plan)}')
        for role, size in sorted(by_role.items()):
            print(f'{role}: {size / 2**30:.3f} GiB')
        if args.dry_run:
            for item in plan:
                print(f'{item.name} {item.shape} {NAMES[item.qtype]}')
        else:
            write_gguf(args, source, plan, records)
    finally:
        source.close()


if __name__ == '__main__':
    main()
