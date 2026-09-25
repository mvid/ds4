#!/usr/bin/env python3
"""Check a MiMo V2.6 Flash DFlash GGUF against the pinned source sidecar.

Checks: agreed metadata schema values, exact metadata bytes versus the
converter recipe, the 64-tensor inventory (names, ggml shapes, types,
offsets), file size, every F32 tensor bit-exact against BF16 source widening,
every Q8_0 tensor fully dequantized within the Q8_0 rounding bound of the
source, and sampled Q8_0 rows byte-identical to a fresh libds4quants encode.
"""

import argparse
import io
import os
from pathlib import Path
import pickle
import random
import struct
import sys

import numpy as np

from mimo26_dflash_convert import load
from mimo26_quantize import F32, Q8_0, REVISION, Quantizer, align
from mimo26_validate_gguf import exact, gguf_string, integer, record

# Agreed runtime schema, written out literally so a converter constant change is caught.
SCHEMA = {
    'general.architecture': 'dflash',
    'dflash.decoder_arch': 'mimo2',
    'dflash.block_count': 5,
    'dflash.block_size': 8,
    'dflash.context_length': 1048576,
    'dflash.embedding_length': 4096,
    'dflash.feed_forward_length': 16384,
    'dflash.target_layers': [0, 11, 23, 35, 47],
    'dflash.mask_token_id': 151675,
    'dflash.attention.head_count': 64,
    'dflash.attention.head_count_kv': 8,
    'dflash.attention.key_length': 128,
    'dflash.attention.value_length': 128,
    'dflash.attention.sliding_window': 1024,
    'dflash.attention.value_scale': np.float32(0.612),
    'dflash.attention.sinks': True,
    'dflash.attention.causal': False,
    'dflash.attention.layer_norm_rms_epsilon': np.float32(1e-6),
    'dflash.rope.dimension_count': 64,
    'dflash.rope.freq_base': np.float32(10000.0),
}
LAYER_TENSORS = {  # name -> (ggml shape, type)
    'attn_norm': ((4096,), F32), 'attn_q': ((4096, 8192), Q8_0), 'attn_k': ((4096, 1024), Q8_0),
    'attn_v': ((4096, 1024), Q8_0), 'attn_output': ((8192, 4096), Q8_0),
    'attn_q_norm': ((128,), F32), 'attn_k_norm': ((128,), F32), 'attn_sinks': ((64,), F32),
    'ffn_norm': ((4096,), F32), 'ffn_gate': ((4096, 16384), Q8_0), 'ffn_up': ((4096, 16384), Q8_0),
    'ffn_down': ((16384, 4096), Q8_0)}
INVENTORY = {'dflash.fc.weight': ((20480, 4096), Q8_0), 'dflash.hidden_norm.weight': ((4096,), F32),
             'dflash.output_norm.weight': ((4096,), F32), 'dflash.mask_embd.weight': ((4096,), F32),
             **{f'dflash.blk.{il}.{name}.weight': spec
                for il in range(5) for name, spec in LAYER_TENSORS.items()}}


def decode(encoded):
    fp = io.BytesIO(encoded)
    gguf_string(fp)
    kind = integer(fp, 'I')
    scalar = {4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q'}
    if kind == 8:
        return gguf_string(fp)
    if kind == 9:
        subtype, count = integer(fp, 'I'), integer(fp, 'Q')
        return [integer(fp, scalar[subtype]) for _ in range(count)]
    value = integer(fp, scalar[kind])
    return np.float32(value) if kind == 6 else value


def source_f32(source, name):
    raw = np.frombuffer(source.read(name), '<u2')
    return (raw.astype(np.uint32) << 16).view(np.float32)


def check_q8(data, reference, ncols, name):
    blocks = np.frombuffer(data, np.uint8).reshape(-1, 34)
    scale = blocks[:, :2].copy().view('<f2').astype(np.float32).reshape(-1)
    quants = blocks[:, 2:].view(np.int8).astype(np.float32)
    values = (quants * scale[:, None]).reshape(-1, ncols)
    ref = reference.reshape(-1, 32)
    d = np.abs(ref).max(axis=1) / 127
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name}: nonfinite dequantized values')
    # ggml Q8_0: d = amax / 127 stored as f16 (subnormal below 6.1e-5), q = round(x / d).
    if not np.array_equal(scale, d.astype(np.float16).astype(np.float32)):
        raise ValueError(f'{name}: Q8_0 block scale differs from f16(amax/127)')
    error = np.abs(values.reshape(-1, 32) - ref).max(axis=1)
    bound = 0.5 * d * (1 + 2 ** -20) + 127 * np.abs(scale - d)
    if np.any(error > bound):
        worst = int(np.argmax(error - bound))
        raise ValueError(f'{name}: block {worst} error {error[worst]} exceeds Q8_0 bound {bound[worst]}')
    diff = values.reshape(-1) - reference.reshape(-1)
    subnormal = float(np.mean(d < 2 ** -14))
    return float(np.sqrt(np.mean(diff * diff)) / np.sqrt(np.mean(reference * reference))), subnormal


def validate(args):
    source, plan, records = load(args.hf, args.source_revision, check_hashes=not args.skip_hashes)
    try:
        expected = dict(record(io.BytesIO(data)) for data in records)
        for key, value in SCHEMA.items():
            if key not in expected or decode(expected[key]) != value:
                raise ValueError(f'recipe {key} differs from agreed schema {value!r}')
        if {item.name: (item.shape, item.qtype) for item in plan} != INVENTORY or len(plan) != 64:
            raise ValueError('recipe inventory differs from agreed schema')
        with open(args.gguf, 'rb') as fp:
            if exact(fp, 4) != b'GGUF' or integer(fp, 'I') != 3:
                raise ValueError('expected GGUF v3')
            if integer(fp, 'Q') != len(plan):
                raise ValueError('tensor count differs')
            metadata = {}
            for _ in range(integer(fp, 'Q')):
                key, encoded = record(fp)
                if key in metadata:
                    raise ValueError(f'duplicate GGUF key {key}')
                metadata[key] = encoded
            if metadata != expected:
                differing = sorted(k for k in metadata.keys() | expected.keys()
                                   if metadata.get(k) != expected.get(k))
                raise ValueError(f'metadata differs: {differing}')
            for item in plan:
                name = gguf_string(fp)
                rank = integer(fp, 'I')
                if rank > 4:
                    raise ValueError(f'{name}: invalid rank')
                shape = tuple(integer(fp, 'Q') for _ in range(rank))
                qtype, offset = integer(fp, 'I'), integer(fp, 'Q')
                if (name, shape, qtype, offset) != (item.name, item.shape, item.qtype, item.offset):
                    raise ValueError(f'{item.name}: tensor header differs from recipe')
            start = align(fp.tell())
            size = start + plan[-1].offset + align(plan[-1].nbytes)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f'file size differs: expected {size}')
            print(f'PASS: {len(metadata)} metadata keys match agreed schema and source recipe')
            print(f'PASS: 64 tensor headers, file size {size} bytes')
            quantizer = Quantizer(args.quants_library)
            rng = random.Random(0xDF1A5)
            exact_f32 = sampled = 0
            worst = (0.0, '')
            for item in plan:
                fp.seek(start + item.offset)
                data = exact(fp, item.nbytes)
                if item.qtype == F32:
                    if data != source_f32(source, item.source).astype('<f4').tobytes():
                        raise ValueError(f'{item.name}: F32 payload differs from BF16 source')
                    exact_f32 += 1
                    continue
                ncols = item.shape[0]
                reference = source_f32(source, item.source).reshape(-1, ncols)
                rel, subnormal = check_q8(data, reference, ncols, item.name)
                worst = max(worst, (rel, item.name))
                if args.verbose:
                    print(f'  {item.name}: relative RMS error {rel:.5f}, '
                          f'f16-subnormal block scales {subnormal:.1%}')
                row_bytes = ncols // 32 * 34
                for row in sorted({0, reference.shape[0] - 1, *rng.sample(range(reference.shape[0]), 6)}):
                    fresh = quantizer.encode(np.ascontiguousarray(reference[row:row + 1]), Q8_0)
                    if fresh != data[row * row_bytes:(row + 1) * row_bytes]:
                        raise ValueError(f'{item.name}: row {row} differs from fresh Q8_0 encode')
                    sampled += 1
            print(f'PASS: {exact_f32} F32 tensors bit-exact to BF16 source (including mask embedding)')
            print(f'PASS: {len(plan) - exact_f32} Q8_0 tensors fully within Q8_0 bound; '
                  f'worst relative RMS error {worst[0]:.5f} ({worst[1]})')
            print(f'PASS: {sampled} sampled Q8_0 rows byte-identical to fresh encode')
    finally:
        source.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('gguf')
    parser.add_argument('--hf', required=True, help='MiMo-V2.6-Flash-RL download root (contains dflash/)')
    parser.add_argument('--source-revision', default=REVISION)
    parser.add_argument('--skip-hashes', action='store_true', help='skip HF revision content hashes')
    parser.add_argument('--verbose', action='store_true', help='print per-tensor Q8_0 error')
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--quants-library', default=str(Path(__file__).with_name(f'libds4quants.{suffix}')))
    arguments = parser.parse_args()
    try:
        validate(arguments)
    except (OSError, ValueError, pickle.UnpicklingError) as error:
        sys.exit(f'mimo26-dflash-validate: {error}')
