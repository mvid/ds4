#!/usr/bin/env python3
"""Check MiMo GGUF metadata, tensor layout, and sampled source expert blocks."""

import argparse
import io
import json
import os
from pathlib import Path
import random
import struct
import sys

from mimo26_quantize import (ALIGNMENT, MXFP4, REVISION, Quantizer, Source, align,
                              model_records, plan_model, read_imatrix, repack_expert)


def exact(fp, n):
    value = fp.read(n)
    if len(value) != n:
        raise ValueError('truncated GGUF')
    return value


def integer(fp, fmt):
    return struct.unpack('<' + fmt, exact(fp, struct.calcsize(fmt)))[0]


def gguf_string(fp):
    size = integer(fp, 'Q')
    if size > 1 << 28:
        raise ValueError('unreasonable GGUF string length')
    return exact(fp, size).decode('utf-8')


def record(fp):
    start = fp.tell()
    key = gguf_string(fp)
    kind = integer(fp, 'I')
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
             6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind == 8:
        gguf_string(fp)
    elif kind == 9:
        subtype, count = integer(fp, 'I'), integer(fp, 'Q')
        if count > 1 << 28:
            raise ValueError(f'{key}: unreasonable array length')
        if subtype == 8:
            for _ in range(count):
                gguf_string(fp)
        elif subtype in sizes:
            exact(fp, count * sizes[subtype])
        else:
            raise ValueError(f'{key}: unsupported array type {subtype}')
    elif kind in sizes:
        exact(fp, sizes[kind])
    else:
        raise ValueError(f'{key}: unsupported metadata type {kind}')
    end = fp.tell()
    fp.seek(start)
    raw = exact(fp, end - start)
    return key, raw


def check_expert(fp, start, item, source, quantizer, imatrix):
    stride = item.nbytes // item.experts
    rng = random.Random(26 + item.offset)
    chosen = {0, item.experts - 1, rng.randrange(item.experts)}
    for expert in sorted(chosen):
        name = item.source.format(expert=expert)
        if item.qtype == MXFP4:
            expected = repack_expert(source, name)
        else:
            weights = quantizer.expert(source, name)
            importance = imatrix.get(item.name)
            if importance is not None:
                lo = expert * item.shape[0]
                importance = importance[lo:lo + item.shape[0]]
                if not any(importance > 0):
                    importance = None
            expected = quantizer.encode(weights, item.qtype, importance)
        if len(expected) != stride:
            raise ValueError(f'{item.name}: expert size mismatch')
        for offset in {0, (stride // 2 // 32) * 32, stride - 32}:
            fp.seek(start + expert * stride + offset)
            if exact(fp, 32) != expected[offset:offset + 32]:
                raise ValueError(f'{item.name}: expert {expert} block at {offset} differs')


def validate(args):
    source = Source(args.hf)
    try:
        config = json.loads((Path(args.hf) / 'config.json').read_text())
        plan = plan_model(source, config, args.quant, args.attn_quant)
        expected = {}
        for data in model_records(args.hf, args.source_revision, config, args.quant, args.attn_quant):
            key, encoded = record(io.BytesIO(data))
            if key in expected:
                raise ValueError(f'duplicate planned key {key}')
            expected[key] = encoded
        with open(args.gguf, 'rb') as fp:
            if exact(fp, 4) != b'GGUF' or integer(fp, 'I') != 3:
                raise ValueError('expected GGUF v3')
            if integer(fp, 'Q') != len(plan):
                raise ValueError('tensor count differs from source plan')
            metadata = {}
            for _ in range(integer(fp, 'Q')):
                key, encoded = record(fp)
                if key in metadata:
                    raise ValueError(f'duplicate GGUF key {key}')
                metadata[key] = encoded
            if metadata.keys() != expected.keys():
                raise ValueError(f'metadata keys differ: missing {expected.keys() - metadata.keys()}, '
                                 f'extra {metadata.keys() - expected.keys()}')
            for key, encoded in expected.items():
                if metadata[key] != encoded:
                    raise ValueError(f'{key}: metadata value differs from source')
            for item in plan:
                name = gguf_string(fp)
                rank = integer(fp, 'I')
                if rank > 4:
                    raise ValueError(f'{name}: invalid rank')
                shape = tuple(integer(fp, 'Q') for _ in range(rank))
                qtype, offset = integer(fp, 'I'), integer(fp, 'Q')
                if (name, shape, qtype, offset) != (item.name, item.shape, item.qtype, item.offset):
                    raise ValueError(f'{item.name}: tensor layout differs from source recipe')
            start = align(fp.tell())
            size = start + plan[-1].offset + align(plan[-1].nbytes)
            if os.fstat(fp.fileno()).st_size != size:
                raise ValueError(f'file size differs: expected {size}')
            totals = {}
            for item in plan:
                totals[item.role] = totals.get(item.role, 0) + item.nbytes
            for role, nbytes in sorted(totals.items()):
                print(f'{role}: {nbytes / 2**30:.3f} GiB')
            print(f'PASS: {len(plan)} tensor layouts, all metadata and {size} file bytes')
            q = Quantizer(args.quants_library)
            imatrix = read_imatrix(args.imatrix)
            selected_layers = {1, 24, 47}
            checked = 0
            for item in plan:
                if not item.experts or int(item.name.split('.')[1]) not in selected_layers:
                    continue
                if args.imatrix and item.qtype != MXFP4 and item.name not in imatrix:
                    raise ValueError(f'{item.name}: missing imatrix')
                check_expert(fp, start + item.offset, item, source, q, imatrix)
                checked += 3
            print(f'PASS: {checked} sampled experts against source blocks')
    finally:
        source.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gguf')
    parser.add_argument('--hf', required=True)
    parser.add_argument('--source-revision', default=REVISION)
    parser.add_argument('--quant', choices=('q2', 'mxfp4'), required=True)
    parser.add_argument('--attn-quant', choices=('q8', 'q4'), default='q8')
    parser.add_argument('--imatrix')
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--quants-library', default=str(Path(__file__).with_name(f'libds4quants.{suffix}')))
    arguments = parser.parse_args()
    try:
        validate(arguments)
    except (OSError, ValueError) as error:
        sys.exit(f'mimo26-validate: {error}')
