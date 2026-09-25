#!/usr/bin/env python3
"""Convert the pinned MiMo V2.6 Flash DFlash draft sidecar to a dflash GGUF.

Input is the `dflash/` directory of XiaomiMiMo/MiMo-V2.6-Flash-RL downloaded
at the pinned revision: `dflash_draft_model.safetensors` (63 BF16 tensors),
`mask_embedding.pt` (learned BF16 embedding for mask token 151675, which
differs from the target model's untrained row), `config.json`, and the index.
The output holds only draft weights; the runtime supplies the target model's
token embedding (for real tokens) and output head.

Matrices are written Q8_0 (BF16 -> F32 -> Q8_0 through libds4quants), norms,
sinks, and the mask embedding F32 (exact BF16 -> F32 widening).
"""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import struct
import sys
import zipfile

from mimo26_quantize import (ALIGNMENT, F32, NAMES, Q8_0, REVISION, Quantizer, Tensor, align,
                             array, kv, string, tensor_bytes, u32, u64, f32)

REPOSITORY = 'XiaomiMiMo/MiMo-V2.6-Flash-RL'
SIDECAR = 'dflash'
WEIGHTS = 'dflash_draft_model.safetensors'
MASK_FILE = 'mask_embedding.pt'
MASK_SOURCE = MASK_FILE + ':embedding'
PROVENANCE = ('config.json', 'model.safetensors.index.json', WEIGHTS, MASK_FILE)
N_LAYER, EMBD, FFN, HEADS, KV_HEADS, HEAD_DIM = 5, 4096, 16384, 64, 8, 128
TARGET_LAYERS = [0, 11, 23, 35, 47]
MASK_TOKEN = 151675
BLOCK_SIZE = 8
WINDOW = 1024
VALUE_SCALE = 0.612
ROPE_DIM = 64
ROPE_BASE = 10000.0
CONTEXT = 1048576
VOCAB = 152576


def boolean(key, value):
    return kv(key, 7, struct.pack('<B', 1 if value else 0))


def sidecar_dir(hf):
    return Path(hf) / SIDECAR


def verify_provenance(hf, revision):
    """Check every consumed sidecar file against the HF download record."""
    records = Path(hf) / '.cache' / 'huggingface' / 'download' / SIDECAR
    for name in PROVENANCE:
        meta = records / (name + '.metadata')
        if not meta.is_file():
            raise ValueError(f'{meta}: missing HF download record; download {SIDECAR}/{name} '
                             f'with huggingface_hub at revision {revision}')
        lines = meta.read_text().split('\n')
        if len(lines) < 2 or lines[0] != revision:
            raise ValueError(f'{SIDECAR}/{name}: downloaded at revision {lines[0]!r}, expected {revision}')
        etag = lines[1]
        path = sidecar_dir(hf) / name
        if len(etag) == 64:
            digest = hashlib.sha256()
        elif len(etag) == 40:
            digest = hashlib.sha1(b'blob %d\0' % path.stat().st_size)
        else:
            raise ValueError(f'{SIDECAR}/{name}: unrecognized etag {etag!r}')
        with path.open('rb') as fp:
            for chunk in iter(lambda: fp.read(16 << 20), b''):
                digest.update(chunk)
        if digest.hexdigest() != etag:
            raise ValueError(f'{SIDECAR}/{name}: content hash differs from revision {revision}')


def check_config(config):
    expected = dict(architectures=['DFlashDraftModel'], hidden_size=EMBD, intermediate_size=FFN,
                    num_hidden_layers=N_LAYER, num_attention_heads=HEADS,
                    num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, v_head_dim=HEAD_DIM,
                    partial_rotary_factor=0.5, block_size=BLOCK_SIZE,
                    layer_types=['sliding_attention'] * N_LAYER, sliding_window=WINDOW,
                    use_sliding_window=True, is_causal=False, num_target_layers=48,
                    target_hidden_size=EMBD, vocab_size=VOCAB, max_position_embeddings=CONTEXT,
                    rope_theta=ROPE_BASE, rms_norm_eps=1e-6, hidden_act='silu',
                    attention_bias=False, add_swa_attention_sink_bias=True,
                    tie_word_embeddings=False)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'unsupported DFlash config {key}: {config.get(key)!r}')
    if 'rope_scaling' in config and config['rope_scaling'] is not None:
        raise ValueError('unsupported DFlash rope_scaling')
    dflash = dict(target_layer_ids=TARGET_LAYERS, mask_token_id=MASK_TOKEN, num_anchors=4096,
                  block_size=BLOCK_SIZE, attention_value_scale=VALUE_SCALE, attention_sink_bias=True)
    for key, value in dflash.items():
        if config['dflash_config'].get(key) != value:
            raise ValueError(f'unsupported dflash_config {key}: {config["dflash_config"].get(key)!r}')
    if int(HEAD_DIM * config['partial_rotary_factor']) != ROPE_DIM:
        raise ValueError('rotary dimension mismatch')


class _TorchPickle(pickle.Unpickler):
    """Decode a plain torch.save dict of one tensor without importing torch."""

    def find_class(self, module, name):
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return lambda storage, offset, size, stride, *_: (storage, offset, tuple(size), tuple(stride))
        if (module, name) == ('torch', 'BFloat16Storage'):
            return 'BF16'
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f'unexpected global {module}.{name}')

    def persistent_load(self, pid):
        return tuple(pid)


def read_mask_embedding(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        roots = {name.split('/')[0] for name in names}
        if len(roots) != 1:
            raise ValueError(f'{path}: unexpected archive layout')
        root = roots.pop()
        if archive.read(f'{root}/byteorder') != b'little':
            raise ValueError(f'{path}: expected little-endian storage')
        payload = _TorchPickle(archive.open(f'{root}/data.pkl')).load()
        if not isinstance(payload, dict) or set(payload) != {'mask_token_id', 'embedding'}:
            raise ValueError(f'{path}: unexpected payload keys')
        if payload['mask_token_id'] != MASK_TOKEN:
            raise ValueError(f'{path}: mask token {payload["mask_token_id"]}, expected {MASK_TOKEN}')
        storage, offset, size, stride = payload['embedding']
        if storage[:2] != ('storage', 'BF16') or storage[4] != EMBD or \
                (offset, size, stride) != (0, (EMBD,), (1,)):
            raise ValueError(f'{path}: embedding must be contiguous BF16 [{EMBD}]')
        data = archive.read(f'{root}/data/{storage[2]}')
    if len(data) != 2 * EMBD:
        raise ValueError(f'{path}: embedding storage has {len(data)} bytes')
    return data


class DFlashSource:
    """safetensors + mask_embedding.pt reader with the Source interface Quantizer uses."""

    def __init__(self, hf):
        directory = sidecar_dir(hf)
        index = json.loads((directory / 'model.safetensors.index.json').read_text())
        if set(index['weight_map'].values()) != {WEIGHTS}:
            raise ValueError('DFlash index must reference only ' + WEIGHTS)
        path = directory / WEIGHTS
        self.fd = os.open(path, os.O_RDONLY)
        with path.open('rb') as fp:
            length = struct.unpack('<Q', fp.read(8))[0]
            header = json.loads(fp.read(length))
        header.pop('__metadata__', None)
        if header.keys() != index['weight_map'].keys():
            raise ValueError('safetensors header and index disagree')
        self.tensors = {}
        end = 0
        for name, info in sorted(header.items(), key=lambda item: item[1]['data_offsets']):
            lo, hi = info['data_offsets']
            if lo != end:
                raise ValueError(f'{name}: non-contiguous safetensors payload')
            end = hi
            self.tensors[name] = (info['dtype'], info['shape'], WEIGHTS, 8 + length + lo, hi - lo)
        if 8 + length + end != path.stat().st_size:
            raise ValueError(f'{WEIGHTS}: size differs from header')
        if end != index['metadata']['total_size']:
            raise ValueError(f'{WEIGHTS}: total_size differs from index')
        self.mask = read_mask_embedding(directory / MASK_FILE)
        self.tensors[MASK_SOURCE] = ('BF16', [EMBD], MASK_FILE, 0, len(self.mask))

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            raise ValueError(f'missing tensor: {name}') from None

    def read(self, name):
        if name == MASK_SOURCE:
            return self.mask
        _, _, _, offset, size = self.info(name)
        data = os.pread(self.fd, size, offset)
        if len(data) != size:
            raise ValueError(f'{name}: short read')
        return data

    def close(self):
        os.close(self.fd)


def plan_dflash(source):
    plan = []

    def add(target, origin, dims, qtype, role):
        dtype, shape, *_ = source.info(origin)
        if dtype != 'BF16' or shape != list(dims):
            raise ValueError(f'{origin}: expected BF16 {list(dims)}, got {dtype} {shape}')
        item = Tensor(target, tuple(reversed(dims)), qtype, role, origin)
        item.nbytes = tensor_bytes(qtype, item.shape)
        plan.append(item)

    add('dflash.fc.weight', 'fc.weight', (EMBD, EMBD * len(TARGET_LAYERS)), Q8_0, 'projection')
    add('dflash.hidden_norm.weight', 'hidden_norm.weight', (EMBD,), F32, 'norm')
    for il in range(N_LAYER):
        p, b = f'layers.{il}', f'dflash.blk.{il}'
        add(f'{b}.attn_norm.weight', f'{p}.input_layernorm.weight', (EMBD,), F32, 'norm')
        add(f'{b}.attn_q.weight', f'{p}.self_attn.q_proj.weight', (HEADS * HEAD_DIM, EMBD), Q8_0, 'attention')
        add(f'{b}.attn_k.weight', f'{p}.self_attn.k_proj.weight', (KV_HEADS * HEAD_DIM, EMBD), Q8_0, 'attention')
        add(f'{b}.attn_v.weight', f'{p}.self_attn.v_proj.weight', (KV_HEADS * HEAD_DIM, EMBD), Q8_0, 'attention')
        add(f'{b}.attn_output.weight', f'{p}.self_attn.o_proj.weight', (EMBD, HEADS * HEAD_DIM), Q8_0, 'attention')
        add(f'{b}.attn_q_norm.weight', f'{p}.self_attn.q_norm.weight', (HEAD_DIM,), F32, 'norm')
        add(f'{b}.attn_k_norm.weight', f'{p}.self_attn.k_norm.weight', (HEAD_DIM,), F32, 'norm')
        add(f'{b}.attn_sinks.weight', f'{p}.self_attn.attention_sink_bias', (HEADS,), F32, 'sink')
        add(f'{b}.ffn_norm.weight', f'{p}.post_attention_layernorm.weight', (EMBD,), F32, 'norm')
        add(f'{b}.ffn_gate.weight', f'{p}.mlp.gate_proj.weight', (FFN, EMBD), Q8_0, 'ffn')
        add(f'{b}.ffn_up.weight', f'{p}.mlp.up_proj.weight', (FFN, EMBD), Q8_0, 'ffn')
        add(f'{b}.ffn_down.weight', f'{p}.mlp.down_proj.weight', (EMBD, FFN), Q8_0, 'ffn')
    add('dflash.output_norm.weight', 'norm.weight', (EMBD,), F32, 'norm')
    add('dflash.mask_embd.weight', MASK_SOURCE, (EMBD,), F32, 'mask')
    used = {item.source for item in plan}
    if used != source.tensors.keys():
        raise ValueError(f'unmapped source tensors: {sorted(source.tensors.keys() - used)}')
    offset = 0
    for item in plan:
        item.offset = offset
        offset += align(item.nbytes)
    return plan


def dflash_records(revision):
    return [string('general.architecture', 'dflash'),
            string('general.name', 'MiMo V2.6 Flash DFlash'),
            string('general.source.huggingface.repository', REPOSITORY),
            string('general.source.revision', revision),
            u32('general.alignment', ALIGNMENT),
            u32('general.file_type', 7),
            string('dflash.decoder_arch', 'mimo2'),
            u32('dflash.block_count', N_LAYER),
            u32('dflash.block_size', BLOCK_SIZE),
            u64('dflash.context_length', CONTEXT),
            u32('dflash.embedding_length', EMBD),
            u32('dflash.feed_forward_length', FFN),
            u32('dflash.vocab_size', VOCAB),
            array('dflash.target_layers', 4, TARGET_LAYERS),
            u32('dflash.target_layer_count', 48),
            u32('dflash.mask_token_id', MASK_TOKEN),
            u32('dflash.anchor_count', 4096),
            u32('dflash.attention.head_count', HEADS),
            u32('dflash.attention.head_count_kv', KV_HEADS),
            u32('dflash.attention.key_length', HEAD_DIM),
            u32('dflash.attention.value_length', HEAD_DIM),
            u32('dflash.attention.sliding_window', WINDOW),
            array('dflash.attention.sliding_window_pattern', 4, [1] * N_LAYER),
            boolean('dflash.attention.causal', False),
            boolean('dflash.attention.sinks', True),
            f32('dflash.attention.value_scale', VALUE_SCALE),
            f32('dflash.attention.layer_norm_rms_epsilon', 1e-6),
            u32('dflash.rope.dimension_count', ROPE_DIM),
            f32('dflash.rope.freq_base', ROPE_BASE),
            string('dflash.rope.scaling.type', 'none')]


def gguf_header(plan, records):
    header = b'GGUF' + struct.pack('<IQQ', 3, len(plan), len(records)) + b''.join(records) + \
        b''.join(item.header() for item in plan)
    return header + bytes(align(len(header)) - len(header))


def load(hf, revision, check_hashes=True):
    """Validated source, plan, and metadata records shared by convert and validate."""
    config = json.loads((sidecar_dir(hf) / 'config.json').read_text())
    check_config(config)
    if check_hashes:
        verify_provenance(hf, revision)
    source = DFlashSource(hf)
    try:
        return source, plan_dflash(source), dflash_records(revision)
    except BaseException:
        source.close()
        raise


def write_gguf(args, source, plan, records):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f'refusing to overwrite {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    header = gguf_header(plan, records)
    total = len(header) + plan[-1].offset + align(plan[-1].nbytes)
    if shutil.disk_usage(out.resolve().parent).free < total + (4 << 30):
        raise ValueError('insufficient disk space for GGUF plus 4 GiB reserve')
    quantizer = Quantizer(args.quants_library)
    partial = Path(str(out) + '.partial')
    partial.unlink(missing_ok=True)

    def encode(item):
        return quantizer.encode(quantizer.regular(source, item.source), item.qtype)

    try:
        with partial.open('xb') as fp, concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
            fp.write(header)
            for index, (item, data) in enumerate(zip(plan, pool.map(encode, plan))):
                if fp.tell() != len(header) + item.offset or len(data) != item.nbytes:
                    raise ValueError(f'{item.name}: payload layout mismatch')
                fp.write(data)
                fp.write(bytes(align(item.nbytes) - item.nbytes))
                print(f'[{index + 1}/{len(plan)}] {item.name} {item.shape} {NAMES[item.qtype]}', flush=True)
            if fp.tell() != total:
                raise ValueError('output size mismatch')
            fp.flush()
            os.fsync(fp.fileno())
        os.rename(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f'wrote {out}: {total} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hf', required=True, help='MiMo-V2.6-Flash-RL download root (contains dflash/)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-revision', default=REVISION)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    parser.add_argument('--quants-library', default=str(Path(__file__).with_name(f'libds4quants.{suffix}')))
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_revision):
        parser.error('source revision must be a 40-character commit hash')
    if not 1 <= args.threads <= 32:
        parser.error('threads must be between 1 and 32')
    source, plan, records = load(args.hf, args.source_revision)
    try:
        header = gguf_header(plan, records)
        by_role = collections.Counter()
        for item in plan:
            by_role[item.role] += item.nbytes
        print(f'output: {args.out}')
        print(f'tensors: {len(plan)}, file_bytes: {len(header) + plan[-1].offset + align(plan[-1].nbytes)}')
        for role, size in sorted(by_role.items()):
            print(f'{role}: {size / 2**20:.1f} MiB')
        if args.dry_run:
            for item in plan:
                print(f'{item.name} {item.shape} {NAMES[item.qtype]} <- {item.source}')
        else:
            write_gguf(args, source, plan, records)
    finally:
        source.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, pickle.UnpicklingError) as error:
        sys.exit(f'mimo26-dflash-convert: {error}')
