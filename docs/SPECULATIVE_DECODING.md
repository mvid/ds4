# Speculative Decoding

[README](../README.md)

Speculation proposes future tokens with a smaller draft block, then checks
them with the main model. An accepted prefix advances generation by several
tokens in one verification pass. It does not accelerate prefill.

It is opt-in. Gains depend on the prompt, model, backend, and context length;
poor acceptance can make it slower. Measure your workload rather than assuming
that a draft model always helps.

## DeepSeek Flash: DSpark

DSpark is a separate support GGUF, not a standalone language model. It proposes
up to five future tokens. Match its checkpoint to the main model:

| Main checkpoint | Download | Support file |
| --- | --- | --- |
| Flash 0731 | `ds4f-dspark` | `gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` |
| Flash Vision Experimental | `ds4f-vision-dspark` | `gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf` |

For the 0731 Q2 model:

```sh
./download_model.sh ds4f-q2
./download_model.sh ds4f-dspark
./ds4 --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

For Vision Experimental, substitute its matching main model and support file.
Do not mix the two checkpoints. DSpark is not supported for PRO.
The same flags work in `ds4-agent` and non-batched `ds4-server` requests.

The support file adds about 5.6 GiB of weights plus runtime state. On Metal,
the main model can be resident or SSD-streamed. DSpark replaces the legacy
one-stage MTP drafter for that run; the two are not stacked.

Resident M5 paths batch supported verifier expert rows, including two-Mac TP.
On DGX Spark, resident Q2 also batches the seed with longer drafts and uses
small-batch Q8 and expert kernels. No extra flags are needed.
The scheduler can back off when drafting is unproductive. Defaults select the
fast paths; diagnostic environment variables are not needed for normal use.
Recorded comparisons are in [the QA guide](../QA_BEFORE_RELEASES.md).

For the tested Strix Halo coding configuration, use `--dspark --dspark-confidence 0.7` with the default five-token draft cap and scheduler. Client sampling is temperature `1.0`, `top_p=0.95`, `min_p=0`, and `top_k=0`; high reasoning was also checked on coding and tool-use requests. This uses opportunistic sampling as described below; exact-mode throughput is not qualified by these measurements. `--mtp-draft` controls legacy autoregressive MTP, not the DSpark draft width.

## GLM: built-in MTP

GLM's draft block is already in its main GGUF:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp
```

`--mtp-timing` also enables it and prints acceptance and timing counters.
The current GLM cycle commits up to two tokens. No external support file is
needed, and ordinary decode remains the default.

## Qwen3.8: built-in MTP

Both Qwen downloads include MTP and native BF16 n-grams in the main GGUF:

```sh
./download_model.sh qwen38-q4k
./ds4 --mtp
```

Ordinary decode uses the same file with `--mtp` omitted. For non-zero
temperature, add `--mtp-exact-sampling` to preserve the target sampling
distribution. See [Qwen setup](QWEN38_FLASH_NEXT.md) for Metal and CUDA.

The cycle drafts one token ahead by default and engages a **second, chained
draft** (one extra nextn-layer step conditioned on the predictor's own
stream, verified in a 3-row pass) while recent first-draft acceptance is
perfect, disengaging after repeated second-draft rejections.
`DS4_QWEN4_MTP_DEPTH=2` or `=3` fixes the depth;
`0` (default) is the adaptive policy.

## MiMo V2.6 Flash: built-in MTP

The three nextn layers (`blk.48..50`) ship in both MiMo GGUFs:

```sh
./ds4 -m gguf/MiMo-V2.6-Flash-IQ2_XXS-Q2_K-Q8Attn.gguf --mtp
```

Each cycle drafts up to `DS4_MIMO2_MTP_DEPTH` tokens (1 to 3, default 2).
Every depth is conditioned on the same trunk hidden state and on the previous
depth's draft token, then the target verifies the evaluated token and all
drafts in one batch. Rejected rows get their 128-row sliding-window cache
slots back from a pre-verify snapshot. The MTP layers keep their own short
cache history, restarted after any rejection. Metal only.

At non-zero temperature MiMo always uses exact sampling: a draft is accepted
with its target probability, and a rejection samples the replacement from the
remaining distribution. `--mtp-exact-sampling` is not needed.
`--mtp-timing` prints verify cycles, draft acceptance, and committed tokens per
verify when each session ends. `./ds4_test --mimo2-mtp-verify` with
`DS4_TEST_MODEL` set to a MiMo GGUF and `DS4_TEST_GLM_MTP=1` compares greedy
MTP output with plain greedy decoding. The batched verifier rounds differently
from one-token decode, so a near tie can continue differently; the test then
replays the speculative tokens through plain decode and requires each to be
within 2.0 logits of the plain argmax, the same bound the GLM and DSpark
oracles use. It checks that bound, not strict token identity. (On the
first-imatrix Q2 file the first divergence came at token 22 with a worst gap of
0.15; the MXFP4 file matched byte for byte on all three prompts.)

First-imatrix Q2 on an M5 Max 128 GB decoded the Olympiad prompt at
39.63 t/s without speculation and 27.68 t/s with MTP (100 of 311 drafts
accepted). MTP is correct under the near-argmax oracle but slower on this
workload. Benchmark it before enabling it for throughput.

## MiMo V2.6 Flash: DFlash

Xiaomi ships a DFlash block drafter next to the checkpoint
(`dflash/`: 5 layers, block 8, 1.47B parameters). Convert it once with
`gguf-tools/mimo26_dflash_convert.py`, then pass the sidecar:

```sh
./ds4 -m gguf/MiMo-V2.6-Flash-IQ2_XXS-Q2_K-Q8Attn.gguf \
  --dflash gguf/MiMo-V2.6-Flash-DFlash-Q8.gguf
```

The drafter reads the target's residual stream after layers 0, 11, 23, 35
and 47 for every committed token. `fc` and `hidden_norm` turn those five
rows into one context row, and each draft layer keeps its context K/V in a
1024-row ring. Each cycle drafts a block of eight rows: the sampled token
through the target embedding, then seven copies of the trained mask
embedding. The block attends to its whole self (no causal mask) and to
the context inside the 1024-token window, and the target output head
turns rows 1..7 into seven drafts in one pass. The draft layers are
Qwen3-style: per-head q/k RMSNorm, RoPE on 64 of 128 dims, a per-head sink
logit and values scaled by 0.612. The target verifies the sampled token
and the drafts in one batch. It commits the prefix that matches its
argmax, restores the rejected rows' sliding-window cache slots, and adds
only the committed rows to the drafter context.

`--dflash-draft N` caps the drafts per block (1 to 7, default 7).
`--dflash-p-min P` stops before the draft where the product of the draft
softmax maxima, the drafter's estimate that every draft so far is
accepted, falls below `P` (default 0.4; 0 always sends the full block).
Every verified row costs its own routed experts and output head, so short
blocks are cheaper when the drafter is unsure. Sampling at non-zero
temperature uses the same exact p/q rule as MiMo MTP. `--dflash` and
`--mtp` are alternatives. Metal only. After a disk KV-cache restore the
drafter context starts empty and refills as tokens are committed, which
lowers acceptance but never changes output.
On that first-imatrix Q2 Olympiad prompt, plain decode was 39.63 t/s, the
default seven-draft setting 32.22 t/s, and `--dflash-draft 3` 41.69 t/s.
The shorter setting had only one matched run; benchmark your workload before
assuming a speedup.

The session prints verify cycles, drafts, acceptance and committed tokens
per verify when it ends. `./ds4_test --mimo2-dflash-verify` with
`DS4_TEST_MODEL` set to a MiMo GGUF and `DS4_TEST_DFLASH` set to the sidecar
applies the 2.0-logit oracle to three prompts. It also fails unless some
block commits three or more tokens and calls average more than 1.5 committed
tokens. `make test-dflash-kernels` checks the drafter kernels against a
double-precision reference without a model.

## Sampling and reproducibility

At temperature zero, accepted drafts must match the target's greedy
continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy
drafts are accepted directly. Sampling resumes when the proposed suffix does
not match. This is deliberately more deterministic than ordinary sampling.

Use `--mtp-exact-sampling` to preserve the ordinary target sampling
distribution. Exact mode accepts greedy proposals with their target
probability and samples from the remaining distribution on rejection.

When a verified block crosses a tool sampling-mode boundary (for example
entering tool-call syntax during server decoding), the server rewinds to the
block start and re-evaluates the boundary token so the next sample uses the
new mode. Under exact sampling that rewind restores a pre-verify snapshot of
the recurrent state instead of resetting the graph, so long retained
contexts are not replayed at every boundary.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. For DeepSeek comparisons against the ordinary
target-only path, use `--quality` or `--dspark-strict`; these disable the
speculative acceptance path. They do not promise identical output across
different hardware or execution configurations.

Session-batched serving uses ordinary target decoding instead of combining
DSpark/MTP with the session batch. See [serving](SERVER.md#multiple-sessions).
