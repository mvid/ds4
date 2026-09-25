# MiMo V2.6 Flash first-party continuations

100 prompts from `gguf-tools/quality-testing/prompts.jsonl`, collected through
OpenRouter on September 24, 2026 with provider routing pinned to Xiaomi
(`xiaomi/fp8`, fallbacks disabled). Every response reports model
`xiaomi/mimo-v2.6-flash` and provider `Xiaomi`.

Temperature is 0, reasoning is disabled (`reasoning.effort=none`; every
response reports zero reasoning tokens), no system message is sent, and the
output limit is 24 tokens. Neither OpenRouter endpoint for this model returns
token logprobs, so the responses contain text only. `collection.json` records
the request settings; raw responses are retained.

## Scoring

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/mimo-v2.6-flash-20260924-openrouter/manifest.tsv \
  /tmp/mimo-scores.tsv 4096
```

Add `--ssd-streaming` for the MXFP4 file on a 128 GB Mac. The scorer warns
that no API logprobs were parsed; that is expected. Only teacher-forced NLL,
first-token agreement and greedy LCP are meaningful here. API top-token and
pair metrics are zero because there is no API probability data.

The local no-think renderer and API usage report the same prompt-token count
for all 100 cases. This checks framing length, not byte-identical token IDs.

The hosted model serves FP8 weights, so even the source-precision MXFP4 file
is not expected to reproduce every greedy continuation.

The first MiMo Q2 imatrix corpus contained all 100 of these prompts. Q2
results below use the rebuilt corpus
`gguf-tools/imatrix/dataset/mimo26-v2.6-flash-clean.txt`, which has no exact
fixture match and a maximum word Jaccard similarity of 0.278 to any prompt.

## Results

| File | Average NLL | First-token match | Average greedy LCP |
| --- | ---: | ---: | ---: |
| `MiMo-V2.6-Flash-IQ2_XXS-Q2_K-Q8Attn.gguf` (clean imatrix) | 0.339311 | 65/100 | 6.99 |
| `MiMo-V2.6-Flash-IQ2_XXS-Q2_K-AttnQ4.gguf` (clean imatrix) | 0.356618 | 62/100 | 6.23 |
| `MiMo-V2.6-Flash-MXFP4-Q8Attn.gguf` (SSD) | 0.262704 | 92/100 | 16.81 |

The clean imatrix recorded 200 prompts and 35,106 tokens from the resident
weight-energy Q2 bootstrap. It covered 12,020/12,032 expert slots (99.90%);
the 12 unvisited experts use weight-energy importance. Its SHA-256 is
`184ca5c03645e30e69ab81e6a0aa1ed85442268960d8b6e98f1523c73d916214`.
Both Q2 files and MXFP4 passed 508 tensor-layout checks and 27 sampled
expert comparisons against the pinned source revision.

The GLM 5.3 Flash Q2 reference is 89/100 first-token matches on its own
provider fixture. MXFP4 exceeds that number here; neither resident Q2 does.
Across these same 100 cases, MXFP4 and clean Q2 disagree in API-match status
on 29 cases. Their direct local first-token agreement can therefore be at
most 71/100, below the plan's 89/100 pair gate. MXFP4 and AttnQ4 differ
in status on 34 cases, giving a 66/100 upper bound. These are upper bounds,
not measured direct pair agreements. Do not treat the resident Q2 variants
as quality-equivalent to source MXFP4.
