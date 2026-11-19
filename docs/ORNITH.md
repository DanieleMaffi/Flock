# Ornith-1.5-122B

This page is for people who run [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) in Strata on an
NVIDIA card. It is experimental.

Ornith-1.5-122B has the same architecture as Qwen3.6-35B-A3B (`qwen35moe`), only bigger: 48 layers instead of 40,
3072 wide instead of 2048, experts 1024 wide instead of 512, and the same 256 experts with 8 per token. Strata runs it on
the same engine path ([QWEN36.md](QWEN36.md)).

## Which size

**UD-IQ2_XXS** (42 GB, the default): on two 24 GB cards every routed expert sits in VRAM at a 64K context, and the engine
runs the decode as CUDA graphs over both cards and pipelines prompt chunks across them. That is the fast setup (see
Measured). **UD-Q4_K_XL** (80 GB) is closer to the full model but does not fit: about a third of its experts stay in
VRAM, the CPU does the rest, and it runs several times slower.

## The tiered mode

The engine path used to keep a copy of every expert in RAM. Ornith's experts are 68 GiB in Unsloth's UD-Q4_K_XL, more
than a 64 GB PC has. When the copy would not fit (85% of the available RAM), the engine runs tiered instead:

- The most-used experts sit in VRAM slots, as before.
- There is no RAM copy. The CPU computes the other experts straight from the mapped GGUF, and the OS page cache keeps
  the ones read most in RAM. An expert on a VRAM slot is not read from the file, so VRAM and RAM hold different experts.
- Prompt chunks run like the decode (slots and CPU), not streamed: a streamed chunk reads every expert of every layer
  from the file, more than the page cache holds. `STRATA_Q36_TIERED_STREAM=1` streams them anyway.

`STRATA_Q36_TIERED=1` forces the tiered mode and `=0` turns it off.

## Getting the model

The engine takes Unsloth's files as published. `tools/ornith/fetch.py` downloads a size and the picture reader at a
pinned commit, resumes a broken download and checks every file's size and SHA-256:

```sh
python3 tools/ornith/fetch.py /data/Strata-data/models/ornith-ud-iq2_xxs UD-IQ2_XXS
python3 tools/ornith/fetch.py /data/Strata-data/models/ornith-ud-q4_k_xl              # UD-Q4_K_XL, 80 GB in three files
```

## Running it

```sh
strata launch --family ornith --model ud-iq2_xxs --model-path /data/Strata-data/models/ornith-ud-iq2_xxs --context 64k
```

`ornith` is opt-in. Strata registers it only when the command names `--family ornith`, so the menu and the app do not
list it.

## Two GPUs

With two cards or more, setup splits the layers over the first two: layers 0-23 on the first, 24-47, the MTP block and
the output head on the second (`STRATA_Q36_SPLIT=auto` in the run config; a number sets the first layer of the second
card, `off` runs on one). Each card fills its free VRAM with the experts its layers use most, so about half of every
layer's experts sit in VRAM instead of an eighth. Between the cards only the hidden rows travel, once per forward, by
explicit copies: GeForce cards have no peer access. Pictures are off with the split so far.

`STRATA_Q36_SPLIT=auto` picks the layer that balances what each card holds (all of a layer's tensors; the embeddings on
both cards; the MTP block and the head on the second). The split's prompt chunks are 1,024 tokens.

**Every expert in VRAM.** When each card's experts fit beside its buffers with 0.5 GiB to spare (the engine prints both
cards' numbers), there is no hybrid mode: no CPU pool, no host copy, and the decode runs as CUDA graphs over both cards
(a start-up probe checks the driver takes such a graph). A prompt of several chunks is pipelined: the first card starts
chunk n + 1 while the second runs chunk n (`STRATA_Q36_PIPE=0` turns it off).

**When they do not all fit**, a prompt chunk above 1,024 rows streams each layer's other experts into a staging pair on
that layer's card, through a copy thread per card with pinned bounce buffers (`STRATA_Q36_GATHER=0`: the plain copies),
and runs the whole layer on the GPU. `STRATA_Q36_SPLIT_STREAM=0` runs prompt chunks as the decode does instead.

**The attention cache in int8** (`STRATA_Q36_KV=int8`, which setup sets for Ornith): a row of 256 values in 256 bytes
plus one scale, a quarter of fp32. About 1.8 GB for 13 attention layers at 128K instead of 6.8.

**Expert-usage profile.** `data/ornith/<size>.profile` holds how often the router picked each expert on agent-style
text; setup passes it as `STRATA_Q36_PROFILE`, and the engine fills each layer's VRAM slots in that order at start-up.
`STRATA_Q36_PROFILE_OUT` records one, `tools/ornith/profile_curve.py` prints its curve. Ornith's routing (UD-Q4_K_XL,
48 layers, 13.5 M picks): the top 64 of 256 experts per layer take 49% of the work, the top 128 74%, the top 192 91%,
the top 224 96.5%.

With the same experts on the GPU (`--expert-slots 32`, `STRATA_Q36_FIXED_SLOTS=1`), one card and two give the same
tokens, bit for bit, on all 10 parity prompts.

## Measured

> **Measured on:** two RTX 3090 24 GB (PCIe, no peer access), i9-11900 (8 cores, AVX-512), 62 GB RAM, NVMe; greedy,
> the 10 parity prompts (`tools/qwen35/parity_prompts.ids`) x 64 tokens and one 3,445-token prompt; 2026-11-19.

UD-IQ2_XXS, both cards, 64K context, every expert in VRAM, int8 cache, chunk 1024:

| | Decode | 3,445-token prompt |
| --- | ---: | ---: |
| `--mtp 0` (graphs) | 95-101 tok/s | 3.70 s with `STRATA_Q36_PIPE=0` |
| `--mtp 2` (graphs; the same tokens as `--mtp 0` on all 10 prompts) | 96-129 tok/s | 2.58 s pipelined (1,336 tok/s) |
| UD-IQ2_XXS at 128K (not every expert fits: 216 and 167 of 256 per layer) | 46-65 tok/s | 5.0 s |
| llama.cpp `--fit on`, same file | 87-88 tok/s | 2.4 s |

At 128K the second card falls about 0.3 GiB short of every expert; the second copy of the token embeddings it keeps for
the MTP block is 0.43 GiB.

UD-Q4_K_XL, earlier runs (16K context, fp32 cache):

| | Decode | 3,445-token prompt |
| --- | ---: | ---: |
| Strata, one RTX 3090, `--mtp 0` | 12-18 tok/s (17-32 of 256 experts per layer in VRAM) | 102 s (34 tok/s) |
| Strata, both RTX 3090s split, streamed prompts, chunk 2048 (the default) | 13-23 tok/s (96 and 56 of 256 in VRAM) | 46 s (75 tok/s) |
| Strata, both split, streamed prompts, chunk 4096 | 2-16 tok/s (83 and 45 of 256 in VRAM) | 43 s (80 tok/s) |
| Strata, both split, `STRATA_Q36_SPLIT_STREAM=0` | 15-28 tok/s (122 and 89 of 256 in VRAM; 66% of the expert work on the GPUs) | 69 s (50 tok/s) |
| Strata, both split, `--mtp 2` | 5-17 tok/s | |
| llama.cpp `--fit on`, both RTX 3090s, same file | 17-19 tok/s | 34 s (102 tok/s) |

The decode speed climbs over the first prompts while the slots follow the routing.

**Correctness.** Against llama.cpp on the same file (`tools/ornith/llama_ref.py`), every prompt starts with the same tokens.
Where the two later part, `tools/ornith/divergence.py` asks llama.cpp how likely Strata's token was on Strata's own
prefix: in most cases llama.cpp prefers Strata's token itself there. Once (the split run, one prompt) Strata picked a
token llama.cpp gives 0.2%; llama.cpp does the same to its own greedy pick elsewhere (0.2% for its own token, its prompt
and decode paths disagreeing), so a single such step is not an error by itself. Strata's two-card run equals its
one-card run bit for bit (above), so the split adds no error of its own.

## What changed in the engine for it

- The routed experts' Q4_K down projection on the GPU (decode and the int8 prompt kernels).
- The CPU pool's per-token buffers held experts up to 640 wide (Flash-Next's compiled shape). Ornith's 1024-wide experts
  overflowed them and gave fluent but wrong answers. They now hold up to 1024, and a wider layer is refused.
- The tiered mode above.
- The two-GPU layer split, the int8 cache, the fully resident split with CUDA graphs and the prompt pipeline above.
- IQ1_S experts and the 256-value i-quants as down projections (UD-IQ2_XXS), on the decode and the int8 prompt kernels.

## Not done yet

- Every expert in VRAM at 128K and 256K on two 24 GB cards: the second card's spare copy of the token embeddings for
  the MTP block (0.43 GiB) is the next room to take.
- With UD-Q4_K_XL the profile gave no measurable gain on the parity prompts (55.7% of the expert work on the GPUs with and
  without it): the live rebalancing reaches the same placement within a prompt or two.
- The int8 cache and the IQ prompt kernels against llama.cpp step by step; one parity prompt now parts from llama.cpp at
  its first token.
- Pictures with the two-GPU split.
