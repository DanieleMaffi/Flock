# Ornith-1.5-122B

This page is for people who run [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) in Strata on an
NVIDIA card. It is experimental.

Ornith-1.5-122B has the same architecture as Qwen3.6-35B-A3B (`qwen35moe`), only bigger: 48 layers instead of 40,
3072 wide instead of 2048, experts 1024 wide instead of 512, and the same 256 experts with 8 per token. Strata runs it on
the same engine path ([QWEN36.md](QWEN36.md)).

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

The engine takes Unsloth's UD-Q4_K_XL as published. `tools/ornith/fetch.py` downloads its three files (80 GB) and the
picture reader at a pinned commit, resumes a broken download and checks every file's size and SHA-256:

```sh
python3 tools/ornith/fetch.py /data/models/ornith-ud-q4_k_xl
```

## Running it

```sh
strata launch --family ornith --model ud-q4_k_xl --model-path /data/models/ornith-ud-q4_k_xl --context 64k
```

`ornith` is opt-in. Strata registers it only when the command names `--family ornith`, so the menu and the app do not
list it.

## Two GPUs

With two cards or more, setup splits the layers over the first two: layers 0-23 on the first, 24-47, the MTP block and
the output head on the second (`STRATA_Q36_SPLIT=auto` in the run config; a number sets the first layer of the second
card, `off` runs on one). Each card fills its free VRAM with the experts its layers use most, so about half of every
layer's experts sit in VRAM instead of an eighth. Between the cards only the hidden rows travel, once per forward, by
explicit copies: GeForce cards have no peer access. Pictures are off with the split so far.

A prompt chunk above 128 rows streams each layer's other experts into a staging pair on that layer's card, from the
mapped GGUF, and runs the whole layer on the GPU (the int8 prompt kernels). The staging pairs take about 2.9 GB per card
from the slots. `STRATA_Q36_SPLIT_STREAM=0` runs prompt chunks as the decode does instead (more slots, slower prompts).

With the same experts on the GPU (`--expert-slots 32`, `STRATA_Q36_FIXED_SLOTS=1`), one card and two give the same
tokens, bit for bit, on all 10 parity prompts.

## Measured

> **Measured on:** one or two RTX 3090 24 GB (PCIe, no peer access), i9-11900 (8 cores, AVX-512), 62 GB RAM, NVMe, UD-Q4_K_XL, 16K context,
> greedy, the 10 parity prompts (`tools/qwen35/parity_prompts.ids`) x 64 tokens and one 3,445-token prompt.

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
- The two-GPU layer split above.

## Not done yet

- Prompt reading: llama.cpp is still faster on a long prompt (34 s against 46 s). Each streamed chunk copies about 48 GB
  of experts from the page cache: the file is pageable memory, so CUDA stages every copy itself and the host waits for
  it, and the two cards' copies and their compute do not overlap. A copy thread with pinned bounce buffers is next.
- MTP drafting does not pay while a third of the expert work is on the CPU.
- Pictures with the two-GPU split.
