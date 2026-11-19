# Flock

**Ornith-1.5-122B on your own two GPUs, at full GPU speed.**

Flock is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221, specialised for
[Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), a 122B mixture-of-experts model (256 experts, 8 per
token) built for coding and agent work. It runs Ornith on two 24 GB consumer cards with every expert in VRAM, and serves
it through an OpenAI- and Anthropic-compatible API.

> Status: experimental. Measured on one machine (below). The quality comparison with other models is in progress.

## What it does

- **Two GPUs, one model.** The layers are split over two cards; each card holds its layers' weights and experts, and
  only the hidden rows travel between them. No NVLink or peer access needed (GeForce cards have none).
- **Every expert in VRAM.** With Unsloth's UD-IQ2_XXS (42 GB), an int8 attention cache and tight buffers, all of
  Ornith's experts fit on two 24 GB cards at a 64K context. The CPU does no expert work at all.
- **Fast decode.** The token loop runs as CUDA graphs across both cards, and Ornith's own MTP layer drafts tokens ahead:
  96-129 tokens/s.
- **Fast prompts.** Prompt chunks are pipelined across the cards (the first starts chunk n + 1 while the second
  finishes chunk n), on int8 kernels for Ornith's quant formats: a 3,445-token prompt is read in 2.6 s.
- **Bigger quants when you want them.** Sizes that do not fit in VRAM run in a hybrid mode: the busiest experts on the
  GPUs (pre-filled from a recorded usage profile), the rest computed on the CPU straight from the mapped file.

## Measured

> Two RTX 3090 24 GB (PCIe 4.0, no peer access), Intel i9-11900 (8 cores), 62 GB RAM, NVMe; Ornith-1.5-122B
> UD-IQ2_XXS, 64K context; greedy, 64 tokens on each of 10 prompts and one 3,445-token prompt; November 2026.

| | Decode | 3,445-token prompt |
| --- | ---: | ---: |
| Flock, MTP drafting on (the default) | 96-129 tokens/s | 2.6 s |
| Flock, MTP off | 95-101 tokens/s | 2.6 s |
| llama.cpp (`--fit on`), same file and cards | 87-88 tokens/s | 2.4 s |

A 32,000-token prompt takes 28 s (1,130 tokens/s). Where the time goes, and what is still slow:
[docs/ORNITH.md](docs/ORNITH.md).

**Quality.** UD-IQ2_XXS is a 2-bit file: Unsloth measures it at about 87.7% top-1 agreement with the full model in chat,
against 93.8% for UD-Q4_K_XL and 95.3% for Q8_0. Larger sizes run too, more slowly; a coding comparison of the sizes
against other models is under way and will be added here.

## Quick start (Docker)

You need two NVIDIA GPUs with 24 GB each, about 64 GB of RAM, the NVIDIA driver 580 or newer and the NVIDIA Container
Toolkit.

```sh
git clone https://github.com/DanieleMaffi/Flock.git && cd Flock
cp .env.example .env            # set API_KEY, and FLOCK_COMMIT to the output of: git rev-parse HEAD
docker volume create flock_data
docker compose build            # compiles the engine for your cards, 20-40 minutes
```

Download the model into the volume (42 GB; the files are checked against pinned SHA-256 sums, and a broken download
resumes):

```sh
docker run --rm -v flock_data:/data -v "$PWD/tools/ornith:/t:ro" --entrypoint /opt/strata/terminal/python/bin/python3 \
  flock:local /t/fetch.py /data/Strata-data/models/ornith-ud-iq2_xxs UD-IQ2_XXS
```

Start it:

```sh
docker compose up -d && docker compose logs -f
```

The server answers at `http://<host>:8000/v1` (OpenAI) and `http://<host>:8000/v1/messages` (Anthropic), with the
`API_KEY` from `.env`. Ornith is a reasoning model: its thinking comes back in `reasoning_content`.

## Sizes

| Size | File | Fits two 24 GB cards | Notes |
| --- | ---: | --- | --- |
| UD-IQ2_XXS | 42 GB | every expert, at 64K | the default; the speeds above |
| UD-Q3_K_XL | 60 GB | about 63% of the experts | about half the 2-bit size's quality loss; slower |
| UD-Q4_K_XL | 80 GB | about a third | close to the full model; 12-23 tokens/s |

To change size, set `MODEL` and `MODEL_PATH` in `.env`, fetch it with `tools/ornith/fetch.py` and restart.

## Documentation

- [docs/ORNITH.md](docs/ORNITH.md): how Flock runs Ornith, every setting, every measurement, what is not done yet.
- [docs/README.md](docs/README.md): the other documents, most of them Strata's, kept for the engine they describe.

## Credits and license

Flock is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors: the engine, the server
and the tools are theirs; Flock adds the Ornith path (the two-GPU split, the tiered and resident modes, the int8 cache,
the formats and kernels Ornith needs). The model is [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B)
by ornith-ai (MIT); the GGUF files are [Unsloth's](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF).

[MIT License](LICENSE), as Strata.
