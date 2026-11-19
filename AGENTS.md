# AGENTS.md

Flock runs Ornith-1.5-122B on two 24 GB GPUs. It is a fork of [Strata](https://github.com/Niko1221/Strata),
version 0.2.15, engine 0.1.60. Read **[docs/ORNITH.md](docs/ORNITH.md)** first: how Ornith runs, every setting and
every measurement.

Inside the code the Strata names stay: the `strata` command, the Python packages (`strata_cli`, `strata_control`,
`serve`), the `STRATA_*` settings and the data folders. Do not rename them.

## Parts

- **The engine** (`src/`, `include/`): C++/CUDA. Ornith (`qwen35moe`) runs on the q36 path,
  `src/qwen35/q36_engine.cu`; its quant kernels are in `src/kernels/cuda/`.
- **The server** (`serve/`): an OpenAI- and Anthropic-compatible API and a web page.
- **Setup and launch** (`setup.py`, `strata.py`, `strata_cli/`): the `ornith` family, its sizes and its engine
  settings are in `setup.py` (`LOCAL_FAMILIES`).
- **Ornith tools** (`tools/ornith/`): `fetch.py` downloads and checks the GGUF files, `code_eval.py` is the coding
  eval, `profile_curve.py` reads an expert-usage profile; recorded profiles are in `data/ornith/`.
- **Docker**: `Dockerfile`, `docker/entrypoint.sh`, and `docker-compose.yml` with `.env.example` at the root.

Strata's desktop app (`apps/`), installers (`packaging/`) and its other model families are still in the tree but are
not what Flock builds or tests.

## Running it

See the Quick start in [README.md](README.md): `docker compose build`, fetch the model into the `flock_data` volume
with `tools/ornith/fetch.py`, `docker compose up -d`. Settings go in `.env`; engine settings in `STRATA_SET`.

## Working on the code

- Tests: `python -m pytest tests/terminal`; setup's own tests run without a GPU: `python tools/test_setup_<name>.py`.
- Build the CUDA engine before asking for review of an engine change, and run Ornith on two cards.
- Give measured numbers with what they were measured on (cards, CPU, size, context, prompt). Make no claim without
  a measurement. Keep the docs' style: plain words, short sentences.
- Strata's engine documentation is kept in [docs/](docs/README.md) for the parts Flock shares.

## Contributing

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**: search first, one problem per pull request, real evidence
only, and mark a finding from reading code as unconfirmed static analysis.
