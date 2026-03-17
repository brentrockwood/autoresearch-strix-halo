# autoresearch — AMD Strix Halo (ROCm) port

This is a port of [Karpathy's autoresearch project](https://github.com/karpathy/autoresearch)
to AMD's Strix Halo platform via ROCm PyTorch — no NVIDIA GPU required.

Tested on the [Framework Desktop AMD Ryzen AI MAX 300 Series motherboard](https://frame.work/products/framework-desktop-mainboard-amd-ryzen-ai-max-300-series?v=FRAFMK0006)
with 128GB unified memory, running Ubuntu 25.10.

Special thanks to [LokiMetaSmith](https://github.com/LokiMetaSmith) and
[khimaros](https://github.com/khimaros) for [this thread](https://github.com/karpathy/nanochat/discussions/363),
which was essential in getting ROCm working on gfx1151.

---

## What this fork changes

Minimum changes to get autoresearch running on Strix Halo:

- **FA3 → SDPA.** Flash Attention 3 is not available on gfx1151. Replaced with
  `torch.nn.functional.scaled_dot_product_attention`.
- **`WINDOW_PATTERN = "L"`.** Banded/sliding window attention is unsupported on this
  platform. Full attention only.
- **Conservative defaults.** `DEPTH=4`, `DEVICE_BATCH_SIZE=8`. The Strix Halo is
  compute-bound at small batch sizes — more optimizer steps beats larger batches.
- **Single knob.** `TOTAL_BATCH_SIZE` is derived from `DEVICE_BATCH_SIZE` automatically.
  Change one number, nothing breaks.
- **ROCm dependencies.** `pyproject.toml` adds a `rocm-gfx1151` extra pointing to AMD's
  wheel index.
- **AOTriton enabled.** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` set in the script
  environment.

All changes are tagged `# Rockwood Lab (ROCm)` in `train.py`.

---

## Quick start

**Requirements:** AMD Strix Halo system (gfx1151), Ubuntu 25.10, ROCm installed,
user in `render` and `video` groups, `uv` installed.

```bash
# 1. Clone this repo
git clone https://github.com/brentrockwood/autoresearch-strix-halo
cd autoresearch-strix-halo

# 2. Set up environment (creates venv, installs ROCm PyTorch)
bash scripts/setup.sh

# 3. Verify GPU is visible
bash scripts/verify.sh

# 4. Download data and train tokenizer (one-time, ~5 min)
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python prepare.py

# 5. Run a baseline experiment (~5 min training + overhead)
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python train.py

# 6. Start the autonomous agent loop
#    Point Claude Code at program.md and say:
#    "Have a look at program.md and let's kick off a new experiment."
```

> **Important:** Use `.venv/bin/python` directly, not `uv run`. The ROCm wheel is
> installed manually after `uv sync` — `uv run` will silently use the CUDA wheel from
> its lockfile and fail with "No CUDA GPUs are available".

---

## ROCm install

If ROCm is not yet installed on your system:

```bash
sudo bash scripts/install_rocm.sh
# reboot required after
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/setup.sh` | Create venv, install ROCm PyTorch, verify |
| `scripts/install_rocm.sh` | Install ROCm system-wide (Ubuntu 25.10, run as root) |
| `scripts/verify.sh` | Sanity check: GPU visibility, groups, patch status, data |

---

## Original README

Everything below is from the upstream autoresearch README.

---

*One day, frontier AI research used to be done by meat computers in between eating,
sleeping, having other fun, and synchronizing once in a while using sound wave
interconnect in the ritual of "group meeting". That era is long gone. Research is now
entirely the domain of autonomous swarms of AI agents running across compute cluster
megastructures in the skies. The agents claim that we are now in the 10,205th generation
of the code base, in any case no one could tell if that's right or wrong as the "code"
is now a self-modifying binary that has grown beyond human comprehension. This repo is
the story of how it all began. -@karpathy, March 2026*

The idea: give an AI agent a small but real LLM training setup and let it experiment
autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result
improved, keeps or discards, and repeats. You wake up in the morning to a log of
experiments and (hopefully) a better model. The training code here is a simplified
single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core
idea is that you're not touching any of the Python files like you normally would as a
researcher. Instead, you are programming the `program.md` Markdown files that provide
context to the AI agents and set up your autonomous research org.

## How it works

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains
  a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model,
  optimizer (Muon + AdamW), and training loop. **This file is edited and iterated on by
  the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let
  it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding
startup/compilation), regardless of the details of your compute. The metric is
**val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so
architectural changes are fairly compared.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
scripts/        — setup, install, and verify scripts (this fork)
```

## License

MIT
