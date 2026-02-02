# Fine-tuning PI0.5 (pi05) with OpenPI

This repository documents a **PI0.5 (pi05)** fine-tuning workflow built on top of the Physical-Intelligence **OpenPI** framework.
It keeps the **original OpenPI installation + PyTorch support steps**, and adds the **end-to-end fine-tuning & inference commands**
used for the `pick_and_feed_headmove220` dataset.

> Note: This repo focuses on **code and configuration**. Model checkpoints are **not included**.

---

## Installation (same as upstream OpenPI)

### Clone with submodules

When cloning, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
```

If you already cloned the repo:

```bash
git submodule update --init --recursive
```

### Python env with uv

OpenPI uses `uv` to manage Python dependencies. After installing `uv`, run:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**NOTE:** `GIT_LFS_SKIP_SMUDGE=1` is needed to pull **LeRobot** as a dependency.

### Docker (optional)

If you run into system dependency issues, consider using Docker to simplify setup (see upstream OpenPI “Docker Setup”).

---

## PyTorch Support (same as upstream OpenPI)

OpenPI provides PyTorch implementations of **π₀** and **π₀.₅** alongside the original JAX versions.

### Setup / patches (required)

1) Make sure dependencies are up to date:

```bash
uv sync
```

2) Double check `transformers==4.53.2`:

```bash
uv pip show transformers
```

3) Apply transformers patches:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

These patches overwrite several files in `transformers` to enable:
1) AdaRMS support  
2) correct activation precision control  
3) KV cache usage without being updated  

**WARNING (important):** With the default `uv` link mode (hardlink), this may permanently affect the `transformers` library in your `uv` cache.
To fully undo: `uv cache clean transformers`.

---

## Converting JAX Models to PyTorch (upstream step)

To finetune in PyTorch, you typically convert a JAX base model checkpoint to PyTorch format first.

### Convert JAX base model to PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name <config name> \
    --checkpoint_dir /path/to/jax/base/model \
    --output_path /path/to/pytorch/base/model
```

Then, specify the converted PyTorch model path in your config via `pytorch_weight_path`.

### Convert a general JAX checkpoint to PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

---

## Dataset Preparation (LeRobot-style)

### 1) Prepare a LeRobot-style dataset

Assume your dataset directory is named:

```text
pick_and_feed_headmove220
```

### 2) Copy dataset to the LeRobot cache (machine-dependent)

This step depends on your machine setup and filesystem layout.

Example (copying from one machine to another):

```bash
(base) tu@tams98:~/.cache/huggingface/lerobot$ \
scp -r pick_and_feed_headmove220/ \
tu@tamsgpu6:~/.cache/huggingface/lerobot
```

Expected location after copying:

```text
~/.cache/huggingface/lerobot/pick_and_feed_headmove220
```

---

## Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_pick_and_feed_headmove220
```

---

## Training (Fine-tuning PI0.5)

Example command using 2 GPUs:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/train_pytorch.py \
  pi05_pick_and_feed_headmove220 \
  --exp-name=pick_and_feed \
  --overwrite
```

Notes:
- `pi05_pick_and_feed_headmove220` is the training config name
- `--exp-name` controls the experiment subdirectory
- Outputs/checkpoints are saved under `checkpoints/`

---

## Inference / Serving a Trained Policy

```bash
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_pick_and_feed_headmove220 \
  --policy.dir=checkpoints/pi05_pick_and_feed_headmove220/pick_and_feed/12000
```

---

## Checkpoints

Model checkpoints are **NOT included** in this repository. Train your own, or provide your own converted base weights.

Expected directory structure (example):

```text
checkpoints/
 └── pi05_pick_and_feed_headmove220/
     └── pick_and_feed/
         └── 12000/
```

---

## License

Please refer to the upstream OpenPI repository for license information.

