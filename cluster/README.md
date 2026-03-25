# Clariden Cluster

This directory contains the environment definition and job scripts for running
experiments on the [Clariden](https://docs.cscs.ch/clusters/clariden/) cluster at CSCS.
Clariden is part of the [Alps](https://docs.cscs.ch/alps/) platform and uses
GH200 nodes with the
[Container Engine](https://docs.cscs.ch/software/container-engine/) (CE) as the primary
runtime.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Container image built on top of NGC PyTorch 25.06 |
| `build.sbatch` | Builds the image and exports it as a `.sqsh` file to `$SCRATCH/ce-images/` |
| `env.toml` | [Environment Definition File](https://docs.cscs.ch/software/container-engine/run/); defines mounts, env vars, NCCL hooks |
| `train.sbatch` | Single training run (draft) |
| `train_multirun.sbatch` | Parallel sweep across a GPU array (draft) |

## Environment Design

The environment is split into two layers:

```
Image (.sqsh)  — rebuilt only when pyproject.toml changes
──────────────────────────────────────────────────────────
NGC PyTorch 25.06-py3 (torch, transformers, numpy, ...)
+ extra deps installed via uv (wandb, hydra, peft, ...)

Mounted at runtime via env.toml
──────────────────────────────────────────────────────────
$SCRATCH/feature-probes/        ← source code (git pull to update)
~/keys/                  ← HF / W&B tokens
/capstor/scratch/$USER/  ← HF cache, probe checkpoints
```

Code changes never require a rebuild — only `pyproject.toml` changes do, such as adding new dependencies.

> Please note, that this environment contains only the training scripts, not the `annotation` and `generation` pipeline also used in the [old repo](https://github.com/sevdari/hallucination_probes).

## First-Time Setup

**1. Create the image directory with Lustre striping** ([required by CSCS](https://docs.cscs.ch/software/container-engine/run/)):
```bash
mkdir -p $SCRATCH/ce-images
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M $SCRATCH/ce-images
```

**2. Store your API keys:**
```bash
mkdir -p ~/keys
echo "hf_..." > ~/keys/.hf_token
echo "..." > ~/keys/.wandb_key
echo "hf_..." > ~/keys/.hf_token_write  # optional, only for HF uploads
```

**3. Build the image:**
```bash
# first exec into an interactive job
srun --account=infra01 --partition=normal --time=01:00:00 --pty bash
# run the build
bash cluster/build.sh
```
Unfortunately, this needs to be done in an interactive job, because sbatch does not enable NAT connections from the node.

> Note: optionally, you can 'borrow' an existing `enroot` image, e.g. from here: `/iopsstor/scratch/cscs/tkwiecinski/ce-images/feature-probes+25.06.sqsh`

**4. Submit a training job:**
```bash
sbatch cluster/train.sbatch
```

## Rebuilding vs. Updating Code

| What changed | Action needed |
|---|---|
| Source code | `git pull` inside the running container — no rebuild |
| `pyproject.toml` (new dep) | `sbatch cluster/build.sbatch` |
| Base image tag | Update `FROM` in `Dockerfile`, then rebuild |



### Notes

To see some insights about working with the cluster, feel free to browse some [tips](../docs/tips_and_tricks.md).


> Please also note: The current version of [train_multirun](./train_multirun.sbatch) could still be buggy, please refer to [issues](https://github.com/swiss-ai/feature-probes/issues/1).