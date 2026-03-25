# Tips & Tricks

A collection of practical notes accumulated while working on this project on Clariden. Contributions welcome.

---

## Clariden: Useful Tricks

**Author:** Tymoteusz Kwieciński
**Last updated:** 2026-03-25

---

### Point All Caches to Scratch

Any tool that caches large files should be redirected to scratch — never let them write to `$HOME`, which has limited quota. This is already handled in `cluster/env.toml`, but for local runs or interactive sessions set these manually:

```bash
export HF_HOME=/capstor/scratch/cscs/$USER/feature-probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/$USER/feature-probes/hf_cache
export UV_CACHE_DIR=/capstor/scratch/cscs/$USER/feature-probes/uv_cache
export WANDB_DIR=/capstor/scratch/cscs/$USER/feature-probes/wandb
```

All experiment results are synced to W&B and checkpoints uploaded to HuggingFace anyway, so losing scratch contents is not critical.

---

### Lustre Striping for `.sqsh` Images

CSCS [requires](https://docs.cscs.ch/software/container-engine/run/) the
`ce-images/` directory to be created with specific Lustre striping before the
first `enroot import`. Without it, `.sqsh` reads at job start are slow.

```bash
mkdir -p $SCRATCH/ce-images
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M $SCRATCH/ce-images
```

This only needs to be done once per user.

---

### Keeping Scratch Files Alive

Scratch is purged after **30 days of inactivity without warning**. To reset the inactivity clock before a long break:

```bash
find /capstor/scratch/cscs/$USER/feature-probes -exec touch {} +
```

---

### Interactive Session with VS Code Tunnel

For debugging or interactive development, start an interactive job and attach VS Code to it via a tunnel. CSCS documents this workflow [here](https://docs.cscs.ch/software/vscode/).

```bash
# 1. Start an interactive session
srun --account=infra01 --partition=debug --time=00:30:00 \
    --environment=/users/$USER/feature-probes/cluster/env.toml \
    --pty bash

# 2. Inside the session, start a VS Code tunnel
code tunnel
```

This gives you a full VS Code environment attached directly to the compute node. One could also setup a sbatch script which does the same.
