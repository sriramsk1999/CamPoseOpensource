---
name: reproduce-paper
description: Reproduce numbers from "Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning" (Jiang et al., ICRA 2026). Use when the user asks to reproduce a figure, table, row, or experiment from the paper (e.g. "reproduce fig 6 lift randomized with conditioning", "rerun table 1 row 3"). Resolves the user's request to an entry in `reproduce/paper_runs.yaml` and prints the exact `reproduce/reproduce.py` command (or cluster job script) to run.
---

# Reproducing paper numbers

Follow this workflow whenever the user wants to reproduce a number from the paper. Default to **print and confirm**: never launch a training run without the user's explicit go-ahead.

## 1. Identify the experiment

Every paper experiment is registered in `reproduce/paper_runs.yaml`. Read that file first — do not rely on memory, the list of figures/entries can change.

`reproduce/reproduce.py` takes three required arguments:

- `--paper_item`: a top-level key under `paper_items:` (e.g. `fig6`, `fig7`, `table1`).
- `--exp`: a sub-key under that `paper_item` (e.g. `lift_randomized_with_conditioning`).
- `--seed`: an integer present in `experiments[<exp_id>].seeds` (typically `[0, 1, 2]`). One run = one seed; for multi-seed results, fan out in step 3.

Rules:

- Never invent a `paper_item` or `exp`. Explicitly verify that `paper_items[<paper_item>][<exp>]` resolves in the YAML before proceeding — a bad pair will crash `reproduce.py` with an unhelpful `KeyError`.
- If the user's description matches multiple entries, or none clearly, list the candidates and ask them to pick one. Do not guess. Ask at most two clarifying questions before falling back to the best-match candidates.
- If the user gives only a figure (e.g. "fig 6 row 3") without naming the task/condition, ask which entry they mean — the `paper_items.<figX>` mapping shows the entry names.
- Default seed: `0` (one run). Only fan out to all listed seeds if the user asks for mean/std across seeds.
- If `reproduce/paper_runs.yaml` has changed substantially since publication, warn the user that the mapping may no longer match the published numbers.

## 2. Check the computing environment

Assume by default the user is running on a cluster, not a single workstation. Ask the user up front:

1. Scheduler/environment (SLURM, PBS, LSF, k8s, cloud VM, local GPU).
2. Cluster settings you'll need for the job script: partition/queue, GPU type and memory, CPU count per task (dataloaders benefit from ≥4), account/project, time limit, and where logs should go.
3. How conda (or equivalent) activates on a **compute node** — some clusters need `source <conda>/etc/profile.d/conda.sh`, others use `module load ...`, others use mamba/micromamba. Do not assume.
4. wandb plan: either run `wandb login` ahead of time, or set `WANDB_MODE=offline` in the job script. `train.py` calls `wandb.init(...)` unconditionally and will hang on an interactive login prompt under batch schedulers otherwise.

Before launching, verify (on the machine that will actually run the training, which on a cluster is the compute node — not the login node):

- Current directory is the `CamPoseOpensource` repo root (`README.md` starts with "Do You Know Where Your Camera Is?"). `reproduce.py` does not care about CWD, but the job script must `cd` to the repo root so relative paths and `wandb` output land in the right place.
- The conda env `know_your_camera` can be activated; if not, point the user at `bash setup.sh`.
- A GPU is visible: `nvidia-smi`.
- The entrypoint can import its full stack. Look up `experiments[<exp_id>].entrypoint` in the YAML — it will be `policy_robosuite/train.py` or `policy_maniskill/train.py` — and run `python <entrypoint> --help`. That is the most reliable check because it exercises the same imports as the real run (including the vendored `robosuite_source` / `maniskill_source`). If the user commented out either side in `setup.sh`, it must be installed first.
- The demo file the experiment needs is actually on disk. `train.py` loads `policy_<side>/demos/<dataset_suffix>.hdf5`, where `<dataset_suffix>` is either the YAML override or the default (`liftrand_eef_delta` for robosuite). A missing or truncated file surfaces much later as an opaque HDF5 error. If the file is missing or obviously incomplete, rerun `bash setup.sh`.

If the user only has a single machine, warn that one seed typically takes hours to a day of wall-clock time and ask them to confirm they want to proceed anyway.

## 3. Print and confirm (do not auto-launch)

Default behavior: print the exact command(s) — and, on a cluster, a proposed job script — and wait for the user's explicit go-ahead. Do not submit or run the training yourself unless the user clearly asks you to.

The core command (from the repo root) is always:

```
python reproduce/reproduce.py --paper_item <paper_item> --exp <exp> --seed <seed>
```

On a cluster, wrap that command in a minimal job script tailored to the scheduler the user described in step 2. Use the user's partition/GPU/CPU/time/account settings — do not invent them. If the user didn't give a time limit, suggest 24h as a starting point (and tell them to watch the first few evaluation epochs to calibrate for future runs). For SLURM, a one-seed job looks roughly like:

```
#!/usr/bin/env bash
#SBATCH -J reproduce_<paper_item>_<exp>_seed_<seed>
#SBATCH -p <user_partition>
#SBATCH -G 1
#SBATCH --cpus-per-task=<user_cpus>            # >=4 recommended
#SBATCH --constraint=<user_gpu_mem>            # or --gres=gpu:<type>:1; ensure enough GPU memory
#SBATCH --time=<user_time_limit>               # default suggestion: 24:00:00
#SBATCH -o <user_log_dir>/%x_%j.out
#SBATCH -e <user_log_dir>/%x_%j.out

set -euo pipefail

<user_conda_activation>                        # e.g. source <conda>/etc/profile.d/conda.sh; conda activate know_your_camera
# Optional: export WANDB_MODE=offline          # if the user prefers offline wandb
cd <path_to_repo_root>
python reproduce/reproduce.py --paper_item <paper_item> --exp <exp> --seed <seed>
```

For non-SLURM schedulers, produce the equivalent minimal script (e.g. PBS `#PBS` headers + `qsub`, LSF `#BSUB` + `bsub`). Keep placeholders for anything the user hasn't specified — do not fill them in with guesses.

For multiple seeds, print one job script (or one command) per seed so the user can submit them in parallel. Only launch or submit after the user confirms.

## 4. Set expectations

After launching, tell the user:

- Artifacts: checkpoints go to `policy_<side>/checkpoints/<name>/` and wandb logs to the `test` project under a group derived from `<name>`. Evaluation success rates appear in wandb as `success_rate_*`, starting at `eval_start_epoch` (see the YAML overrides for that entry).
- To judge whether a run reproduces the paper, point the user at the wandb `success_rate_*` curves for that run — the paper's numbers come from the same metric. Do not assert a specific pass/fail threshold here; it depends on the figure.
- Results will not be bitwise-identical across machines (this is not guaranteed on modern GPUs; see https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ for background), but numbers should match the paper in expectation.
- If the final metrics look off, or anything else goes wrong, please open an issue on the repo or email `tianchongj [at] ttic [dot] edu`.
