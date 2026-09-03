# Table 1 — Square Assembly

The three Table-1 baselines on robosuite Square Assembly (`squarerand_eef_delta`),
seeds 1/2/3. Cabinet lives in the ArticuBot repo (`paper/icra/`, `paper/41510/`);
Square Assembly has no task config there, so it runs here.

## The arms

All three use the **same** visual encoder — cross-view DINOv2 ViT-B with
`alt/qknorm/rope_start=4`, blocks 0–3 frozen — the same DiT width/depth, and the
same optimizer. The only variable is how camera pose reaches the policy.

| arm | `--policy_class` | pose enters as | table row |
|---|---|---|---|
| `canonical` | `dit_dino_cv` (`--use_canonical_views 1`) | nothing; the two chosen cameras are reprojected into fixed canonical viewpoints, so geometry is baked into the RGB | Canonical Views |
| `plucker` | `dit_dino_cv` (`--use_plucker 1`) | 6-channel Plücker raymap, fused into the patch tokens (channel concat → project down) | Plücker Raymaps |
| `vgp` | `dit_rope4d_dino_cv` | RoPE4D token positions off the metric pointmap, action tokens anchored on the current noisy trajectory (`action_pos_mode=cumsum`) | VGP |

This matches ArticuBot `paper/icra/README.md`'s definition of VGP
(`visual_encoder=dino_crossview` + 4D RoPE + noisy-action positions), which is
**not** what this repo used to build: the RoPE4D wrapper previously used the
DepthAnything3 encoder with CameraEnc, and inherited `action_pos_mode=blended`
from the policy's Python default. The ICRA resubmission drops DA3, so both were
changed. Old `dit_rope4d_dino_cv_square_seed_0` checkpoints are from the DA3
recipe and are not comparable to these runs.

The Plücker arm needed a new encoder, which lives at
`campose_wrappers/crossview_plucker.py` — **ArticuBot is untouched**. The
pre-existing upstream `dinov2_plucker` is single-view, so pairing it with
plücker would have confounded "how pose enters" with "which backbone encodes
the image" — the thing this table is trying to isolate. That file's docstring
explains how it attaches without modifying the sidecar.

## Running it

```bash
DRY_RUN=1 paper/square/run_all.sh    # print the 9-run plan
paper/square/run_all.sh              # launch, 2 background workers
```

One job at a time per GPU, seed-major order, so all three arms finish at seed 1
before seed 2 starts. Single runs:

```bash
GPU=0 paper/square/table1.sh vgp 1
GPU=1 EPOCHS=201 EVAL_MAX_STEPS=30 paper/square/table1.sh plucker 1   # smoke test
```

Everything that matters is passed explicitly — several argparse defaults do not
match this recipe (`lr` 2e-5, `lr_scheduler` const, `batch_size` 70,
`prob_drop_proprio` 1.0, `num_side_cam` 1, `n` 3, `m` 1).

Re-running an arm+seed **resumes** from the newest `epoch_*.pth` in
`policy_robosuite/checkpoints/<arm>_square_seed_<N>/`, so interrupting the sweep
is safe. Note the checkpoint stores the EMA weights, so a resume restarts the
online policy from the EMA copy rather than being bit-exact.

## Cost, and why it is one job per GPU

~40 h per run at 30k epochs, batch 35. Five runs on GPU0 and four on GPU1, so
the full sweep is **roughly 8 days**. Checkpoints are 4 GB each, keep-3, so
~12 GB per run and ~110 GB for all nine.

`JOBS_PER_GPU=2` was tried at batch 35 and **does not fit**. Measured on this
box (2× RTX 4090, 24 GiB):

| | steady-state training | epoch-0 eval phase |
|---|---|---|
| `vgp` | 14.5 GiB | 3.2 GiB |
| `plucker` | 16.5 GiB | 3.4 GiB |

Two training jobs are 29–33 GiB against a 24 GiB card. In the 2-per-GPU
attempt, whichever job reached the training loop first ran fine and the second
OOM'd — always at the eval→training transition, never in steady state, because
the eval phase runs under `no_grad` with no optimizer state and its 3 GiB
footprint hides what the job is about to need. Do not size a co-tenant against
the eval-phase number.

Note also that **`CUDA_VISIBLE_DEVICES` does not confine these jobs to one
GPU**: mujoco's EGL renderer opens a graphics context on the *other* card,
225 MiB–1.1 GiB per job (the `G` rows in `nvidia-smi`, vs `C` for compute). The
two GPUs are therefore not independent when packing them.

Co-location is still the right instinct — the GPU is badly under-used at one
job per GPU. Measured utilisation is mean 40% with 79% of samples under 50%,
because `num_workers=0` forces mujoco to render every frame on the main
process; with two jobs per GPU it went to 100%/64% before the OOMs. Getting
that headroom without OOM means either dropping to batch ~16–18 (which changes
the recipe, so only do it if all nine cells use it) or fixing the dataloader —
pre-rendering camera views to disk, or a worker pool. The latter is worth more
and touches no hyperparameter.

Results land in wandb project `CamPose_training`, grouped as `<arm>_square`,
with per-epoch `success_rate_train_cameras` / `success_rate_test_cameras`.
Test-camera success is the table number.
