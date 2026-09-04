# Square Assembly — Tables 1 and 2

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

## Table 2 arms

All nine Table-2 rows run from the same `dit_rope4d_dino_cv` wrapper, one flag
each, so every arm is one change away from VGP:

| arm | row | how |
|---|---|---|
| `dp` | Diffusion Policy | `--policy_class dp` |
| `spatial_softmax` | VGP Spatial Softmax | `--visual_token_compression softmax` |
| `max_pool` | VGP Max Pooling | `--visual_token_compression max` |
| `resnet18` | ResNet-18 | `--visual_encoder resnet18` |
| `dinov2_frozen` | DINOv2 Frozen | `--visual_encoder dinov2_frozen` |
| `dinov2_finetuned` | DINOv2 Finetuned | `--visual_encoder dinov2_finetuned` |
| `act` | ACT | `--policy_class act` |
| `vgp_sinusoidal` | VGP sinusoidal | `--policy_class dit_dino_cv` |
| `rope3d` | VGP Rope3D | `--pos_encoding rope3d` |

Three of these carry a trap, all handled in the wrapper — don't undo them:

- **`resnet18` forces `patch_size=32`.** The policy re-derives token positions
  from the pointmap via `_extract_patch_centers`; the default 14 (ViT-B/14)
  mismatches ResNet-18's 7×7 output stride and training dies. It also drops
  `use_separate_wrist_encoder`, which `resnet.yaml` ships as true.
- **`rope3d` runs at hidden 1008 / head_dim 72, not 1024 / 64.** 3D RoPE splits
  head_dim over 3 axes and asserts `head_dim % 6 == 0`, which 64 fails. This
  mirrors upstream's `train_flow_matching_rope3d_dit_workspace.yaml`, so the
  arm is ~7M params lighter than VGP by construction, not by accident.
- **`vgp_sinusoidal` is not a one-knob change from VGP.** Dropping RoPE also
  removes the pointmap's only consumer, so it trains on RGB with no geometry
  at all — the confound ArticuBot's `paper/icra/README.md` documents. `rope3d`
  is the clean single-knob comparison for the positional-encoding claim.

Verified: every arm builds, takes a training step with finite loss and
gradients, and returns an action chunk. `dinov2_frozen` has 0.8M trainable
encoder params against `dinov2_finetuned`'s 57.5M, so the freeze is real.

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

## On Grogu

    paper/square/submit_grogu.sh                 # table 1: 3 arms x seeds 1 2 3
    DRY_RUN=1 paper/square/submit_grogu.sh       # print, submit nothing
    ARMS="vgp plucker canonical dp act" paper/square/submit_grogu.sh

Defaults to `-p dheld -C rtx3090`: 15 of 16 3090s free vs 0 H200s, tier 5, and
a 14-day cap that a ~3-day run clears without the `CHAIN=` dependency juggling
ArticuBot needs under `deepaklong`. A 3090 is 24 GB, same as the 4090 these
were profiled on, so one job per GPU still applies unchanged.

`grogu.slurm` asks for `-c 4 --mem=24G`, not the `-c 12 --mem=64G` of
ArticuBot's script: CamPose is `num_workers=0` with a 204 MB in-repo dataset,
so a job is ~1 busy core and nothing needs staging. At 64 GB apiece, 8 jobs
would not fit on grogu-4-8's 281 GB.

**Headless rendering.** Nothing in CamPose sets `MUJOCO_GL`, so `grogu.slurm`
exports `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl`. Verified working on Grogu.
Keep those exports: without them mujoco fails silently with black frames rather
than crashing, which looks like a run whose loss falls normally but whose
success rate sits at 0 on both pose files.

`SAVE_EVERY` is 2000 there rather than 5000: every Grogu partition is
`PreemptMode=REQUEUE`, and at ~3.2 s/epoch 5000 would put ~4.4 h of work at
risk per preemption.

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
