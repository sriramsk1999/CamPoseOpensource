"""MolmoBot baseline — CamPose sidecar wrapper.

Bridges the CamPose training/eval loop to the vendored MolmoBot model that
lives in ArticuBot (see ``ARTICUBOT_DP/diffusion_policy/policy/
molmobot_image_policy.py``). Mirrors the ArticuBot-sidecar pattern used by
the other DiT/3DFA/RoPE4D wrappers in this directory: sidecar-import, batch
adaptation, normalizer plumbing.

What this wrapper does at a glance
----------------------------------
* Constructor: load the pretrained MolmoBot (Qwen3-4B MoE VLM + SigLIP2 +
  flow-matching action expert), freeze SigLIP2 + projector, install LoRA on
  the LLM linear layers, register q01/q99 stats so the
  pretrained policy's robot_preprocessor normalizes to *our* action /
  state distribution rather than SynthManip's.
* ``forward(batch)`` (training): un-z-score the CamPose batch (qpos, actions,
  gripper) → raw → MolmoBot's robot_preprocessor q01/q99-normalizes →
  example dict list → preprocessor → collator → model.compute_loss.
* ``_predict(batch)`` (eval): same batch un-z-score path, no actions, ask
  MolmoBot for an action chunk (in raw joint targets via
  action_postprocessor.unnormalize_action), then re-z-score so the
  evaluator's ``* action_std + action_mean`` recovers raw meters.

Why double-renormalization?
---------------------------
The CamPose dataloader z-scores qpos and actions against CamPose's own
state_mean/std and action_mean/std (z-score). The pretrained MolmoBot
expects q01/q99-bounded inputs. So at the wrapper boundary we un-z-score
to raw joint targets and let MolmoBot's internal normalizer do its q01/q99
thing. At inference, MolmoBot returns raw joint targets; the evaluator
expects z-scored output (`policy(batch)[B, n_steps, action_dim]` then
``* action_std + action_mean``), so we re-z-score on the way out.

Action / state layout (joint_abs dataset)
-----------------------------------------
* state (8-dim): ``[arm_qpos (7), gripper_qpos (1)]`` — raw radians, single
  timestep. CamPose dataloader emits ``batch["qpos"]`` (z-scored 7-dim arm)
  and ``batch["gripper_qpos"]`` (raw scalar gripper mean). We assemble the
  8-dim state here.
* action (8-dim): ``[arm_joint_targets (7), gripper_command (1)]`` — raw,
  arm targets in radians, gripper in [-1, 1] (robosuite convention).

Cameras
-------
We feed the n_side_cam exo views as the MolmoBot ``image`` list (same as
the other CamPose baselines). The pretrained model expected exo+wrist
slots; we send exo+exo. The ViT is shared across slots so the
distribution shift is bounded — LoRA on the LLM compensates.

Language prompt
---------------
Fixed per-task. Override via the wrapper's ``task_prompt`` arg if needed.
"""
from __future__ import annotations

import os
import sys
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


log = logging.getLogger(__name__)


def _ensure_articubot_on_path() -> None:
    path = os.environ.get("ARTICUBOT_DP") or os.path.expanduser(
        "~/Desktop/ArticuBot/diffusion_policy"
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"ArticuBot diffusion_policy dir not found at {path!r}. "
            f"Set ARTICUBOT_DP env var or symlink it into the default path."
        )
    if path not in sys.path:
        sys.path.insert(0, path)


_NORM_REPO_ID = "campose_stats"

# Hardcoded LoRA config
_LORA_CFG = dict(
    r=8,
    alpha=16,
    dropout=0.0,
    target_modules=[
        "att_proj",   # QKV (fused)
        "attn_out",   # O proj
        "ff_proj",    # gate + up (fused)
        "ff_out",     # down proj
    ],
)


class MolmoBotWrapper(nn.Module):
    """CamPose-side wrapper for the MolmoBot baseline."""

    def __init__(
        self,
        args,
        state_dim: int,
        action_dim: int,
        num_cams: int,
        image_size: int,
        norm_stats: Optional[dict] = None,
    ):
        super().__init__()
        _ensure_articubot_on_path()
        assert args.task_prompt is not None
        from diffusion_policy.policy.molmobot_image_policy import MolmoBotImagePolicy

        self._norm_stats = norm_stats or {}
        self._lr = float(args.lr)
        self._weight_decay = float(args.weight_decay)
        self.horizon = int(args.horizon)
        self.n_action_steps = int(args.n_action_steps)
        self.num_cams = int(num_cams)
        self.state_dim = int(state_dim)        # 8 (arm 7 + gripper 1)
        self.action_dim = int(action_dim)      # 8
        self.task_prompt = args.task_prompt

        from huggingface_hub import snapshot_download
        # MolmoBot-Img-DROID is the single-frame variant: n_obs_steps=1,
        # single_frame=true, max_images=2 — matches CamPose's 2-cam 1-step
        # input. The plain MolmoBot-DROID is multi-frame (4 images, 2 timesteps)
        # and would feed our 2 single-frame inputs into a model expecting 4.
        checkpoint_path = snapshot_download("allenai/MolmoBot-Img-DROID")

        log.info(f"[MolmoBotWrapper] loading from {checkpoint_path}")
        self._print_checkpoint_config(checkpoint_path)

        self.policy = MolmoBotImagePolicy(
            checkpoint_path=checkpoint_path,
            train_mode=True,
            use_bfloat16=bool(getattr(args, "use_fp16", True)),
            freeze_vision=True,
            lora_cfg=_LORA_CFG,
            norm_repo_id=_NORM_REPO_ID,
        )

        assert self.policy.action_dim == self.action_dim, (
            f"MolmoBot action_dim={self.policy.action_dim} != "
            f"wrapper action_dim={self.action_dim} (expected 8 for joint_abs)"
        )

        self._register_campose_stats()
        self._verify_setup()

    # ------------------------------------------------------------------ #
    # Verification prints (first-run sanity)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _print_checkpoint_config(checkpoint_path: str) -> None:
        """V1: print key shape/normalization knobs straight from the checkpoint's
        config.yaml so we can compare against what the wrapper assumes."""
        import os
        import yaml
        cfg_path = os.path.join(checkpoint_path, "config.yaml")
        if not os.path.isfile(cfg_path):
            log.warning(f"[verify V1] no config.yaml at {cfg_path}")
            return
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        m = cfg.get("model", cfg)
        img = (m.get("mm_preprocessor") or {}).get("image") or {}
        rp = m.get("robot_preprocessor") or {}
        print(
            f"[verify V1] n_obs_steps={m.get('n_obs_steps')} "
            f"single_frame={img.get('single_frame')} "
            f"max_images={img.get('max_images')} "
            f"max_frames={(m.get('mm_preprocessor') or {}).get('max_frames')} "
            f"action_dim={m.get('action_dim')} "
            f"action_horizon={m.get('action_horizon')} "
            f"action_norm_mode={rp.get('action_norm_mode')}"
        )

    def _verify_setup(self) -> None:
        """V2 + V3: dump what the LoRA install + freezing actually did, so we
        can confirm our assumptions before launching a multi-day run."""
        from collections import Counter
        model = self.policy.model

        # V2: which modules got LoRA-adapted? Suffix counter reveals whether
        # target_modules names matched anything (MoE blocks lack ff_proj/ff_out).
        adapted = [n for n, mod in model.named_modules() if hasattr(mod, "lora_A")]
        suffixes = Counter(n.rsplit(".", 1)[-1] for n in adapted)
        print(f"[verify V2] LoRA adapted modules: {len(adapted)}  by suffix: {dict(suffixes)}")

        # V3: top-level frozen vs trainable param counts. Anything trainable
        # outside vision_backbone / LoRA / action_expert is unexpected.
        frozen, train = Counter(), Counter()
        for n, p in model.named_parameters():
            top = n.split(".")[0]
            if top == "base_model" and len(n.split(".")) > 2:
                # peft wraps as base_model.model.<orig>; report the orig top
                top = n.split(".")[2]
            (train if p.requires_grad else frozen)[top] += p.numel()
        fmt = lambda c: {k: f"{v/1e6:.1f}M" for k, v in c.items()}
        print(f"[verify V3] frozen top-level:    {fmt(frozen)}")
        print(f"[verify V3] trainable top-level: {fmt(train)}")
        n_train = sum(train.values())
        n_total = n_train + sum(frozen.values())
        print(f"[verify V3] total trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")

    # ------------------------------------------------------------------ #
    # Stat registration                                                   #
    # ------------------------------------------------------------------ #

    def _register_campose_stats(self) -> None:
        ns = self._norm_stats
        for k in ("state_q01", "state_q99", "action_q01", "action_q99"):
            if k not in ns or ns[k] is None:
                raise RuntimeError(
                    f"norm_stats missing {k!r} — re-run get_norm_stats with a "
                    f"joint-action dataset (path must contain 'joint')."
                )
        self.policy.register_norm_stats(
            repo_id=_NORM_REPO_ID,
            state_q01=np.asarray(ns["state_q01"], dtype=np.float32),
            state_q99=np.asarray(ns["state_q99"], dtype=np.float32),
            action_q01=np.asarray(ns["action_q01"], dtype=np.float32),
            action_q99=np.asarray(ns["action_q99"], dtype=np.float32),
        )

    # ------------------------------------------------------------------ #
    # Trainable params surface                                            #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        # Per-group LRs from upstream MolmoBot recipe
        # (train_molmobot.py:566-584, optim.py:66): action_expert trains at
        # ~10x the LLM rate (1e-4 vs 1e-5). Single bundled LR starves the
        # action head, which is the BC signal.
        llm_params, ae_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "action_expert" in n:
                ae_params.append(p)
            else:
                llm_params.append(p)
        n_llm = sum(p.numel() for p in llm_params)
        n_ae = sum(p.numel() for p in ae_params)
        print(
            f"[MolmoBotWrapper] optimizer groups — LLM/LoRA: {n_llm/1e6:.1f}M @ lr={self._lr:.0e}, "
            f"action_expert: {n_ae/1e6:.1f}M @ lr={self._lr * 10:.0e}"
        )
        return torch.optim.AdamW(
            [
                {"params": llm_params, "lr": self._lr,        "weight_decay": self._weight_decay},
                {"params": ae_params,  "lr": self._lr * 10.0, "weight_decay": self._weight_decay},
            ],
        )

    # ------------------------------------------------------------------ #
    # Batch ↔ MolmoBot example-dict adaptation                            #
    # ------------------------------------------------------------------ #

    def _unnormalize_actions(self, batch, norm_stats):
        """CamPose dataloader z-scores actions; recover raw joint targets."""
        actions = batch["actions"][:, : self.horizon]
        am = torch.as_tensor(
            norm_stats["action_mean"], device=actions.device, dtype=actions.dtype,
        )
        ast = torch.as_tensor(
            norm_stats["action_std"], device=actions.device, dtype=actions.dtype,
        )
        return actions * ast + am

    def _unnormalize_state(self, batch, norm_stats):
        """Build 8-dim raw joint state from CamPose's z-scored qpos + raw gripper."""
        qpos = batch["qpos"]                       # (B, 7) z-scored arm joints
        sm = torch.as_tensor(
            norm_stats["state_mean"], device=qpos.device, dtype=qpos.dtype,
        )
        ss = torch.as_tensor(
            norm_stats["state_std"], device=qpos.device, dtype=qpos.dtype,
        )
        arm_raw = qpos * ss + sm                   # (B, 7) raw arm joints
        gripper = batch["gripper_qpos"]            # (B,) raw scalar
        if gripper.dim() == 1:
            gripper = gripper.unsqueeze(-1)        # (B, 1)
        return torch.cat([arm_raw, gripper], dim=-1)   # (B, 8)

    def _image_batch_to_lists(self, image_tensor: torch.Tensor) -> List[List[np.ndarray]]:
        """(B, n_cams, C, H, W) torch float in [0, 1] → list of B lists of
        n_cams numpy ``(H, W, 3) uint8`` arrays (MolmoBot's expected format).

        We feed RGB only — channels 3+ (plucker if present, depth if
        ManiWhere-shaped) are dropped.
        """
        if image_tensor.shape[2] >= 3:
            rgb = image_tensor[:, :, :3]
        else:
            raise ValueError(
                f"image tensor has fewer than 3 channels: {image_tensor.shape}"
            )
        rgb = (rgb.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        rgb = rgb.permute(0, 1, 3, 4, 2).contiguous()   # (B, n_cams, H, W, 3)
        rgb_np = rgb.cpu().numpy()
        return [list(rgb_np[b]) for b in range(rgb_np.shape[0])]

    def _build_examples(self, batch, norm_stats, with_actions: bool):
        """Assemble the per-sample synthmanip-format example dicts MolmoBot's
        preprocessor expects."""
        image_lists = self._image_batch_to_lists(batch["image"])
        state_raw_batch = self._unnormalize_state(batch, norm_stats).detach().cpu().numpy()
        actions_raw_batch = None
        is_pad_batch = None
        if with_actions:
            actions_raw_batch = self._unnormalize_actions(batch, norm_stats).detach().cpu().numpy()
            is_pad_batch = batch["is_pad"][:, : self.horizon].detach().cpu().numpy().astype(bool)

        examples: List[Dict] = []
        # MolmoBot's q01/q99 normalizers are pure numpy — apply per-example.
        state_pre = self.policy.state_preprocessor
        for b in range(len(image_lists)):
            example = {
                "image": image_lists[b],
                "question": self.task_prompt,
                "answers": "",
                "style": "demo",
                "state": state_pre.normalize_state(state_raw_batch[b], _NORM_REPO_ID),
                "action_is_pad": (
                    is_pad_batch[b] if with_actions else np.zeros(self.horizon, dtype=bool)
                ),
                "metadata": {"repo_id": _NORM_REPO_ID},
            }
            if with_actions:
                example["action"] = state_pre.normalize_action(actions_raw_batch[b], _NORM_REPO_ID)
            examples.append(example)
        return examples

    # ------------------------------------------------------------------ #
    # Forward dispatch                                                    #
    # ------------------------------------------------------------------ #

    def forward(self, batch, norm_stats=None):
        if norm_stats is None:
            norm_stats = self._norm_stats
        if "actions" not in batch:
            return self._predict(batch, norm_stats)

        examples = self._build_examples(batch, norm_stats, with_actions=True)
        loss = self.policy.compute_loss(examples)
        return {"loss": loss}

    def _predict(self, batch, norm_stats):
        """Return z-scored action chunk (B, n_action_steps, action_dim).

        MolmoBot returns raw joint targets (q01/q99-unnormalized); CamPose
        evaluator multiplies by action_std + action_mean to un-z-score, so
        we re-z-score here for an identity round-trip.
        """
        examples = self._build_examples(batch, norm_stats, with_actions=False)
        out = self.policy.predict_action(examples)
        raw_actions = out["action"]                          # (B, n_action_steps, action_dim) raw
        am = torch.as_tensor(
            norm_stats["action_mean"], device=raw_actions.device, dtype=raw_actions.dtype,
        )
        ast = torch.as_tensor(
            norm_stats["action_std"], device=raw_actions.device, dtype=raw_actions.dtype,
        )
        return (raw_actions - am) / ast
