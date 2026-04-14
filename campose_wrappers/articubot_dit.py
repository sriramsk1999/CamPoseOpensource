"""Adapters that train ArticuBot's flow-matching DiT policies with the
CamPose train loops in ``policy_robosuite/`` and ``policy_maniskill/``.

Two variants are exposed:
  - ``ArticubotDiTWrapper``    → FlowMatchingRoPE4DDiTImagePolicy (geometry-aware)
  - ``ArticubotDiTRGBWrapper`` → FlowMatchingDiTImagePolicy       (RGB-only)

ArticuBot is imported as a sidecar package (no vendoring) — set the
``ARTICUBOT_DP`` env var to the ``ArticuBot/diffusion_policy`` directory
(defaults to ``~/Desktop/ArticuBot/diffusion_policy``).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_articubot_on_path():
    path = os.environ.get("ARTICUBOT_DP") or os.path.expanduser(
        "~/Desktop/ArticuBot/diffusion_policy"
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"ArticuBot diffusion_policy dir not found at {path!r}. "
            "Set ARTICUBOT_DP env var."
        )
    if path not in sys.path:
        sys.path.insert(0, path)


def _passthrough_normalizer(shape_meta):
    """LinearNormalizer with identity per-field normalization."""
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer, SingleFieldLinearNormalizer,
    )

    def _identity(shape):
        shape = tuple(shape)
        scale = torch.ones(shape, dtype=torch.float32)
        offset = torch.zeros(shape, dtype=torch.float32)
        stats = {
            "min": -torch.ones(shape, dtype=torch.float32),
            "max": torch.ones(shape, dtype=torch.float32),
            "mean": torch.zeros(shape, dtype=torch.float32),
            "std": torch.ones(shape, dtype=torch.float32),
        }
        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stats,
        )

    norm = LinearNormalizer()
    norm["action"] = _identity(shape_meta["action"]["shape"])
    for k, attr in shape_meta["obs"].items():
        norm[k] = _identity(attr["shape"])
    return norm


_HIDDEN = 512
# RoPE4DDiT's default output_dim=26 is hardcoded for a specific task;
# override so action_decoder's in_dim (hidden_size=512) matches.
_DIFFUSION_MODEL_CFG = {
    "num_attention_heads": 8,
    "attention_head_dim": _HIDDEN // 8,
    "output_dim": _HIDDEN,
    "num_layers": 12,
}

# Mirror ArticuBot/diffusion_policy/config/visual_encoder/dino_crossview.yaml.
# The encoder's __init__ defaults are 4 but the shipping YAML overrides to 6,
# and its ``visual_encoder_cfg={}`` path would otherwise fall back to 4.
_DINO_CROSSVIEW_CFG = {
    "backbone": "vitb",
    "pretrained": True,
    "alt_start": 6,
    "qknorm_start": 6,
    "rope_start": 6,
    "cat_token": True,
    "camera_noise_cfg": {
        "translation_std": 0.01,
        "rotation_deg_std": 1.5,
        "focal_rel_std": 0.015,
        "principal_point_px_std": 3.0,
    },
}


class _ArticubotWrapperBase(nn.Module):
    """Shared CamPose→ArticuBot batch adaptation + flow-matching loss.

    Subclasses provide:
      ``_build_shape_meta``   — which obs keys to advertise to the policy
      ``_build_policy``       — the concrete FlowMatching*DiTImagePolicy instance
      ``_add_geometry_obs``   — optional per-cam geometry keys (pointmap/extr/intr)
      ``_predict_velocity``   — variant-specific encode → DiT → decoder path
    """

    def __init__(self, args, state_dim, action_dim, num_cams, image_size,
                 norm_stats=None):
        super().__init__()
        self._norm_stats = norm_stats or {}
        self._lr = float(args.lr)
        self._weight_decay = float(args.weight_decay)
        _ensure_articubot_on_path()

        self.n_obs_steps = 1
        self.horizon = int(args.horizon)
        self.n_action_steps = int(args.n_action_steps)
        self.num_cams = num_cams
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        shape_meta = self._build_shape_meta(num_cams, image_size, state_dim, action_dim)
        self._shape_meta = shape_meta
        self.policy = self._build_policy(shape_meta)
        self.policy.normalizer = _passthrough_normalizer(shape_meta)

    # ------------------------------------------------------------------ #
    # Template methods                                                    #
    # ------------------------------------------------------------------ #
    def _build_shape_meta(self, num_cams, image_size, state_dim, action_dim):
        raise NotImplementedError

    def _build_policy(self, shape_meta):
        raise NotImplementedError

    def _add_geometry_obs(self, obs, batch, n_cams):
        """Insert optional per-cam geometry keys into ``obs``. No-op by default."""
        pass

    def _predict_velocity(self, policy, nobs, obs, noisy_actions, t_disc):
        """Variant-specific encode → DiT → decoder path.

        Returns predicted velocity of shape (B, horizon, action_dim).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Batch adaptation                                                    #
    # ------------------------------------------------------------------ #
    def _build_ab_obs(self, batch, norm_stats):
        """CamPose batch dict → ArticuBot obs dict (no action targets).

        Expected CamPose keys:
            image    (B, n_cams, 9, H, W) — RGB | Plucker (first 3 channels = RGB)
            eef_xyz  (B, 3) world frame
            qpos     (B, D_qpos) normalized
        Plus (RoPE4D only, via ``_add_geometry_obs``):
            pointmap            (B, n_cams, 3, H, W)
            cam_extrinsics_full (B, n_cams, 4, 4)
            cam_intrinsics_full (B, n_cams, 3, 3)
        """
        device = batch["image"].device
        B, n_cams, Cimg, H, W = batch["image"].shape
        assert n_cams == self.num_cams, (
            f"num_cams mismatch: wrapper={self.num_cams}, batch={n_cams}"
        )

        rgb = batch["image"][:, :, :3]          # (B, n_cams, 3, H, W)
        rgb_m11 = rgb * 2.0 - 1.0                # [0,1] → [-1,1]

        state_mean = torch.as_tensor(
            norm_stats["state_mean"], dtype=torch.float32, device=device,
        )
        state_std = torch.as_tensor(
            norm_stats["state_std"], dtype=torch.float32, device=device,
        )
        qpos_raw = batch["qpos"] * state_std + state_mean           # (B, D_qpos)
        state_raw = torch.cat([batch["eef_xyz"], qpos_raw], dim=-1)  # (B, state_dim)
        assert state_raw.shape[-1] == self.state_dim, (
            f"state_dim mismatch: wrapper={self.state_dim}, "
            f"actual={state_raw.shape[-1]} (3 eef + {qpos_raw.shape[-1]} qpos)"
        )

        obs = {}
        for i in range(n_cams):
            obs[f"cam{i}_image"] = rgb_m11[:, i].unsqueeze(1)  # (B, To=1, 3, H, W)
        obs["state"] = state_raw.unsqueeze(1)
        self._add_geometry_obs(obs, batch, n_cams)
        return obs

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self._lr, weight_decay=self._weight_decay,
        )

    # ------------------------------------------------------------------ #
    # Forward dispatch: training (masked loss) vs inference (action chunk)
    # ------------------------------------------------------------------ #
    def forward(self, batch, norm_stats=None):
        if norm_stats is None:
            norm_stats = self._norm_stats
        if "actions" not in batch:
            return self._predict(batch, norm_stats)

        obs = self._build_ab_obs(batch, norm_stats)
        actions = batch["actions"][:, : self.horizon]       # (B, horizon, D_act)
        is_pad = batch["is_pad"][:, : self.horizon]         # (B, horizon)
        assert actions.shape[1] == self.horizon, (
            "max_seq_length < horizon — re-export dataset or lower --horizon"
        )

        policy = self.policy
        nobs = policy.normalizer.normalize(obs)
        nactions = policy.normalizer["action"].normalize(actions)
        B = nactions.shape[0]
        device, dtype = nactions.device, nactions.dtype

        from diffusion_policy.common.obs_util import process_observations
        process_observations(nobs, policy.observation_mode)

        noise = torch.randn_like(nactions)
        t = policy._sample_time(B, device=device, dtype=dtype)
        t_bc = t[:, None, None]
        noisy_actions = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise
        t_disc = (t * policy.num_timestep_buckets).long()

        pred_velocity = self._predict_velocity(policy, nobs, obs, noisy_actions, t_disc)

        # Masked MSE over non-padded timesteps.
        mask = (~is_pad).to(dtype=dtype).unsqueeze(-1)  # (B, horizon, 1)
        sq = (pred_velocity - velocity_target) ** 2 * mask
        denom = mask.sum().clamp_min(1.0) * pred_velocity.shape[-1]
        loss = sq.sum() / denom
        return {"loss": loss}

    def _predict(self, batch, norm_stats):
        """Obs-only CamPose batch → action tensor (B, horizon, action_dim)."""
        obs = self._build_ab_obs(batch, norm_stats)
        return self.policy.predict_action(obs)["action_pred"]


class ArticubotDiTWrapper(_ArticubotWrapperBase):
    """Trains FlowMatchingRoPE4DDiTImagePolicy (RoPE4D, geometry-aware)."""

    def _build_shape_meta(self, num_cams, image_size, state_dim, action_dim):
        obs = {}
        for i in range(num_cams):
            obs[f"cam{i}_image"] = {"shape": [3, image_size, image_size], "type": "rgb"}
            obs[f"cam{i}_pointmap"] = {"shape": [3, image_size, image_size], "type": "pointmap"}
            obs[f"cam{i}_extrinsic"] = {"shape": [4, 4], "type": "extrinsic"}
            obs[f"cam{i}_intrinsic"] = {"shape": [3, 3], "type": "intrinsic"}
        obs["state"] = {"shape": [state_dim], "type": "low_dim"}
        return {"obs": obs, "action": {"shape": [action_dim]}}

    def _build_policy(self, shape_meta):
        from diffusion_policy.policy.flow_matching_rope4d_dit_image_policy import (
            FlowMatchingRoPE4DDiTImagePolicy,
        )
        return FlowMatchingRoPE4DDiTImagePolicy(
            shape_meta=shape_meta,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            visual_encoder_type="dino_crossview",
            visual_encoder_cfg=dict(_DINO_CROSSVIEW_CFG),
            crop_shape=(self.image_size, self.image_size),
            input_embedding_dim=_HIDDEN,
            hidden_size=_HIDDEN,
            diffusion_model_cfg=_DIFFUSION_MODEL_CFG,
        )

    def _add_geometry_obs(self, obs, batch, n_cams):
        pm = batch["pointmap"]                    # (B, n_cams, 3, H, W)
        extr = batch["cam_extrinsics_full"]       # (B, n_cams, 4, 4)
        intr = batch["cam_intrinsics_full"]       # (B, n_cams, 3, 3)
        for i in range(n_cams):
            obs[f"cam{i}_pointmap"] = pm[:, i].unsqueeze(1)
            obs[f"cam{i}_extrinsic"] = extr[:, i].unsqueeze(1)
            obs[f"cam{i}_intrinsic"] = intr[:, i].unsqueeze(1)

    def _predict_velocity(self, policy, nobs, obs, noisy_actions, t_disc):
        visual_tokens, state_tokens, visual_pos, state_pos = policy._encode_obs(
            nobs, raw_obs=obs,
        )
        action_features = policy.action_encoder(noisy_actions, t_disc)

        gripper_xyz = obs["state"][:, policy.n_obs_steps - 1, :3]
        action_pos = policy._build_action_pos(gripper_xyz)
        if state_pos is not None:
            hidden_pos = torch.cat([state_pos, action_pos], dim=1)
        else:
            hidden_pos = action_pos

        dit_out = policy._run_dit(
            action_features, visual_tokens, state_tokens, t_disc,
            hidden_pos=hidden_pos, encoder_pos=visual_pos,
        )
        return policy.action_decoder(dit_out)


class ArticubotDiTRGBWrapper(_ArticubotWrapperBase):
    """Trains FlowMatchingDiTImagePolicy (RGB-only, no pointmap/extrinsic)."""

    def _build_shape_meta(self, num_cams, image_size, state_dim, action_dim):
        obs = {}
        for i in range(num_cams):
            obs[f"cam{i}_image"] = {"shape": [3, image_size, image_size], "type": "rgb"}
        obs["state"] = {"shape": [state_dim], "type": "low_dim"}
        return {"obs": obs, "action": {"shape": [action_dim]}}

    def _build_policy(self, shape_meta):
        from diffusion_policy.policy.flow_matching_dit_image_policy import (
            FlowMatchingDiTImagePolicy,
        )
        return FlowMatchingDiTImagePolicy(
            shape_meta=shape_meta,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            visual_encoder_type="dino_crossview",
            visual_encoder_cfg=dict(_DINO_CROSSVIEW_CFG),
            crop_shape=(self.image_size, self.image_size),
            input_embedding_dim=_HIDDEN,
            hidden_size=_HIDDEN,
            diffusion_model_cfg=_DIFFUSION_MODEL_CFG,
        )

    def _predict_velocity(self, policy, nobs, obs, noisy_actions, t_disc):
        B = noisy_actions.shape[0]
        visual_tokens, state_tokens = policy._encode_obs(nobs, B)
        action_features = policy.action_encoder(noisy_actions, t_disc)
        if policy.pos_embed_type == "pos":
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long, device=action_features.device,
            )
            action_features = action_features + policy.position_embedding(pos_ids).unsqueeze(0)
        dit_out = policy._run_dit(action_features, visual_tokens, state_tokens, t_disc)
        return policy.action_decoder(dit_out)
