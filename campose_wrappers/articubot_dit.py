"""Adapters that train ArticuBot's flow-matching DiT policies with the
CamPose train loops in ``policy_robosuite/`` and ``policy_maniskill/``.

Two variants are exposed (= the two GROOT baselines we evaluate against):
  - ``ArticubotRoPE4DWrapper``         → FlowMatchingRoPE4DDiTImagePolicy
                                          (GROOT-DINO-CV-RoPE4D: RoPE4D DiT +
                                          DA3 cross-view DINO with CameraEnc +
                                          pointmap geometry)
  - ``ArticubotDiTSingleViewWrapper``  → FlowMatchingDiTImagePolicy
                                          (GROOT-DINO-SV-Plucker: vanilla DiT +
                                          single-view DINOv2 + Plucker ViT)

ArticuBot is imported as a sidecar package (no vendoring) — set the
``ARTICUBOT_DP`` env var to the ``ArticuBot/diffusion_policy`` directory
(defaults to ``~/Desktop/ArticuBot/diffusion_policy``).
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _c2w_gl_to_w2c_cv(c2w_gl: torch.Tensor) -> torch.Tensor:
    """Convert a batched (..., 4, 4) c2w in OpenGL/mujoco convention to
    w2c in OpenCV convention. OpenGL → OpenCV flips the Y and Z camera
    axes (right-multiply by diag(1, -1, -1, 1)); c2w → w2c inverts.
    """
    flip = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        device=c2w_gl.device, dtype=c2w_gl.dtype,
    )
    c2w_cv = c2w_gl @ flip
    R = c2w_cv[..., :3, :3]
    t = c2w_cv[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    w2c = torch.zeros_like(c2w_cv)
    w2c[..., :3, :3] = R_inv
    w2c[..., :3, 3] = t_inv
    w2c[..., 3, 3] = 1.0
    return w2c


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


def _build_normalizer(shape_meta, norm_stats):
    """Build the policy's ``LinearNormalizer``.

    ``action``: real z-score over raw action deltas. With this in place
       ``policy.normalizer["action"].normalize`` z-scores raw → z-score and
       ``.unnormalize`` recovers meters, so RoPE4D's cumsum-based action
       positions land in pointmap world coords without any monkey-patching.
       ``input_stats`` carries real min/max so 3DFA's ``set_normalizer``
       path can pull them.
    obs keys (``state``, images, pointmaps, …): identity passthrough — state
       mixes meters + radians and the policy reads ``raw_obs["state"][..., :3]``
       directly for spatial anchors, so z-scoring state would be a net loss.

    Wrapper boundary un-z-scores ``batch["actions"]`` on entry (the
    dataloader emits z-scored) and re-z-scores the policy's raw-meter
    output before returning, so the evaluator's ``* std + mean`` step still
    produces correct raw meters.

    IMPORTANT: ``SingleFieldLinearNormalizer.create_manual`` builds an
    ``nn.ParameterDict`` whose entries default to ``requires_grad=True``.
    Upstream's ``set_normalizer`` calls ``requires_grad_(False)`` after
    loading; mirror that or AdamW + weight_decay slowly drift the
    scale/offset during training (fits per-step loss, breaks integration).
    """
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer, SingleFieldLinearNormalizer,
    )

    def _passthrough(shape):
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

    mean = torch.as_tensor(norm_stats["action_mean"], dtype=torch.float32).flatten()
    std = torch.as_tensor(norm_stats["action_std"], dtype=torch.float32).flatten().clamp_min(1e-6)
    a_min = torch.as_tensor(norm_stats["action_min"], dtype=torch.float32).flatten()
    a_max = torch.as_tensor(norm_stats["action_max"], dtype=torch.float32).flatten()

    norm = LinearNormalizer()
    norm["action"] = SingleFieldLinearNormalizer.create_manual(
        scale=1.0 / std,
        offset=-mean / std,
        input_stats_dict={"mean": mean, "std": std, "min": a_min, "max": a_max},
    )
    for k, attr in shape_meta["obs"].items():
        norm[k] = _passthrough(attr["shape"])
    norm.requires_grad_(False)
    return norm


# Match ArticuBot/diffusion_policy/config/train_flow_matching_{,rope4d_}dit_workspace.yaml.
_HIDDEN = 1024
_HEAD_DIM = 64
_NUM_LAYERS = 12

# Shared DiT params (both FlowMatchingDiTImagePolicy and FlowMatchingRoPE4DDiTImagePolicy).
# Default output_dim=26 is hardcoded for an ArticuBot task; override to hidden_size
# so action_decoder (in_dim=hidden_size) matches DiT output.
_DIFFUSION_MODEL_CFG_BASE = {
    "num_attention_heads": _HIDDEN // _HEAD_DIM,
    "attention_head_dim": _HEAD_DIM,
    "output_dim": _HIDDEN,
    "num_layers": _NUM_LAYERS,
}

# RGB variant adds interleave_self_attention=True (BasicTransformerBlock-based DiT).
_DIFFUSION_MODEL_CFG_RGB = {
    **_DIFFUSION_MODEL_CFG_BASE,
    "interleave_self_attention": True,
}

# RoPE4D variant adds RoPE4D-specific base_frequency. interleave_self_attention is
# rejected by RoPE4DDiT's constructor, so it does NOT inherit _DIFFUSION_MODEL_CFG_RGB.
_DIFFUSION_MODEL_CFG_ROPE4D = {
    **_DIFFUSION_MODEL_CFG_BASE,
    "base_frequency": 100.0,
}

# Mirror ArticuBot/diffusion_policy/config/visual_encoder/dino_crossview_da3.yaml
# (the DepthAnything3-pretrained variant — alt/qknorm/rope_start pinned to 4 for
# byte-compatible state_dict load, no camera noise). Override include_camera_enc
# because the GROOT-DINO-CV-RoPE4D baseline uses CameraEnc (geometry-aware
# camera tokens injected at alt_start).
_DINO_CROSSVIEW_DA3_CFG = {
    "backbone": "vitb",
    "pretrained": True,
    "weights_source": "da3",
    "alt_start": 4,
    "qknorm_start": 4,
    "rope_start": 4,
    "cat_token": True,
    "include_camera_enc": True,
    "camera_noise_cfg": None,
}

# Mirror ArticuBot/diffusion_policy/config/visual_encoder/dinov2_plucker.yaml.
# Single-view DINOv2-base with last 8 of 12 transformer blocks + final LN
# unfrozen; sibling Plucker ViT trained from scratch over 6-channel rays.
_DINOV2_PLUCKER_CFG = {
    "model_name": "facebook/dinov2-base",
    "frozen": True,
    "num_unfrozen_blocks": 8,
    "plucker_vit_cfg": {
        "patch_size": 14,
        "embed_dim": 384,
        "depth": 4,
        "num_heads": 6,
        "mlp_ratio": 4.0,
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
        self.policy.normalizer = _build_normalizer(shape_meta, self._norm_stats)
        self._post_normalizer_setup()

    def _post_normalizer_setup(self):
        """Hook for subclasses that need extra setup after the normalizer is
        installed (e.g. 3DFA's ``workspace_normalizer`` population). No-op
        by default.
        """
        pass

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

    def _predict_velocity(self, policy, nobs, obs, noisy_actions, t_disc, t_cont=None):
        """Variant-specific encode → DiT → decoder path.

        ``t_cont`` is the continuous flow-matching time in [0, 1]; passed
        through so subclasses that build trajectory-aware action positions
        (RoPE4D) can interpolate gripper_xyz → predicted endpoint.

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
    # Wrapper-boundary normalization helpers                              #
    # ------------------------------------------------------------------ #
    def _actions_raw(self, batch, norm_stats):
        """Undo the dataloader's action z-score so the policy sees raw deltas.

        The dataloader z-scores actions to ``(raw - mean) / std``; the policy
        carries a real ``LinearNormalizer["action"]`` over raw, so we hand it
        raw and let the policy re-normalize internally (and unnormalize for
        RoPE4D's cumsum etc.).
        """
        actions = batch["actions"][:, : self.horizon]
        am = torch.as_tensor(norm_stats["action_mean"], device=actions.device, dtype=actions.dtype)
        ast = torch.as_tensor(norm_stats["action_std"], device=actions.device, dtype=actions.dtype)
        return actions * ast + am

    def _zscore_actions(self, raw_actions, norm_stats):
        """Re-z-score raw policy output so the evaluator's ``* std + mean``
        recovers raw meters."""
        am = torch.as_tensor(
            norm_stats["action_mean"], device=raw_actions.device, dtype=raw_actions.dtype,
        )
        ast = torch.as_tensor(
            norm_stats["action_std"], device=raw_actions.device, dtype=raw_actions.dtype,
        )
        return (raw_actions - am) / ast

    # ------------------------------------------------------------------ #
    # Forward dispatch: training (masked loss) vs inference (action chunk)
    # ------------------------------------------------------------------ #
    def forward(self, batch, norm_stats=None):
        if norm_stats is None:
            norm_stats = self._norm_stats
        if "actions" not in batch:
            return self._predict(batch, norm_stats)

        obs = self._build_ab_obs(batch, norm_stats)
        actions_raw = self._actions_raw(batch, norm_stats)   # (B, horizon, D_act)
        is_pad = batch["is_pad"][:, : self.horizon]
        assert actions_raw.shape[1] == self.horizon, (
            "max_seq_length < horizon — re-export dataset or lower --horizon"
        )

        policy = self.policy
        nobs = policy.normalizer.normalize(obs)
        nactions = policy.normalizer["action"].normalize(actions_raw)   # real z-score
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

        pred_velocity = self._predict_velocity(
            policy, nobs, obs, noisy_actions, t_disc, t_cont=t,
        )

        # Masked MSE over non-padded timesteps.
        mask = (~is_pad).to(dtype=dtype).unsqueeze(-1)  # (B, horizon, 1)
        sq = (pred_velocity - velocity_target) ** 2 * mask
        denom = mask.sum().clamp_min(1.0) * pred_velocity.shape[-1]
        loss = sq.sum() / denom
        return {"loss": loss}

    def _predict(self, batch, norm_stats):
        """Obs-only CamPose batch → z-scored action tensor (B, n_action_steps, action_dim).

        ``policy.predict_action`` returns actions in raw meters (the policy's
        real ``unnormalize`` runs at the end of its sampling loop). The
        CamPose evaluator does ``action * action_std + action_mean`` to
        un-normalize, so we re-z-score here for an identity round-trip.

        Uses ``predict_action`` -> "action" (the n_action_steps-truncated chunk)
        rather than "action_pred" (full horizon) — open-looping the full horizon
        is what flow-matching policies are most fragile to.
        """
        obs = self._build_ab_obs(batch, norm_stats)
        raw_actions = self.policy.predict_action(obs)["action"]
        return self._zscore_actions(raw_actions, norm_stats)


class ArticubotRoPE4DWrapper(_ArticubotWrapperBase):
    """GROOT-DINO-CV-RoPE4D: RoPE4D DiT + DA3-pretrained cross-view DINO.

    Mirrors ArticuBot's ``visual_encoder=dino_crossview_da3`` +
    ``train_flow_matching_rope4d_dit_workspace`` recipe, with CameraEnc
    enabled and the auxiliary pointmap loss disabled. The DA3 path also
    loads pretrained CameraEnc weights (see ``load_pretrained_da3_cam_enc``).
    """

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
            visual_encoder_cfg=dict(_DINO_CROSSVIEW_DA3_CFG),
            crop_shape=(self.image_size, self.image_size),
            input_embedding_dim=_HIDDEN,
            hidden_size=_HIDDEN,
            diffusion_model_cfg=dict(_DIFFUSION_MODEL_CFG_ROPE4D),
            # World coords are in meters (~0–1.5). xyz_scale=100 puts them in
            # RoPE's resolvable frequency band; time_scale=18 maps t to integer
            # step indices over n_obs_steps + horizon.
            xyz_scale=100.0,
            time_scale=18.0,
            # Aux pointmap loss intentionally disabled — flip on by passing a
            # weight (e.g. 0.1) to mirror the ArticuBot reference run.
            aux_pointmap_loss_weight=None,
        )

    def _add_geometry_obs(self, obs, batch, n_cams):
        pm = batch["pointmap"]                    # (B, n_cams, 3, H, W)
        extr = batch["cam_extrinsics_full"]       # (B, n_cams, 4, 4) c2w-GL
        intr = batch["cam_intrinsics_full"]       # (B, n_cams, 3, 3)
        # Upstream CameraEnc + DA3 weights expect w2c in OpenCV convention
        # (dino_cross_view_encoder.py:697,721). The dataloader stores c2w in
        # mujoco/GL convention; convert here so DA3-pretrained pose tokens see
        # the geometry they were trained on.
        extr_w2c_cv = _c2w_gl_to_w2c_cv(extr)
        for i in range(n_cams):
            obs[f"cam{i}_pointmap"] = pm[:, i].unsqueeze(1)
            obs[f"cam{i}_extrinsic"] = extr_w2c_cv[:, i].unsqueeze(1)
            obs[f"cam{i}_intrinsic"] = intr[:, i].unsqueeze(1)

    def _predict_velocity(self, policy, nobs, obs, noisy_actions, t_disc, t_cont=None):
        visual_tokens, state_tokens, visual_pos, state_pos = policy._encode_obs(
            nobs, raw_obs=obs,
        )
        action_features = policy.action_encoder(noisy_actions, t_disc)

        gripper_xyz = obs["state"][:, policy.n_obs_steps - 1, :3]
        # Match upstream's compute_loss: build action positions from the
        # current noisy trajectory estimate. Static positions (no noisy_actions)
        # diverge from inference, where predict_action *does* refresh positions
        # from the in-flight trajectory at every denoising step.
        action_pos = policy._build_action_pos(
            gripper_xyz, noisy_actions=noisy_actions, t_cont=t_cont,
        )
        if state_pos is not None:
            hidden_pos = torch.cat([state_pos, action_pos], dim=1)
        else:
            hidden_pos = action_pos

        dit_out = policy._run_dit(
            action_features, visual_tokens, state_tokens, t_disc,
            hidden_pos=hidden_pos, encoder_pos=visual_pos,
        )
        return policy.action_decoder(dit_out)


_DINOV2_BARE_CFG = {
    "model_name": "facebook/dinov2-base",
    "frozen": True,
    "num_unfrozen_blocks": 8,
}


class ArticubotDiTSingleViewWrapper(_ArticubotWrapperBase):
    """Trains FlowMatchingDiTImagePolicy with single-view DINOv2.

    With ``use_plucker=True`` (default, GROOT-DINO-SV-Plucker): each camera is
    encoded by a partially fine-tuned DINOv2; sibling Plucker ViT tokens are
    fused token-wise before the projector.

    With ``use_plucker=False`` (canonical-view baseline, GROOT-DINO-SV): just
    the partially fine-tuned DINOv2; no plucker stream, no fusion. Pair with
    the canonical-view dataloader path so input RGB already encodes the scene
    from a fixed viewpoint.
    """

    def __init__(self, *args, use_plucker: bool = True, **kwargs):
        self._use_plucker = use_plucker
        super().__init__(*args, **kwargs)

    def _build_shape_meta(self, num_cams, image_size, state_dim, action_dim):
        obs = {}
        for i in range(num_cams):
            obs[f"cam{i}_image"] = {"shape": [3, image_size, image_size], "type": "rgb"}
            if self._use_plucker:
                obs[f"cam{i}_plucker"] = {"shape": [6, image_size, image_size], "type": "plucker"}
        obs["state"] = {"shape": [state_dim], "type": "low_dim"}
        return {"obs": obs, "action": {"shape": [action_dim]}}

    def _build_policy(self, shape_meta):
        from diffusion_policy.policy.flow_matching_dit_image_policy import (
            FlowMatchingDiTImagePolicy,
        )
        encoder_type = "dinov2_plucker" if self._use_plucker else "dinov2"
        encoder_cfg = dict(_DINOV2_PLUCKER_CFG if self._use_plucker else _DINOV2_BARE_CFG)
        return FlowMatchingDiTImagePolicy(
            shape_meta=shape_meta,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            visual_encoder_type=encoder_type,
            visual_encoder_cfg=encoder_cfg,
            crop_shape=(self.image_size, self.image_size),
            input_embedding_dim=_HIDDEN,
            hidden_size=_HIDDEN,
            pos_embed_type="none",
            diffusion_model_cfg=dict(_DIFFUSION_MODEL_CFG_RGB),
        )

    def _add_geometry_obs(self, obs, batch, n_cams):
        if not self._use_plucker:
            return
        # Channels 3-9 of the CamPose-loader 9-channel image are 6-D Plucker rays.
        plucker = batch["image"][:, :, 3:9]      # (B, n_cams, 6, H, W)
        for i in range(n_cams):
            obs[f"cam{i}_plucker"] = plucker[:, i].unsqueeze(1)

    def _predict_velocity(self, policy, nobs, obs, noisy_actions, t_disc, t_cont=None):
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
