"""ManiWhere (Yuan et al., 2024) sidecar wrapper.

Pairs ArticuBot's ``FlowMatchingDiTImagePolicy`` (vanilla DiT action head)
with the vendored ManiWhere visual encoder + auxiliary losses (see
``diffusion_policy/model/maniwhere/`` and ``visual_encoders/maniwhere``).
ManiWhere is RL-published; this wrapper trains its visual representation
contribution via BC over CamPose demos. The auxiliary InfoNCE / L2 losses
are computed across the (move, fixed) view pair the dataloader emits.

Why DiT not ACT: pick is symmetric with ``dit_dino_sv`` / ``dit_dino_sv``
canonical baseline so the action head is fixed and the only varying piece
is the visual representation.
"""
import torch
import torch.nn as nn

from campose_wrappers.articubot_dit import (
    _ArticubotWrapperBase,
    _DIFFUSION_MODEL_CFG_RGB,
    _HIDDEN,
)


# Mirror upstream maniwhere/cfgs/camera_aug_config.yaml (aux_coef=1, aux_l2_coef=1)
# and the paper's reported temperature τ=0.1 (the yaml's 1.0 is a typo / leftover
# from RL ablations). Used for both contrastive and L2 weighting.
_MANIWHERE_AUX_COEF = 1.0
_MANIWHERE_L2_COEF = 1.0
_MANIWHERE_TEMP = 0.1

_MANIWHERE_ENCODER_CFG = {
    "pretrained": True,
    "aux_feature_dim": 256,
    "aux_temp": _MANIWHERE_TEMP,
}


class ArticubotManiWhereWrapper(_ArticubotWrapperBase):
    """dit_maniwhere_sv: vanilla DiT + ManiWhere ResNet18+STN encoder + aux losses."""

    def _build_shape_meta(self, num_cams, image_size, state_dim, action_dim):
        # ManiWhere takes RGB-D (3+1) per cam. Both move (cam{i}_image) and
        # fixed (cam{i}_fixed_image) are 4-channel; the encoder splits by
        # the 'fixed' substring in the key name.
        obs = {}
        for i in range(num_cams):
            obs[f"cam{i}_image"] = {"shape": [4, image_size, image_size], "type": "rgb"}
            obs[f"cam{i}_fixed_image"] = {"shape": [4, image_size, image_size], "type": "rgb"}
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
            visual_encoder_type="maniwhere",
            visual_encoder_cfg=dict(_MANIWHERE_ENCODER_CFG),
            crop_shape=(self.image_size, self.image_size),
            input_embedding_dim=_HIDDEN,
            hidden_size=_HIDDEN,
            pos_embed_type="none",
            diffusion_model_cfg=dict(_DIFFUSION_MODEL_CFG_RGB),
        )

    def _build_ab_obs(self, batch, norm_stats):
        """Override: ManiWhere wants [0,1] RGBD + fixed-view pair, not [-1,1].

        Inputs from the dataloader:
          batch['image']        : (B, n_cams, 4, H, W)  RGBD [0,1] — move views
          batch['image_fixed']  : (B, n_cams, 4, H, W)  RGBD [0,1] — fixed views
          batch['eef_xyz'], batch['qpos'] (z-scored).
        """
        device = batch["image"].device
        B, n_cams, Cimg, H, W = batch["image"].shape
        assert n_cams == self.num_cams, (
            f"num_cams mismatch: wrapper={self.num_cams}, batch={n_cams}"
        )
        assert Cimg == 4, (
            f"ManiWhere expects 4-channel RGBD input, got {Cimg}-channel image."
        )
        assert "image_fixed" in batch, (
            "ManiWhere wrapper needs batch['image_fixed'] — dataloader "
            "must set args.policy_class='dit_maniwhere_sv' to enable it."
        )

        rgbd = batch["image"]                 # (B, n_cams, 4, H, W) [0,1]
        rgbd_fixed = batch["image_fixed"]      # (B, n_cams, 4, H, W) [0,1]

        state_mean = torch.as_tensor(
            norm_stats["state_mean"], dtype=torch.float32, device=device,
        )
        state_std = torch.as_tensor(
            norm_stats["state_std"], dtype=torch.float32, device=device,
        )
        qpos_raw = batch["qpos"] * state_std + state_mean
        state_raw = torch.cat([batch["eef_xyz"], qpos_raw], dim=-1)

        obs = {}
        for i in range(n_cams):
            obs[f"cam{i}_image"] = rgbd[:, i].unsqueeze(1)         # (B, To=1, 4, H, W)
            obs[f"cam{i}_fixed_image"] = rgbd_fixed[:, i].unsqueeze(1)
        obs["state"] = state_raw.unsqueeze(1)
        return obs

    def forward(self, batch, norm_stats=None):
        if norm_stats is None:
            norm_stats = self._norm_stats
        if "actions" not in batch:
            return self._predict(batch, norm_stats)

        # ----- Adapt batch to ArticuBot obs dict (move + fixed pair) -----
        obs = self._build_ab_obs(batch, norm_stats)
        actions = batch["actions"][:, : self.horizon]
        is_pad = batch["is_pad"][:, : self.horizon]
        assert actions.shape[1] == self.horizon

        policy = self.policy
        nobs = policy.normalizer.normalize(obs)
        nactions = policy.normalizer["action"].normalize(actions)
        B = nactions.shape[0]
        device, dtype = nactions.device, nactions.dtype

        from diffusion_policy.common.obs_util import process_observations
        process_observations(nobs, policy.observation_mode)

        # ----- Standard flow-matching loss (RGB path: same as DiT-SV) -----
        noise = torch.randn_like(nactions)
        t = policy._sample_time(B, device=device, dtype=dtype)
        t_bc = t[:, None, None]
        noisy_actions = (1 - t_bc) * noise + t_bc * nactions
        velocity_target = nactions - noise
        t_disc = (t * policy.num_timestep_buckets).long()

        # Encoder forward (move keys only — the encoder reuses this internally
        # when compute_aux_losses runs below). Identical to the SingleView
        # wrapper's _predict_velocity path.
        visual_tokens, state_tokens = policy._encode_obs(nobs, B)
        action_features = policy.action_encoder(noisy_actions, t_disc)
        if policy.pos_embed_type == "pos":
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long, device=action_features.device,
            )
            action_features = action_features + policy.position_embedding(pos_ids).unsqueeze(0)
        dit_out = policy._run_dit(action_features, visual_tokens, state_tokens, t_disc)
        pred_velocity = policy.action_decoder(dit_out)

        mask = (~is_pad).to(dtype=dtype).unsqueeze(-1)
        sq = (pred_velocity - velocity_target) ** 2 * mask
        denom = mask.sum().clamp_min(1.0) * pred_velocity.shape[-1]
        fm_loss = sq.sum() / denom

        # ----- ManiWhere auxiliary losses on the (move, fixed) pair -----
        aux = policy.visual_encoder.compute_aux_losses(nobs)
        aux_loss = (
            _MANIWHERE_AUX_COEF * aux["contrastive"]
            + _MANIWHERE_L2_COEF * (aux["l2_final"] + aux["l2_layers"])
        )

        loss = fm_loss + aux_loss
        return {
            "loss": loss,
            "fm_loss": fm_loss.detach(),
            "aux_contrastive": aux["contrastive"].detach(),
            "aux_l2_final": aux["l2_final"].detach(),
            "aux_l2_layers": aux["l2_layers"].detach(),
        }
