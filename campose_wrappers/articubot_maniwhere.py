"""ManiWhere (Yuan et al., 2024) sidecar wrapper.

Pairs ArticuBot's ``FlowMatchingAdditivePosDiTImagePolicy`` (additive-pos DiT
action head) with the vendored ManiWhere visual encoder + auxiliary losses
(see ``diffusion_policy/model/maniwhere/`` and ``visual_encoders/maniwhere``).
ManiWhere is RL-published; this wrapper trains its visual representation
contribution via BC over CamPose demos. The auxiliary InfoNCE / L2 losses
are computed across the (move, fixed) view pair the dataloader emits.

Why DiT not ACT: pick is symmetric with ``dit_dino_sv`` / ``dit_dino_sv``
canonical baseline so the action head is fixed and the only varying piece
is the visual representation.

STN stabilization
-----------------
Upstream ManiWhere is RL — its ``aux_latency=150000`` (camera_aug_config.yaml:62)
freezes the STNs for the first 150k env steps so the base encoder converges
before any aux-driven STN updates. We saw classic NaN-divergence around BC
step ~2k with the unconstrained STN active from step 0. To port the same
contract to BC scale:
  - ``stn_warmup_steps``: STN params frozen for the first N optimizer steps.
  - Weight regularizer (``stn_reg_weight``) on the STN's final FC weights —
    keeps the predicted homography close to identity (init: weight=0,
    bias=eye, so output theta=I exactly when weight=0).
  - Theta stats logged each step so drift is visible in wandb (max |theta|
    and the perspective-row norm ``||theta[2, :2]||``).
"""
import torch
import torch.nn as nn

from campose_wrappers.articubot_dit import (
    _ArticubotWrapperBase,
    _DIFFUSION_MODEL_CFG_ADDITIVE,
    _HIDDEN,
)


# Mirror upstream maniwhere/cfgs/camera_aug_config.yaml (aux_coef=1, aux_l2_coef=1)
# and the paper's reported temperature τ=0.1 (the yaml's 1.0 is a typo / leftover
# from RL ablations). Used for both contrastive and L2 weighting.
_MANIWHERE_AUX_COEF = 1.0
_MANIWHERE_L2_COEF = 1.0
_MANIWHERE_TEMP = 0.1

# STN stabilization knobs. _WARMUP_STEPS at 2000 mirrors the "stabilize base
# encoder first" intent of upstream's aux_latency, scaled from 150k RL steps
# to a typical BC training horizon. _REG_WEIGHT is small — the regularizer's
# only job is to pull the predicted homography back toward identity after
# unfreezing, not to dominate the BC loss.
_STN_WARMUP_STEPS = 2000
_STN_REG_WEIGHT = 1e-2

_MANIWHERE_ENCODER_CFG = {
    "pretrained": True,
    "aux_feature_dim": 256,
    "aux_temp": _MANIWHERE_TEMP,
}


class ArticubotManiWhereWrapper(_ArticubotWrapperBase):
    """dit_maniwhere_sv: additive-pos DiT + ManiWhere ResNet18+STN encoder + aux losses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # STN stabilization plumbing (see module docstring). Defaults from the
        # module-level constants; safe to mutate after construction.
        self._stn_warmup_steps = _STN_WARMUP_STEPS
        self._stn_reg_weight = _STN_REG_WEIGHT
        self._train_step = 0
        self._theta_capture: dict[str, torch.Tensor] = {}
        self._stn_unfrozen = False

        self._freeze_stns()
        self._register_theta_hooks()

    # ------------------------------------------------------------------ #
    # STN stabilization helpers                                           #
    # ------------------------------------------------------------------ #

    def _stn_modules(self):
        """Yield the two STN modules under the maniwhere encoder."""
        enc = self.policy.visual_encoder.encoder
        return enc.input_stn, enc.conv1_stn

    def _freeze_stns(self):
        for stn in self._stn_modules():
            for p in stn.parameters():
                p.requires_grad = False

    def _unfreeze_stns(self):
        for stn in self._stn_modules():
            for p in stn.parameters():
                p.requires_grad = True
        self._stn_unfrozen = True

    def _stn_weight_reg(self) -> torch.Tensor:
        """L2 on the STN final-FC weights. Init is weight=0 + bias=eye, so
        small weights ↔ theta ≈ identity ↔ warp_perspective stays well-posed.
        """
        reg = 0.0
        for stn in self._stn_modules():
            reg = reg + stn.fc_loc[2].weight.pow(2).sum()
        return reg

    def _register_theta_hooks(self):
        """Capture each STN's predicted theta (3x3) via a forward hook on its
        final FC. Read after each forward pass for logging.
        """
        enc = self.policy.visual_encoder.encoder

        def make_hook(name):
            def hook(module, inp, out):
                # out: (B*, 9) → (B*, 3, 3)
                self._theta_capture[name] = out.detach().reshape(-1, 3, 3)
            return hook

        enc.input_stn.fc_loc[2].register_forward_hook(make_hook("input_stn"))
        enc.conv1_stn.fc_loc[2].register_forward_hook(make_hook("conv1_stn"))

    def _theta_stats(self) -> dict:
        """Summarize captured thetas for wandb. Bottom-row norm is the key
        warp_perspective stability signal: small ||theta[2, :2]|| keeps the
        perspective denominator near 1 across the image.
        """
        stats = {}
        for name, theta in self._theta_capture.items():
            stats[f"theta_{name}_max_abs"] = theta.abs().max()
            # Bottom row [theta[2,0], theta[2,1], theta[2,2]] — first two
            # control perspective skew; third is the homography scale.
            stats[f"theta_{name}_perspective_norm"] = (
                theta[..., 2, :2].pow(2).sum(-1).sqrt().mean()
            )
            stats[f"theta_{name}_diag_dev"] = (
                (theta[..., 2, 2] - 1.0).abs().mean()
            )
        return stats

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
        from diffusion_policy.policy.flow_matching_additive_pos_dit_image_policy import (
            FlowMatchingAdditivePosDiTImagePolicy,
        )
        return FlowMatchingAdditivePosDiTImagePolicy(
            shape_meta=shape_meta,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            visual_encoder_type="maniwhere",
            visual_encoder_cfg=dict(_MANIWHERE_ENCODER_CFG),
            crop_shape=(self.image_size, self.image_size),
            input_embedding_dim=_HIDDEN,
            hidden_size=_HIDDEN,
            diffusion_model_cfg=dict(_DIFFUSION_MODEL_CFG_ADDITIVE),
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

        # ManiWhere's PerspectiveSTN (kornia.warp_perspective) divides by the
        # bottom-row perspective coefficient. In bf16, when warp_perspective's
        # backward sees a small denominator, the gradient through 1/z² blows
        # up to NaN. Run the encoder + DiT + aux losses in fp32 to avoid this
        # — other policies stay on bf16 via train.py's autocast block.
        with torch.autocast("cuda", enabled=False):
            return self._forward_fp32(batch, norm_stats)

    def _forward_fp32(self, batch, norm_stats):
        # ----- Adapt batch to ArticuBot obs dict (move + fixed pair) -----
        obs = self._build_ab_obs(batch, norm_stats)
        actions_raw = self._actions_raw(batch, norm_stats).float()
        is_pad = batch["is_pad"][:, : self.horizon]
        assert actions_raw.shape[1] == self.horizon

        policy = self.policy
        # Cast all obs entries to fp32 — _build_ab_obs preserves whatever the
        # dataloader emits (fp32 in our pipeline) but the autocast context
        # would otherwise downcast on the way in.
        obs = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in obs.items()}
        nobs = policy.normalizer.normalize(obs)
        nactions = policy.normalizer["action"].normalize(actions_raw)
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
        # wrapper's _predict_velocity path; AdditivePosDiT applies positional
        # embeds internally so we hand action_features through unchanged.
        visual_tokens, state_tokens = policy._encode_obs(nobs, B)
        action_features = policy.action_encoder(noisy_actions, t_disc)
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

        # ----- STN warmup gate (RL curriculum ported to BC scale) -----
        # Upstream maniwhere freezes the STN for ~150k env steps via
        # aux_latency. We freeze at construction (see __init__) and unfreeze
        # at the first training step past _stn_warmup_steps so the base
        # encoder + DiT settle before the STN starts learning.
        if self.training:
            self._train_step += 1
            if (
                not self._stn_unfrozen
                and self._train_step >= self._stn_warmup_steps
            ):
                self._unfreeze_stns()

        # STN weight regularizer — pulls predicted theta toward identity
        # even after unfreezing, so warp_perspective's 1/z² backward stays
        # well-conditioned. Cheap (two scalar L2 norms).
        stn_reg = self._stn_weight_reg()

        loss = fm_loss + aux_loss + self._stn_reg_weight * stn_reg
        out = {
            "loss": loss,
            "fm_loss": fm_loss.detach(),
            "aux_contrastive": aux["contrastive"].detach(),
            "aux_l2_final": aux["l2_final"].detach(),
            "aux_l2_layers": aux["l2_layers"].detach(),
            "stn_reg": stn_reg.detach() if torch.is_tensor(stn_reg) else torch.zeros((), device=device),
            "stn_unfrozen": torch.tensor(float(self._stn_unfrozen), device=device),
        }
        out.update(self._theta_stats())
        return out

    def _predict(self, batch, norm_stats):
        """Force fp32 at inference as well — STN warp_perspective NaNs the
        same way during the forward pass under bf16 autocast.
        """
        with torch.autocast("cuda", enabled=False):
            return super()._predict(batch, norm_stats)
