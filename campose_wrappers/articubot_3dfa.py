"""3DFA (3D FlowMatch Actor) sidecar wrapper.

Trains ArticuBot's ``FlowMatching3DFAImagePolicy`` (which itself wraps
upstream 3DFA's ``DenoiseActor3D``, see
https://github.com/nickgkan/3d_flowmatch_actor) with the CamPose train
loops. Mirrors the pattern in ``campose_wrappers.articubot_dit`` — sidecar
import via ``ARTICUBOT_DP``, passthrough normalizer, CamPose batch →
ArticuBot obs adapter.

Configured for CamPose's 7-dim ``eef_delta`` action via
``rotation_format='euler'`` (3 xyz + 3 euler + 1 grip = 7).

Loss override: L1 + 30/10/1 coefs  →  normalized MSE
----------------------------------------------------
Three layers of disagreement on the loss for 3DFA:

* The paper (arXiv:2508.11002) describes L2 on velocity (pos + rot) +
  BCE-with-logits on gripper.
* The original 3DFA repo
  (``3d_flowmatch_actor/modeling/policy/base_denoise_actor.py:235-239``)
  uses ``30 * L1(pos) + 10 * L1(rot) + BCE(gripper)`` — keeps BCE on
  gripper but uses L1 for pos/rot with hardcoded coefficients inherited
  from 3D Diffuser Actor / PerAct.
* The ArticuBot vendored copy at
  ``ArticuBot/.../flowmatch_3dfa/policy/base_denoise_actor.py:230-234``
  silently changed BCE → L1 during the port, leaving every term as L1.

Empirically the vendored loss collapses 3DFA on Square: xyz/rot
under-predicted by ~5×, persistent +0.03 rot drift, success at 0%
(diagnose_3dfa_realobs.py against a 20k-step ckpt).

We swap to a single MSE over the full ``[pos_velocity, rot_velocity,
openess]`` vector in normalized space, equal per-dim weighting — same
loss shape the DiT/RoPE4D wrappers in this same codebase use, and they
train successfully on the same dataset. Gripper stays as a regression
target (not BCE) because (a) it kept the patch minimum-change, (b) on
real obs the model already predicts the right gripper values for the
phases it's seen (only the synthetic-noise "no context" prior collapsed
in diagnose_3dfa.py), and (c) BCE would require extra inference-time
sigmoid + scale plumbing that compute_trajectory in the vendored copy
doesn't have. Set ``_USE_MSE_LOSS = False`` to revert.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from campose_wrappers.articubot_dit import _ArticubotWrapperBase


# Flip to False to restore upstream's L1 + 30/10/1 coef loss.
_USE_MSE_LOSS = True


# Mirror ArticuBot/diffusion_policy/config/train_flow_matching_3dfa_workspace.yaml
# (architecture knobs from the upstream train_peract.sh recipe).
_TDFA_MODEL_CFG = {
    "backbone": "clip",
    "finetune_backbone": False,
    "embedding_dim": 120,
    "num_attn_heads": 8,
    "fps_subsampling_factor": 4,
    "num_shared_attn_layers": 4,
    # eef_delta semantics: per-step delta xyz; relative=True triggers the
    # cumsum+query path inside policy_forward_pass so RoPE3D positions
    # land in world frame alongside the pointmap.
    "relative": True,
    # rotation_format='euler' → action_dim 7 to match CamPose eef_delta
    # [pos(3), rot(3), grip(1)]. Approximate (rotvec deltas fed into the
    # euler slot) but stable for small per-step rotations.
    "rotation_format": "euler",
    "denoise_timesteps": 5,
    "denoise_model": "rectified_flow",
    "lv2_batch_size": 1,
}


class Articubot3DFAWrapper(_ArticubotWrapperBase):
    """3DFA sidecar — vanilla DiT/cross-attn over RGB+pointcloud tokens.

    Differences from the DiT wrappers:
      - Visual input is CLIP-backbone RGB (expects [0, 1]) + per-camera
        pointmaps. _build_ab_obs feeds RGB in [0, 1] (not [-1, 1]).
      - 3DFA computes its flow-matching loss internally; ``forward``
        delegates to ``self.policy.compute_loss`` rather than running the
        flow-matching loop here (cf. ``_ArticubotWrapperBase.forward``).
      - No ``_predict_velocity`` — never used.
    """

    def _build_shape_meta(self, num_cams, image_size, state_dim, action_dim):
        obs = {}
        for i in range(num_cams):
            obs[f"cam{i}_image"] = {"shape": [3, image_size, image_size], "type": "rgb"}
            obs[f"cam{i}_pointmap"] = {"shape": [3, image_size, image_size], "type": "pointmap"}
        obs["state"] = {"shape": [state_dim], "type": "low_dim"}
        return {"obs": obs, "action": {"shape": [action_dim]}}

    def _build_policy(self, shape_meta):
        from diffusion_policy.policy.flow_matching_3dfa_image_policy import (
            FlowMatching3DFAImagePolicy,
        )
        return FlowMatching3DFAImagePolicy(
            shape_meta=shape_meta,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            n_obs_steps=self.n_obs_steps,
            **_TDFA_MODEL_CFG,
        )

    def _post_normalizer_setup(self):
        """Populate ``DenoiseActor3D.workspace_normalizer`` with raw action
        min/max. Upstream wires this via ``set_normalizer``; we install the
        policy normalizer directly (see ``_ArticubotWrapperBase.__init__``),
        so the buffer would otherwise stay at its default
        ``[[0,0,0,0,0,0],[1,1,1,1,1,1]]`` — which collapses meter-scale gt
        deltas to ~-1 in ``normalize_pos`` and breaks the RoPE3D query
        positions (``unnormalize_pos(noisy) + cumsum`` would live in
        arbitrary units rather than the pointmap's world frame).

        Note: upstream's ``set_normalizer`` only copies ``[:3]`` (xyz) into
        the workspace buffer, but for euler the buffer is 6-wide (xyz + 3
        euler), so doing it ourselves here is also a correctness fix.
        """
        norm_stats = self._norm_stats
        policy = self.policy
        nrm_dim = int(policy.model.workspace_normalizer.size(-1))
        action_min = torch.as_tensor(
            norm_stats["action_min"], dtype=torch.float32,
        ).reshape(-1)[:nrm_dim]
        action_max = torch.as_tensor(
            norm_stats["action_max"], dtype=torch.float32,
        ).reshape(-1)[:nrm_dim]
        # Guard against degenerate ranges (zero spread) — replace with ±1
        # fallback to keep normalize_pos finite.
        eps = 1e-3
        spread = action_max - action_min
        for i in range(nrm_dim):
            if spread[i].abs() < eps:
                action_min[i] = -1.0
                action_max[i] = 1.0
        with torch.no_grad():
            policy.model.workspace_normalizer.copy_(
                torch.stack([action_min, action_max])
            )

        if _USE_MSE_LOSS:
            self._install_mse_loss()

    def _install_mse_loss(self):
        """Monkey-patch ``self.policy.model.compute_loss`` to use one MSE
        over the full ``[pos_velocity, rot_velocity, openess]`` prediction
        vector in normalized space — paper-aligned, DiT-wrapper-aligned.

        Verbatim copy of upstream's compute_loss flow (encode → noise
        schedule → policy_forward_pass → per-layer accumulation) except the
        per-layer loss is now a single MSE over the 7-dim target.

        Target construction (still normalized space, same as upstream):
          target[..., :3] = noise[..., :3] - gt_pos    (pos velocity)
          target[..., 3:6] = noise[..., 3:] - gt_rot   (rot velocity)
          target[..., 6:7] = gt_openess                (gripper raw value;
                              not noised — matches paper which uses BCE here,
                              swap to BCE later if this still under-drives gripper)

        Why monkey-patch rather than subclass: ``DenoiseActor3D`` is built
        deep inside ``FlowMatching3DFAImagePolicy.__init__`` so subclassing
        means duplicating that whole construction. The patch is local,
        reversible (``_USE_MSE_LOSS = False``), and survives state_dict
        load/save (we're only swapping a bound method, not a parameter).
        """
        model = self.policy.model

        def compute_loss_mse(self_model, gt_trajectory, rgb3d, rgb2d, pcd, proprio):
            fixed_inputs = self_model.encode_inputs(rgb3d, rgb2d, pcd, proprio)

            gt_openess = gt_trajectory[..., -1:]
            gt_trajectory = gt_trajectory[..., :-1]
            gt_trajectory = self_model.normalize_pos(gt_trajectory)
            _, traj_len, nhand, _ = gt_trajectory.shape
            gt_trajectory = self_model.convert_rot(
                gt_trajectory.flatten(1, 2)
            ).unflatten(1, (traj_len, nhand))

            total_loss = 0
            for _ in range(self_model._lv2_batch_size):
                noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)
                timesteps = self_model.position_scheduler.sample_noise_step(
                    num_noise=len(noise), device=noise.device,
                )
                pos = self_model.position_scheduler.add_noise(
                    gt_trajectory[..., :3], noise[..., :3], timesteps,
                )
                rot = self_model.rotation_scheduler.add_noise(
                    gt_trajectory[..., 3:], noise[..., 3:], timesteps,
                )
                noisy_trajectory = torch.cat((pos, rot), -1)
                pred = self_model.policy_forward_pass(
                    noisy_trajectory, timesteps, fixed_inputs,
                )
                denoise_target = self_model.position_scheduler.prepare_target(
                    noise, gt_trajectory,
                )  # (B, T, 1, 6) = velocity targets for [pos, rot]
                # Concat with gripper raw value to form a single (B, T, 1, 7)
                # target matching layer_pred's [pos_v, rot_v, openess] layout.
                full_target = torch.cat([denoise_target, gt_openess], dim=-1)
                for layer_pred in pred:
                    # F.mse_loss with reduction='mean' averages over all
                    # (B, T, 1, 7) elements — every dim contributes equally,
                    # mirrors the DiT wrapper's `sq.sum() / (mask.sum() *
                    # action_dim)` normalization (no padding mask yet;
                    # padding is ~3% of action entries on Square).
                    loss = F.mse_loss(layer_pred, full_target, reduction="mean")
                    total_loss = total_loss + loss
            return total_loss / self_model._lv2_batch_size

        model.compute_loss = types.MethodType(compute_loss_mse, model)

    # 3DFA's CLIP encoder expects RGB in [0, 1]; override the base wrapper's
    # [-1, 1] conversion to skip that step. Pointmap is added separately.
    def _build_ab_obs(self, batch, norm_stats):
        device = batch["image"].device
        B, n_cams, Cimg, H, W = batch["image"].shape
        assert n_cams == self.num_cams, (
            f"num_cams mismatch: wrapper={self.num_cams}, batch={n_cams}"
        )

        rgb = batch["image"][:, :, :3]  # (B, n_cams, 3, H, W) in [0, 1]

        state_mean = torch.as_tensor(
            norm_stats["state_mean"], dtype=torch.float32, device=device,
        )
        state_std = torch.as_tensor(
            norm_stats["state_std"], dtype=torch.float32, device=device,
        )
        qpos_raw = batch["qpos"] * state_std + state_mean
        state_raw = torch.cat([batch["eef_xyz"], qpos_raw], dim=-1)
        assert state_raw.shape[-1] == self.state_dim, (
            f"state_dim mismatch: wrapper={self.state_dim}, "
            f"actual={state_raw.shape[-1]} (3 eef + {qpos_raw.shape[-1]} qpos)"
        )

        obs = {}
        for i in range(n_cams):
            obs[f"cam{i}_image"] = rgb[:, i].unsqueeze(1)
            obs[f"cam{i}_pointmap"] = batch["pointmap"][:, i].unsqueeze(1)
        obs["state"] = state_raw.unsqueeze(1)
        return obs

    def forward(self, batch, norm_stats=None):
        if norm_stats is None:
            norm_stats = self._norm_stats
        if "actions" not in batch:
            return self._predict(batch, norm_stats)

        obs = self._build_ab_obs(batch, norm_stats)
        actions_raw = self._actions_raw(batch, norm_stats)
        assert actions_raw.shape[1] == self.horizon, (
            "max_seq_length < horizon — re-export dataset or lower --horizon"
        )

        # 3DFA bypasses ``policy.normalizer`` for actions — its
        # ``compute_loss`` feeds raw deltas straight to the model, which
        # runs ``normalize_pos`` against ``workspace_normalizer`` instead.
        loss = self.policy.compute_loss({"obs": obs, "action": actions_raw})
        return {"loss": loss}
