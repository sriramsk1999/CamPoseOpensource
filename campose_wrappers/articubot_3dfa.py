"""3DFA (3D FlowMatch Actor) sidecar wrapper.

Trains ArticuBot's ``FlowMatching3DFAImagePolicy`` (which itself wraps
upstream 3DFA's ``DenoiseActor3D``, see
https://github.com/nickgkan/3d_flowmatch_actor) with the CamPose train
loops. Mirrors the pattern in ``campose_wrappers.articubot_dit`` — sidecar
import via ``ARTICUBOT_DP``, passthrough normalizer, CamPose batch →
ArticuBot obs adapter.

Configured for CamPose's 7-dim ``eef_delta`` action via
``rotation_format='euler'`` (3 xyz + 3 euler + 1 grip = 7).
"""
import torch
import torch.nn as nn

from campose_wrappers.articubot_dit import _ArticubotWrapperBase


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
