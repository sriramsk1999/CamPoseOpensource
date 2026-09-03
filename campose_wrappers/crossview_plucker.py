"""Cross-view DINO encoder with a sibling Plucker ViT stream.

The Plucker-raymap arm of Table 1. Cross-view analogue of ArticuBot's
single-view ``DINOv2PluckerTokenEncoder``: 6-channel Plucker ray maps are
encoded by a small ViT, projected to the backbone's token dim, concatenated
channel-wise with the cross-view DINO tokens, then projected down to the
policy ``embed_dim`` (arXiv:2510.02268: "for pretrained encoders, encode
Plucker with a small conv net to the same dim as the image latent, then
concatenate channel-wise"). Channel concat rather than residual add avoids
the additive-collapse failure mode where the plucker stream decays to ~0.

Why not the existing ``dinov2_plucker``: it is single-view, so using it for
this row would vary the visual backbone *and* the pose pathway at once —
exactly the confound Table 1 exists to avoid. Every Table-1 arm holds the
cross-view DINOv2 backbone fixed.

Why this lives in CamPose rather than upstream: ArticuBot is owned by a
coauthor right now, so nothing here modifies it. That constrains the design
in two ways, both handled below:

  - We cannot add ourselves to ``build_visual_encoder``'s registry (it is a
    local dict inside that function), so ``install_plucker_encoder`` swaps
    the encoder onto an already-built policy. The policy only ever calls
    ``visual_encoder.encode(nobs)`` and counts its parameters, so this is safe.
  - We cannot add a fusion hook inside upstream's ``encode``. Rather than
    vendoring a copy of that ~60-line method — which would silently drift as
    the coauthor edits it — we inherit it verbatim and do the fusion inside a
    replacement ``projector`` module, handing it the plucker tokens just
    before the call. Any upstream change that stops routing tokens through
    ``self.projector`` trips a hard assert instead of silently dropping the
    geometry stream.

The plucker branch itself reuses upstream's ``PluckerViT`` unchanged, so the
single-view and cross-view plucker baselines get an identical geometry stream.
"""

import torch
import torch.nn as nn


class _PluckerFusionProjector(nn.Module):
    """Drop-in replacement for the encoder's ``projector`` linear.

    Concatenates the plucker tokens staged in ``pending`` onto the incoming
    DINO tokens, then projects the fused width down to ``embed_dim``.

    ``pending`` is set by :meth:`_CrossViewPluckerMixin.encode` immediately
    before it delegates to upstream's ``encode``, and cleared on the way out.
    Passing it this way (rather than as an argument) is what lets us inherit
    upstream's ``encode`` untouched.
    """

    def __init__(self, token_dim: int, embed_dim: int):
        super().__init__()
        self.token_dim = token_dim
        self.linear = nn.Linear(2 * token_dim, embed_dim)
        # Zero the plucker half so the encoder starts exactly at the RGB-only
        # baseline and lifts the geometry weights as they earn their place,
        # instead of perturbing the pretrained features at step 0. The RGB
        # half mirrors nn.Linear's own default init.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.linear.weight[:, :token_dim], a=5 ** 0.5)
            self.linear.weight[:, token_dim:].zero_()
            self.linear.bias.zero_()
        self.pending = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        plk, self.pending = self.pending, None
        assert plk is not None, (
            "plucker tokens were not staged before the projector ran — "
            "upstream DinoCrossViewTokenEncoder.encode no longer routes its "
            "tokens through self.projector exactly once. Re-check "
            "campose_wrappers/crossview_plucker.py against it."
        )
        assert tokens.shape[:-1] == plk.shape[:-1], (
            f"plucker tokens {tuple(plk.shape)} do not line up with DINO "
            f"tokens {tuple(tokens.shape)}"
        )
        return self.linear(torch.cat([tokens, plk.to(tokens.dtype)], dim=-1))


# The concrete encoder subclasses an ArticuBot class, so it can only be defined
# once the sidecar is importable. Built on first use and cached.
_ENCODER_CLS = None


def _encoder_cls():
    global _ENCODER_CLS
    if _ENCODER_CLS is not None:
        return _ENCODER_CLS

    from diffusion_policy.model.flow_matching.dino_cross_view_encoder import (
        DinoCrossViewTokenEncoder,
    )
    from diffusion_policy.model.flow_matching.dinov2_plucker_encoder import PluckerViT

    class DinoCrossViewPluckerTokenEncoder(DinoCrossViewTokenEncoder):
        """Cross-view DINO + sibling Plucker ViT, fused by channel concat."""

        def __init__(self, *, cam_keys=None, plucker_vit_cfg=None, **kwargs):
            all_keys = list(cam_keys or [])
            rgb_keys = sorted([k for k in all_keys if "plucker" not in k])
            plucker_keys = sorted([k for k in all_keys if "plucker" in k])
            assert rgb_keys, f"need RGB keys, got cam_keys={all_keys}"
            assert plucker_keys, f"need plucker keys, got cam_keys={all_keys}"
            assert len(rgb_keys) == len(plucker_keys), (
                f"need one plucker per RGB cam, got {len(rgb_keys)} RGB and "
                f"{len(plucker_keys)} plucker"
            )
            # Both lists are sorted, so index i must name the same camera in
            # each — the fusion pairs them positionally.
            for rgb_k, plk_k in zip(rgb_keys, plucker_keys):
                assert rgb_k.split("_")[0] == plk_k.split("_")[0], (
                    f"cam key mismatch: {rgb_k!r} vs {plk_k!r}"
                )

            # Only the RGB keys reach the parent, so it sizes its camera
            # embeddings and derives its extrinsic/intrinsic keys off the real
            # camera count.
            super().__init__(cam_keys=rgb_keys, **kwargs)
            self.plucker_keys = plucker_keys

            plk_cfg = dict(plucker_vit_cfg or {})
            plk_cfg.setdefault("patch_size", 14)
            plk_cfg.setdefault("embed_dim", 384)
            plk_cfg.setdefault("depth", 4)
            plk_cfg.setdefault("num_heads", 6)
            plk_cfg.setdefault("mlp_ratio", 4.0)

            self.plucker_vit = PluckerViT(
                crop_shape=(self._crop_h, self._crop_w), **plk_cfg,
            )
            # Plucker tokens enter the fusion at the backbone's per-token
            # width. Note cross-view token_dim is 2*vit_embed_dim under
            # cat_token=True, so this is wider than the single-view encoder's.
            self.plk_proj = nn.Linear(plk_cfg["embed_dim"], self._token_dim)
            self.projector = _PluckerFusionProjector(self._token_dim, self.embed_dim)

            print(f"[DinoCrossViewPluckerTokenEncoder] rgb_keys={rgb_keys}, "
                  f"plucker_keys={plucker_keys}, plucker_vit_cfg={plk_cfg}, "
                  f"fusion=channel_concat+project")

        def _plucker_tokens(self, nobs):
            """-> (B*To, n_cams, N_tok, token_dim), matching the parent's
            token layout and camera ordering."""
            plk = torch.stack(
                [nobs[k][:, :self.n_obs_steps] for k in self.plucker_keys], dim=2,
            )  # (B, To, n_cams, 6, H, W)
            B, To, n_cams = plk.shape[:3]
            assert plk.shape[-2:] == (self._crop_h, self._crop_w), (
                f"plucker maps must be pre-cropped to "
                f"({self._crop_h}, {self._crop_w}), got {tuple(plk.shape[-2:])}"
            )
            tok = self.plucker_vit(
                plk.reshape(B * To * n_cams, 6, self._crop_h, self._crop_w)
            )
            tok = self.plk_proj(tok)                    # -> token_dim
            assert tok.shape[1] == self.num_tokens, (
                f"plucker token count {tok.shape[1]} must equal the DINO patch "
                f"count {self.num_tokens}; check patch_size vs the backbone."
            )
            return tok.reshape(B * To, n_cams, self.num_tokens, self._token_dim)

        def encode(self, nobs: dict) -> torch.Tensor:
            # Stage the geometry stream, then run upstream's encode verbatim;
            # the replacement projector consumes it mid-pipeline.
            self.projector.pending = self._plucker_tokens(nobs)
            try:
                return super().encode(nobs)
            finally:
                self.projector.pending = None

    _ENCODER_CLS = DinoCrossViewPluckerTokenEncoder
    return _ENCODER_CLS


def install_plucker_encoder(policy, *, cam_keys, encoder_cfg, n_obs_steps,
                            embed_dim, crop_shape, image_size):
    """Swap ``policy.visual_encoder`` for the plucker variant, in place.

    ``cam_keys`` carries both the RGB and plucker shape_meta keys; the encoder
    splits them. The kwargs mirror what ``FlowMatchingDiTImagePolicy`` injects
    into every encoder it builds, so the replacement is configured identically
    to the one it discards.
    """
    policy.visual_encoder = _encoder_cls()(
        cam_keys=list(cam_keys),
        n_obs_steps=n_obs_steps,
        embed_dim=embed_dim,
        crop_shape=crop_shape,
        in_channels=3,
        image_size=image_size,
        **encoder_cfg,
    )
    return policy
