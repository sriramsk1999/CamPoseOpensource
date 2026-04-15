"""ACT policy variant with a DINO Cross-View visual backbone.

Mirrors ``models.act`` but swaps the CNN backbone for ArticuBot's
``DinoCrossViewTokenEncoder``. Imported via sidecar (no vendoring), matching
the pattern in ``campose_wrappers/articubot_dit.py``.
"""
import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat, rearrange

from .act import kl_divergence
from .detr_vae import reparametrize, get_sinusoid_encoding_table
from .transformer import (
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class PluckerViT(nn.Module):
    """Small ViT trained from scratch over 6-channel Plucker ray maps.

    Patchifies with the same 14x14 stride as DINOv2 so output tokens align
    1:1 with ``DinoCrossViewTokenEncoder`` patch tokens, which makes a
    token-wise late concat trivial.
    """

    def __init__(
        self,
        crop_shape=(224, 224),
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        crop_h, crop_w = crop_shape
        assert crop_h % patch_size == 0 and crop_w % patch_size == 0
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (crop_h // patch_size) * (crop_w // patch_size)

        self.patch_embed = nn.Conv2d(
            6, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, plucker):
        """plucker: (B, num_cam, 6, H, W) -> (B, num_cam, num_patches, embed_dim)"""
        b, s = plucker.shape[:2]
        x = rearrange(plucker, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        x = rearrange(x, "bs d h w -> bs (h w) d")
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return rearrange(x, "(b s) n d -> b s n d", b=b, s=s)


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


class BackboneDinoCrossView(nn.Module):
    """DINO Cross-View backbone wrapping ``DinoCrossViewTokenEncoder``.

    Forward signature extends the existing CNN backbones to also accept
    per-camera extrinsics (c2w) and intrinsics, so the underlying
    ``CameraEnc`` can inject geometry-aware camera tokens.
    """

    def __init__(
        self,
        hidden_dim: int,
        backbone: str = "vitb",
        pretrained: bool = True,
        crop_shape=(224, 224),
        alt_start: int = 6,
        qknorm_start: int = 6,
        rope_start: int = 6,
        cat_token: bool = True,
        include_camera_enc: bool = True,
        max_cams: int = 4,
        use_plucker: bool = False,
        plucker_vit_embed_dim: int = 384,
        plucker_vit_depth: int = 4,
        plucker_vit_num_heads: int = 6,
        plucker_vit_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        _ensure_articubot_on_path()
        from diffusion_policy.model.flow_matching.dino_cross_view_encoder import (
            DinoCrossViewTokenEncoder,
        )

        self.encoder = DinoCrossViewTokenEncoder(
            cam_keys=[],
            n_obs_steps=1,
            embed_dim=hidden_dim,
            crop_shape=crop_shape,
            in_channels=3,
            image_size=crop_shape[0],
            backbone=backbone,
            pretrained=pretrained,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            include_camera_enc=include_camera_enc,
        )

        self.crop_h, self.crop_w = crop_shape
        self.num_tokens_per_cam = self.encoder.num_tokens
        self.num_channels = hidden_dim
        self.include_camera_enc = include_camera_enc
        self.use_plucker = use_plucker

        if use_plucker:
            self.plucker_vit = PluckerViT(
                crop_shape=crop_shape,
                patch_size=14,
                embed_dim=plucker_vit_embed_dim,
                depth=plucker_vit_depth,
                num_heads=plucker_vit_num_heads,
                mlp_ratio=plucker_vit_mlp_ratio,
            )
            assert self.plucker_vit.num_patches == self.num_tokens_per_cam, (
                f"PluckerViT patch count {self.plucker_vit.num_patches} must match "
                f"DINO token count {self.num_tokens_per_cam}"
            )
            self.fused_projector = nn.Linear(
                self.encoder.token_dim + plucker_vit_embed_dim, hidden_dim,
            )

        self.pos_embed = nn.Embedding(max_cams * self.num_tokens_per_cam, hidden_dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, images, extrinsics=None, intrinsics=None):
        """
        Args:
            images:     (B, num_cam, C, H, W) — RGB (+optional Plucker) in [0, 1].
                         Channels [0:3] are RGB, [3:9] Plucker rays when
                         ``use_plucker`` is set.
            extrinsics: (B, num_cam, 4, 4) c2w (optional).
            intrinsics: (B, num_cam, 3, 3)       (optional).
        Returns:
            features  : (B, num_cam * num_tokens_per_cam, hidden_dim)
            pos_embed : (B, num_cam * num_tokens_per_cam, hidden_dim)
        """
        b, s = images.shape[:2]
        rgb = images[:, :, :3]
        assert rgb.shape[-2:] == (self.crop_h, self.crop_w), (
            f"act_dino expects pre-cropped ({self.crop_h}x{self.crop_w}) images, "
            f"got {tuple(rgb.shape[-2:])}"
        )

        # CameraEnc internally does affine_inverse(extrinsics), i.e. it expects w2c.
        # The CamPose dataloader supplies c2w, so invert here. Skip the work
        # entirely when CameraEnc is disabled — the learned per-view camera
        # token handles the None case inside forward_features.
        w2c = None
        intr = None
        if self.include_camera_enc and extrinsics is not None and intrinsics is not None:
            w2c = torch.linalg.inv(extrinsics)
            intr = intrinsics

        tokens = self.encoder(rgb, extrinsics=w2c, intrinsics=intr)

        if self.use_plucker:
            assert images.size(2) == 9, (
                f"use_plucker expects 9-channel images (3 RGB + 6 Plucker), "
                f"got {images.size(2)}"
            )
            plucker = images[:, :, 3:9]
            plucker_tokens = self.plucker_vit(plucker)  # (B, S, N_tok, plk_dim)
            fused = torch.cat([tokens, plucker_tokens], dim=-1)
            tokens = self.fused_projector(fused)  # (B, S, N_tok, hidden_dim)
        else:
            tokens = self.encoder.projector(tokens)  # (B, S, N_tok, hidden_dim)

        features = rearrange(tokens, "b s n d -> b (s n) d")
        total_len = features.shape[1]
        pos_embed = repeat(
            self.pos_embed.weight[:total_len], "n d -> b n d", b=b,
        )
        return features, pos_embed


class DETRVAEDino(nn.Module):
    """DETR-VAE decoder that consumes DINO Cross-View tokens.

    Mirrors ``detr_vae.DETRVAE`` but routes per-camera extrinsics and
    intrinsics to the backbone (which owns the camera encoder), so the
    legacy 2x4x4 ``cam_extrinsics`` side-input is no longer needed.
    """

    def __init__(self, backbone, transformer, encoder, action_dim, obs_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        self.backbone = backbone
        self.input_proj_robot_state = nn.Linear(obs_dim, hidden_dim)
        self.input_embed = nn.Embedding(2, hidden_dim)  # [proprio, latent]

        self.latent_dim = 32
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(obs_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + chunk_size, hidden_dim)
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

    def forward(self, data):
        qpos = data["qpos"]
        image = data["image"]
        actions = data.get("actions")
        is_pad = data.get("is_pad")
        cam_extrinsics_full = data.get("cam_extrinsics_full")  # (B, num_cam, 4, 4) c2w
        cam_intrinsics_full = data.get("cam_intrinsics_full")  # (B, num_cam, 3, 3)

        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos)
            qpos_embed = rearrange(qpos_embed, "bs d -> bs 1 d")
            cls_input = torch.zeros(
                [bs, 1, self.hidden_dim], dtype=torch.float32, device=qpos.device
            )
            encoder_input = torch.cat([cls_input, qpos_embed, action_embed], axis=1)
            cls_joint_is_pad = torch.zeros(bs, 2, dtype=torch.bool, device=qpos.device)
            is_pad_full = torch.cat([cls_joint_is_pad, is_pad], axis=1)
            pos_embed_cvae = self.pos_table.clone().detach()
            pos_embed_cvae = repeat(pos_embed_cvae, "1 seq d -> bs seq d", bs=bs)
            encoder_output = self.encoder(
                encoder_input,
                src_key_padding_mask=is_pad_full,
                pos=pos_embed_cvae,
            )
            encoder_output = encoder_output[:, 0, :]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(
                [bs, self.latent_dim], dtype=torch.float32, device=qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        camera_features, camera_pos_embed = self.backbone(
            image,
            extrinsics=cam_extrinsics_full,
            intrinsics=cam_intrinsics_full,
        )

        proprio_input = self.input_proj_robot_state(qpos)
        proprio_input = rearrange(proprio_input, "bs d -> bs 1 d")
        latent_input = rearrange(latent_input, "bs d -> bs 1 d")

        src = torch.cat([camera_features, proprio_input, latent_input], dim=1)
        proprio_latent_pos = repeat(self.input_embed.weight, "s d -> b s d", b=bs)
        pos_embed = torch.cat([camera_pos_embed, proprio_latent_pos], dim=1)

        query_embed = repeat(self.query_embed.weight, "c d -> b c d", b=bs)
        hs = self.transformer(
            src=src, mask=None, query_embed=query_embed, pos_embed=pos_embed,
        )[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


def build_dino(args):
    backbone = BackboneDinoCrossView(
        hidden_dim=args.hidden_dim,
        backbone=getattr(args, "dino_backbone", "vitb"),
        pretrained=getattr(args, "dino_pretrained", True),
        crop_shape=tuple(getattr(args, "dino_crop_shape", (224, 224))),
        alt_start=getattr(args, "dino_alt_start", 6),
        qknorm_start=getattr(args, "dino_qknorm_start", 6),
        rope_start=getattr(args, "dino_rope_start", 6),
        cat_token=getattr(args, "dino_cat_token", True),
        include_camera_enc=getattr(args, "dino_camera_enc", True),
        max_cams=max(2, getattr(args, "num_side_cam", 2)),
        use_plucker=getattr(args, "use_plucker", False),
        plucker_vit_embed_dim=getattr(args, "plucker_vit_embed_dim", 384),
        plucker_vit_depth=getattr(args, "plucker_vit_depth", 4),
        plucker_vit_num_heads=getattr(args, "plucker_vit_num_heads", 6),
        plucker_vit_mlp_ratio=getattr(args, "plucker_vit_mlp_ratio", 4.0),
    )

    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        ffn_dim=args.ffn_dim,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        norm_cls=nn.LayerNorm,
        activation=args.activation,
    )

    encoder_layer = TransformerEncoderLayer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        activation=args.activation,
        normalize_before=args.pre_norm,
        norm_cls=nn.LayerNorm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    encoder = TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)

    model = DETRVAEDino(
        backbone,
        transformer,
        encoder,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        chunk_size=args.chunk_size,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    return model, optimizer


class ACTDinoPolicy(nn.Module):
    """ACT policy with DINO Cross-View backbone."""

    def __init__(self, args):
        super().__init__()
        model, optimizer = build_dino(args)
        self.model = model
        self.optimizer = optimizer

        self.kl_weight = args.kl_weight
        self.prob_drop_proprio = args.prob_drop_proprio
        self.use_cam_pose = args.use_cam_pose
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, data_dict):
        qpos = data_dict["qpos"]
        image = data_dict["image"]
        actions = data_dict.get("actions", None)
        is_pad = data_dict.get("is_pad", None)

        # DINO ViT applies ImageNet normalization internally; feed [0,1] RGB.
        assert image.size(2) in (3, 9)

        model_data = dict(data_dict)
        model_data["qpos"] = qpos
        model_data["image"] = image
        model_data["actions"] = actions
        model_data["is_pad"] = is_pad

        if actions is not None:  # training
            actions = actions[:, : self.model.chunk_size]
            is_pad = is_pad[:, : self.model.chunk_size]
            model_data["actions"] = actions
            model_data["is_pad"] = is_pad

            a_hat, is_pad_hat, (mu, logvar) = self.model(model_data)
            total_kld, _, _ = kl_divergence(mu, logvar)
            loss_dict = {}
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference
            a_hat, _, (_, _) = self.model(model_data)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
