"""Render canonical-view RGB images by fusing multi-camera depth+RGB.

Each input camera's depth + RGB is unprojected to world-frame points, all
clouds are fused, and the fused cloud is z-buffer splatted into one or more
canonical viewpoints. The naive painter-style splat (paint farthest first,
closest overwrites) is sufficient for low-resolution dense maps and matches
the reference at ArticuBot/scripts/project_canonical_views.py.

OpenCV camera convention throughout (+X right, +Y down, +Z forward). Pose
files in CamPose are GL — convert with policy_common.pointmap.c2w_opengl_to_opencv
before passing here.
"""
import numpy as np


def unproject_to_world(depth, rgb, K, c2w):
    """Unproject a (H, W) depth + (H, W, 3) RGB into (N, 3) world points + colors.

    Args:
        depth: (H, W) float, metric meters. Non-positive / non-finite is dropped.
        rgb:   (H, W, 3) uint8.
        K:     (3, 3) OpenCV intrinsics.
        c2w:   (4, 4) OpenCV camera-to-world.
    Returns:
        pts_world: (N, 3) float32 world points.
        colors:    (N, 3) uint8 colors aligned to pts_world.
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)[None, :]
    v = np.arange(H, dtype=np.float32)[:, None]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    z_flat = z.reshape(-1)
    valid = (z_flat > 0) & np.isfinite(z_flat)
    pts_cam = pts_cam[valid]
    colors = colors[valid]

    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    pts_world = pts_cam.astype(np.float32) @ R.T + t
    return pts_world, colors


def render_canonical_view(pts_world, colors, w2c, K, H, W, fill_value=0):
    """Z-buffer splat (N, 3) world points into an (H, W, 3) canonical-view image.

    Args:
        pts_world:  (N, 3) float world points.
        colors:     (N, 3) uint8 colors aligned to pts_world.
        w2c:        (4, 4) OpenCV world-to-camera for the canonical viewpoint.
        K:          (3, 3) intrinsics for the canonical viewpoint.
        H, W:       output image size.
        fill_value: byte written into unwritten pixels (default 0 = black).
    """
    R = w2c[:3, :3].astype(np.float32)
    t = w2c[:3, 3].astype(np.float32)
    pts_cam = pts_world @ R.T + t
    in_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[in_front]
    colors = colors[in_front]
    if pts_cam.shape[0] == 0:
        return np.full((H, W, 3), fill_value, dtype=np.uint8)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = pts_cam[:, 2]
    u = (fx * pts_cam[:, 0] / z + cx).astype(np.int32)
    v = (fy * pts_cam[:, 1] / z + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, colors = u[in_bounds], v[in_bounds], z[in_bounds], colors[in_bounds]
    if u.shape[0] == 0:
        return np.full((H, W, 3), fill_value, dtype=np.uint8)

    # Paint farthest first; closest naturally overwrites.
    order = np.argsort(-z)
    u, v, colors = u[order], v[order], colors[order]
    image = np.full((H, W, 3), fill_value, dtype=np.uint8)
    image[v, u] = colors
    return image


def fuse_and_render(rgbs, depths, Ks, c2ws, canonical_w2cs, canonical_Ks, H, W):
    """Convenience: unproject N input cams, fuse, render to M canonical views.

    Args:
        rgbs:           list of (H, W, 3) uint8 per input cam.
        depths:         list of (H, W) float metric depth per input cam.
        Ks:             list of (3, 3) intrinsics per input cam.
        c2ws:           list of (4, 4) OpenCV c2w per input cam.
        canonical_w2cs: list of (4, 4) OpenCV w2c per canonical view.
        canonical_Ks:   list of (3, 3) intrinsics per canonical view (same length).
        H, W:           output image size.
    Returns:
        canonical_rgbs: list of (H, W, 3) uint8, one per canonical view.
    """
    all_pts, all_colors = [], []
    for rgb, depth, K, c2w in zip(rgbs, depths, Ks, c2ws):
        pts, cols = unproject_to_world(depth, rgb, K, c2w)
        all_pts.append(pts)
        all_colors.append(cols)
    if all_pts:
        pts_world = np.concatenate(all_pts, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        pts_world = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)

    return [
        render_canonical_view(pts_world, colors, w2c, K, H, W)
        for w2c, K in zip(canonical_w2cs, canonical_Ks)
    ]
