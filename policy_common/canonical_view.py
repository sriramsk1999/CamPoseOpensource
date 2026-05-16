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
import torch


def unproject_to_world(depth, rgb, K, c2w):
    """Unproject a (H, W) depth + (H, W, 3) RGB into (N, 3) world points + colors.

    Numpy reference (kept for parity / testing). The training/eval pipeline
    uses ``fuse_and_render`` which dispatches to the GPU path.
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

    Numpy reference path — single threaded ``argsort`` + scatter is the
    bottleneck for the canonical baselines; ``fuse_and_render`` runs the
    vectorized GPU version below instead.
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

    order = np.argsort(-z)
    u, v, colors = u[order], v[order], colors[order]
    image = np.full((H, W, 3), fill_value, dtype=np.uint8)
    image[v, u] = colors
    return image


# ---------------------------------------------------------------------------
# GPU-accelerated fuse_and_render
# ---------------------------------------------------------------------------
# The CPU path's argsort + scatter over ~130k points per canonical view is the
# main bottleneck for the canonical-view baselines (numpy argsort + fancy
# indexing on a single CPU thread). All ops here are O(N) GPU kernels except
# torch.argsort, which is ~100x faster on CUDA than numpy on CPU for N≈1e5.

_GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_torch(arr, dtype, device):
    """np.ndarray | torch.Tensor → torch.Tensor on (device, dtype). Cheap copy if needed."""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _unproject_to_world_gpu(depth, rgb, K, c2w):
    """Torch-on-GPU version of unproject_to_world.

    depth: (H, W) float32
    rgb:   (H, W, 3) uint8
    K, c2w: float32 tensors on the same device
    Returns (pts_world (N,3) float32, colors (N,3) uint8) on the same device.
    """
    H, W = depth.shape
    device = depth.device
    u = torch.arange(W, device=device, dtype=depth.dtype)[None, :]
    v = torch.arange(H, device=device, dtype=depth.dtype)[:, None]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    z_flat = z.reshape(-1)
    valid = (z_flat > 0) & torch.isfinite(z_flat)
    pts_cam = pts_cam[valid]
    colors = colors[valid]

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = pts_cam @ R.T + t
    return pts_world, colors


def _render_canonical_view_gpu(pts_world, colors, w2c, K, H, W, fill_value=0):
    """Torch-on-GPU z-buffer splat. Returns (H, W, 3) uint8 on the same device."""
    device = pts_world.device
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = pts_world @ R.T + t

    z = pts_cam[:, 2]
    in_front = z > 0
    pts_cam = pts_cam[in_front]
    colors = colors[in_front]
    z = z[in_front]

    if pts_cam.shape[0] == 0:
        return torch.full((H, W, 3), fill_value, dtype=torch.uint8, device=device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * pts_cam[:, 0] / z + cx).long()
    v = (fy * pts_cam[:, 1] / z + cy).long()

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, colors = u[in_bounds], v[in_bounds], z[in_bounds], colors[in_bounds]
    if u.shape[0] == 0:
        return torch.full((H, W, 3), fill_value, dtype=torch.uint8, device=device)

    # Sort farthest-first so the closest write wins (matches the CPU painter
    # algorithm). torch.argsort on CUDA over ~130k items is ~1ms.
    order = torch.argsort(z, descending=True)
    u, v, colors = u[order], v[order], colors[order]

    image = torch.full((H, W, 3), fill_value, dtype=torch.uint8, device=device)
    image[v, u] = colors
    return image


def fuse_and_render(rgbs, depths, Ks, c2ws, canonical_w2cs, canonical_Ks, H, W):
    """Convenience: unproject N input cams, fuse, render to M canonical views.

    GPU-accelerated path. Inputs may be numpy arrays or torch tensors (the
    caller in ``policy_robosuite/utils.py`` and ``policy_robosuite/eval.py``
    passes numpy from mujoco render). Returns a list of numpy ``(H, W, 3)``
    uint8 arrays — keeps the existing call-site shape so downstream
    PIL/torch-from-numpy paths don't have to change.
    """
    device = torch.device(_GPU_DEVICE)

    all_pts, all_colors = [], []
    for rgb, depth, K, c2w in zip(rgbs, depths, Ks, c2ws):
        depth_t = _to_torch(depth, torch.float32, device)
        rgb_t = _to_torch(rgb, torch.uint8, device)
        K_t = _to_torch(K, torch.float32, device)
        c2w_t = _to_torch(c2w, torch.float32, device)
        pts, cols = _unproject_to_world_gpu(depth_t, rgb_t, K_t, c2w_t)
        all_pts.append(pts)
        all_colors.append(cols)

    if all_pts:
        pts_world = torch.cat(all_pts, dim=0)
        colors = torch.cat(all_colors, dim=0)
    else:
        pts_world = torch.zeros((0, 3), dtype=torch.float32, device=device)
        colors = torch.zeros((0, 3), dtype=torch.uint8, device=device)

    out = []
    for w2c, K in zip(canonical_w2cs, canonical_Ks):
        w2c_t = _to_torch(w2c, torch.float32, device)
        K_t = _to_torch(K, torch.float32, device)
        img = _render_canonical_view_gpu(pts_world, colors, w2c_t, K_t, H, W)
        out.append(img.cpu().numpy())
    return out
