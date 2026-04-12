"""Per-pixel 3D point maps from depth + camera parameters.

Used to satisfy the ArticuBot RoPE4D DiT policy, which reads each visual
token's (x, y, z) from the patch center of a per-camera point map (see
ArticuBot/diffusion_policy/diffusion_policy/policy/flow_matching_rope4d_dit_image_policy.py:168-176).
"""

import numpy as np


def mujoco_metric_depth(depth_norm, near, far):
    """Convert MuJoCo's normalized depth buffer to metric depth (meters).

    MuJoCo's depth buffer (``env.sim.render(..., depth=True)``) returns values
    in [0, 1] under a non-linear OpenGL projection. Invert to metric depth.
    """
    # See http://www.songho.ca/opengl/gl_projectionmatrix.html
    return near / (1.0 - depth_norm * (1.0 - near / far))


def backproject(depth, K, c2w, invalid_value=0.0, max_depth=None):
    """Back-project a metric depth map to a world-frame point map.

    Args:
        depth: (H, W) metric depth, meters. Positive = forward.
        K: (3, 3) camera intrinsics (OpenCV convention).
        c2w: (4, 4) camera-to-world transform (OpenCV convention: +X right,
             +Y down, +Z forward). If you have GL-convention c2w, pre-convert.
        invalid_value: value written to pixels at/beyond max_depth.
        max_depth: clamp depths greater than this to invalid_value. None = no
                   clamp (only NaN/inf are masked).

    Returns:
        (3, H, W) float32 point map in world frame.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = pts_cam @ R.T + t  # (H, W, 3)

    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    if max_depth is not None:
        invalid = invalid | (depth >= max_depth)
    pts_world[invalid] = invalid_value

    return pts_world.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)


def transform_pointmap(pointmap, T, invalid_mask=None):
    """Apply a 4x4 rigid transform to every valid pixel of a point map.

    Zero (or masked) pixels are preserved as-is so "invalid" pixels remain
    distinguishable from valid points near the origin.

    Args:
        pointmap: (3, H, W) float32.
        T: (4, 4) transform.
        invalid_mask: (H, W) bool. True = pixel is invalid, keep zero.
                      If None, pixels with all-zero xyz are treated as invalid.

    Returns:
        (3, H, W) float32 transformed point map.
    """
    C, H, W = pointmap.shape
    assert C == 3
    pts = pointmap.reshape(3, -1)  # (3, H*W)

    if invalid_mask is None:
        invalid_mask = np.all(pts == 0, axis=0)  # (H*W,)
    else:
        invalid_mask = invalid_mask.reshape(-1)

    R = T[:3, :3].astype(pointmap.dtype)
    t = T[:3, 3].astype(pointmap.dtype)
    out = R @ pts + t[:, None]  # (3, H*W)
    out[:, invalid_mask] = 0.0
    return out.reshape(3, H, W)


def pose_from_pos_ori(pos, ori_3x3):
    """Build a 4x4 homogeneous transform from (pos, 3x3 rotation)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = ori_3x3
    T[:3, 3] = pos
    return T


def invert_pose(T):
    """Closed-form rigid inverse of a 4x4 transform."""
    Ti = np.eye(4, dtype=T.dtype)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def c2w_opencv_to_opengl(c2w_cv):
    """Convert an OpenCV c2w pose to OpenGL convention.

    OpenCV: +X right, +Y down, +Z forward.
    OpenGL: +X right, +Y up, -Z forward.
    Flip y and z axes of the camera frame.
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(c2w_cv.dtype)
    return c2w_cv @ flip


def c2w_opengl_to_opencv(c2w_gl):
    """Inverse of c2w_opencv_to_opengl."""
    return c2w_opencv_to_opengl(c2w_gl)  # involution
