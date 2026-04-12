"""Random crop that can be applied identically to multiple tensors.

Needed for RoPE4D DiT: the RGB image, its associated pointmap, and the
Plucker embedding must be cropped with the SAME window so per-patch 3D
coordinates still correspond to RGB tokens after cropping.
"""

import random
import numpy as np
import torch


class PairedRandomCrop:
    """Random crop a (C, H, W) tensor from src_size to dst_size.

    Samples (top, left) on each call to ``sample_offsets()``; ``__call__``
    applies the most recently sampled offsets. No resize after crop — the
    cropped region is returned at dst_size directly. This matches
    ArticuBot's policy.crop_shape=(224, 224) with patch_size=14.

    Typical usage (CamPose dataset):

        crop = PairedRandomCrop(src=256, dst=224)
        t, l = crop.sample_offsets()
        rgb_c      = crop(rgb)
        pointmap_c = crop(pointmap)
        plucker_c  = crop(plucker)
        K_c        = adjust_intrinsic(K, t, l)

    Thread/worker-safety: one instance per dataset sample (constructed fresh
    or reused single-threaded — the current CamPose dataloaders use
    num_workers=0).
    """

    def __init__(self, src=256, dst=224):
        assert dst <= src
        self.src = src
        self.dst = dst
        self._top = 0
        self._left = 0

    def sample_offsets(self):
        """Pick a new random (top, left) and return it."""
        self._top = random.randint(0, self.src - self.dst)
        self._left = random.randint(0, self.src - self.dst)
        return self._top, self._left

    def center_offsets(self):
        """Set offsets to the center crop."""
        self._top = (self.src - self.dst) // 2
        self._left = (self.src - self.dst) // 2
        return self._top, self._left

    def offsets(self):
        return self._top, self._left

    def __call__(self, tensor):
        """Apply the most recently sampled offsets to a (C, H, W) tensor.

        Accepts a torch.Tensor or numpy array. Returns the same type.
        """
        t, l = self._top, self._left
        if isinstance(tensor, torch.Tensor):
            assert tensor.dim() == 3, f"expected (C,H,W), got {tensor.shape}"
            return tensor[:, t:t + self.dst, l:l + self.dst]
        else:
            assert tensor.ndim == 3, f"expected (C,H,W), got {tensor.shape}"
            return tensor[:, t:t + self.dst, l:l + self.dst]


def adjust_intrinsic(K, top, left):
    """Translate principal point to account for a top-left crop.

    Args:
        K: (3, 3) intrinsic matrix.
        top: pixels cropped from the top.
        left: pixels cropped from the left.

    Returns:
        (3, 3) intrinsic matrix for the cropped image.
    """
    K_out = np.array(K, dtype=np.float32, copy=True)
    K_out[0, 2] -= left
    K_out[1, 2] -= top
    return K_out
