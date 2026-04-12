import h5py
import torch
import os
import sys
import numpy as np
import random
import re
import math
import glob
import json
import einops
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from cam_embedding import PluckerEmbedder
from mani_skill.trajectory.utils import dict_to_list_of_dicts

from eval import to_mp4

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _REPO_ROOT)
from policy_common.paired_crop import PairedRandomCrop, adjust_intrinsic

# --- Utility Functions ---

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_dict_mean(dict_list):
    """Compute the mean value for each key in a list of dictionaries."""
    if len(dict_list) == 0:
        return {}
    
    mean_dict = {}
    for key in dict_list[0].keys():
        if not isinstance(dict_list[0][key], torch.Tensor):
            continue  # Skip non-tensor values
        mean_dict[key] = torch.stack([d[key] for d in dict_list]).mean()
    return mean_dict

def detach_dict(dictionary):
    """Detach all tensors in a dictionary."""
    result = {}
    for k, v in dictionary.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach()
        else:
            result[k] = v
    return result

def cleanup_ckpt(ckpt_dir, keep=1):
    """Keep only the latest N checkpoints."""
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if len(ckpts) <= keep:
        return
    
    epoch_nums = []
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+).pth", ckpt)
        if match:
            epoch_nums.append((int(match.group(1)), ckpt))
    
    epoch_nums.sort(reverse=True)
    
    for _, ckpt in epoch_nums[keep:]:
        os.remove(ckpt)

def get_last_ckpt(ckpt_dir):
    """Get the latest checkpoint in the directory."""
    if not os.path.exists(ckpt_dir):
        return None
    
    ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    if not ckpts:
        return None
    
    latest_epoch = -1
    latest_ckpt = None
    for ckpt in ckpts:
        match = re.search(r"epoch_(\d+).pth", ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt
    
    return latest_ckpt

def cosine_schedule(optimizer, total_steps, eta_min=0.0):
    """Cosine learning rate schedule."""
    def lr_lambda(step):
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * step / total_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min=0.0):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def constant_schedule(optimizer):
    """Constant learning rate schedule (no decay)."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


def get_norm_stats(dataset_path, num_demos, policy_class):
    """
    Compute normalization statistics for ManiSkill trajectories.
    """
    all_states_data = []
    all_action_data = []

    h5_path = os.path.join(dataset_path, 'trajectory.h5')
    json_path = os.path.join(dataset_path, 'trajectory.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    with h5py.File(h5_path, 'r') as f:
        episode_ids = [ep['episode_id'] for ep in json_data['episodes']]
        num_demos_to_use = min(num_demos, len(episode_ids))
        print(f"Computing ManiSkill normalization statistics using {num_demos_to_use} demonstrations from {dataset_path}...")
        for i in range(num_demos_to_use):
            traj_key = f"traj_{episode_ids[i]}"
            if traj_key not in f:
                continue
            states = dict_to_list_of_dicts(f[traj_key]["env_states"])  # list of dicts
            actions = f[traj_key]["actions"][:].astype(np.float32)
            # Extract robot joint positions (Panda indices 13:22)
            for key in ("panda", "panda_wristcam", "panda_stick"):
                if key in states[0]["articulations"]:
                    qpos_list = [s["articulations"][key][13:22] for s in states]
                    break
            robot_qpos = np.stack(qpos_list, axis=0)
            all_states_data.append(robot_qpos)
            all_action_data.append(actions)

    states_array = np.concatenate(all_states_data, axis=0)
    actions_array = np.concatenate(all_action_data, axis=0)

    if policy_class == 'dp':
        s_min = states_array.min(axis=0)
        s_max = states_array.max(axis=0)
        a_min = actions_array.min(axis=0)
        a_max = actions_array.max(axis=0)
        state_std = np.maximum((s_max - s_min) / 2.0, 1e-6)
        state_mean = (s_min + s_max) / 2.0
        action_std = np.maximum((a_max - a_min) / 2.0, 1e-6)
        action_mean = (a_min + a_max) / 2.0
    elif policy_class == 'pi0':
        a_q1 = np.quantile(actions_array, 0.01, axis=0)
        a_q99 = np.quantile(actions_array, 0.99, axis=0)
        action_std = np.maximum((a_q99 - a_q1) / 2.0, 1e-6)
        action_mean = (a_q1 + a_q99) / 2.0
        state_mean = np.mean(states_array, axis=0)
        state_std = np.std(states_array, axis=0)
        state_std = np.clip(state_std, 1e-4, np.inf)
    else:
        state_mean = np.mean(states_array, axis=0)
        state_std = np.std(states_array, axis=0)
        state_std = np.clip(state_std, 1e-4, np.inf)
        action_mean = np.mean(actions_array, axis=0)
        action_std = np.std(actions_array, axis=0)
        action_std = np.clip(action_std, 1e-4, np.inf)

    stats = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std
    }
    print(f"State Mean shape: {stats['state_mean'].shape}, Action Mean shape: {stats['action_mean'].shape}")
    return stats

class RGBJitter(object):
    """
    Apply color jittering to the RGB channels (first 3 channels) of the image.
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.rgb_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast, 
            saturation=saturation, 
            hue=hue
        )
    
    def __call__(self, img):
        assert img.dim() == 3
        img[:3] = self.rgb_jitter(img[:3])
        return img

class RandomCrop(object):
    def __init__(self, min_side=224, max_side=256, output_size=256):
        self.min_side = min_side
        self.max_side = max_side
        self.output_size = output_size

    def __call__(self, image):
        assert image.dim() == 3
        C, H, W = image.shape
        assert H >= self.min_side and W >= self.min_side
        assert H <= self.max_side and W <= self.max_side

        crop_size = random.randint(self.min_side, self.max_side)
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        cropped = image[:, top:top+crop_size, left:left+crop_size]
        resized = TF.resize(cropped, [self.output_size, self.output_size], interpolation=T.InterpolationMode.NEAREST)
        return resized

def save_image_batch_as_mp4(image_batch, save_path):
    rgb_batch = image_batch[:, :3, :, :].cpu().numpy()  # [B, 3, H, W]
    image_list = []
    for i in range(rgb_batch.shape[0]):
        img = rgb_batch[i]  # [3, H, W]
        img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
        img = (img * 255).astype(np.uint8)
        image_list.append(img)
    to_mp4(save_path + '.mp4', image_list)

# --- Simple Image Save Utility ---

def save_first_image_as_png(image_batch, save_path):
    """
    Save the first image of a batch (expects [B, C, H, W]) as a PNG using only the first 3 channels.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = image_batch[0, :3].detach().float().cpu().clamp(0, 1)
    TF.to_pil_image(img).save(save_path)

# --- Dataset Class ---

class EpisodicDataset(Dataset):
    """
    Dataset for ManiSkill trajectories with dynamic camera selection and Plücker embeddings.
    """
    def __init__(self, demo_indices, norm_stats, args,
                 max_seq_length=None, transform="id", env=None):
        super().__init__()
        self.demo_indices = demo_indices
        self.norm_stats = norm_stats
        self.args = args
        self.image_size = 256
        self.use_plucker = args.use_plucker
        self.use_pointmaps = args.use_pointmaps
        self._paired_crop = PairedRandomCrop(src=self.image_size, dst=224)
        self.env = env

        if self.use_plucker:
            self.plucker_embedder = PluckerEmbedder(img_size=self.image_size, device='cuda')
        else:
            self.plucker_embedder = None

        # Load demonstrations
        h5_path = os.path.join(args.dataset_path, 'trajectory.h5')
        json_path = os.path.join(args.dataset_path, 'trajectory.json')

        self.demo_states = []
        self.demo_actions = []
        self.demo_lengths = []

        with open(json_path, 'r') as f:
            json_data = json.load(f)
        with h5py.File(h5_path, 'r') as f:
            for idx in self.demo_indices:
                traj_key = f'traj_{idx}'
                if traj_key not in f:
                    continue
                states = dict_to_list_of_dicts(f[traj_key]["env_states"])  # list of dicts
                actions = f[traj_key]["actions"][:].astype(np.float32)
                self.demo_states.append(states)
                self.demo_actions.append(actions)
                self.demo_lengths.append(len(actions))

        if max_seq_length is None:
            self.max_seq_length = max(self.demo_lengths)
        else:
            self.max_seq_length = max_seq_length

        if transform == "id":
            self.transforms = T.Lambda(lambda x: x)
        elif transform == "crop":
            self.transforms = RandomCrop(min_side=int(self.image_size*0.8), max_side=self.image_size, output_size=self.image_size)
        elif transform == "crop_jitter":
            self.transforms = T.Compose([
                RandomCrop(min_side=int(self.image_size*0.8), max_side=self.image_size, output_size=self.image_size),
                RGBJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ])
        else:
            raise ValueError("Invalid transform type. Choose 'id', 'crop', or 'crop_jitter'.")

    def __len__(self):
        return len(self.demo_indices)

    def _get_eef_xyz_world(self):
        tcp_pos = self.env.unwrapped.agent.tcp_pos  # (num_envs, 3) tensor
        if hasattr(tcp_pos, 'detach'):
            tcp_pos = tcp_pos.detach().cpu().numpy()
        return np.asarray(tcp_pos, dtype=np.float32).reshape(-1)[:3]

    def _pointmap_from_position_texture(self, position_texture, cam2world_gl):
        """Convert SAPIEN camera-frame position texture to world-frame (3,H,W).

        position_texture: (H, W, 3) tensor in GL camera frame.
          - float dtype: meters
          - int16 dtype: millimeters (minimal shader)
        cam2world_gl: (4, 4) numpy array (GL model matrix).
        Returns (3, H, W) float32 numpy in world frame.
        """
        if hasattr(position_texture, 'detach'):
            pos = position_texture.detach().cpu().numpy()
        else:
            pos = np.asarray(position_texture)
        if pos.dtype == np.int16:
            pos = pos.astype(np.float32) * 0.001
        else:
            pos = pos.astype(np.float32)
        H, W, _ = pos.shape
        # Invalid pixels: SAPIEN writes (0,0,0) for no-hit / sky.
        invalid = np.all(pos == 0.0, axis=-1)  # (H, W)
        cam2world_gl = np.asarray(cam2world_gl, dtype=np.float32)
        ones = np.ones((H, W, 1), dtype=np.float32)
        pts_h = np.concatenate([pos, ones], axis=-1)  # (H, W, 4)
        pts_world = pts_h @ cam2world_gl.T  # (H, W, 4)
        pts_world = pts_world[..., :3]
        pts_world[invalid] = 0.0
        return pts_world.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)

    def __getitem__(self, index):
        demo_idx = index
        demo_length = self.demo_lengths[demo_idx]
        states = self.demo_states[demo_idx]
        actions = self.demo_actions[demo_idx]
        start_ts = np.random.randint(demo_length)

        # Set env state
        self.env.unwrapped.set_state_dict(states[start_ts])

        eef_xyz = self._get_eef_xyz_world()
        self._paired_crop.sample_offsets()

        if self.args.default_cam:
            cam_names = ["render_camera" for _ in range(self.args.num_side_cam)]
        else:
            start = self.args.m * demo_idx
            end = start + self.args.n
            window = np.arange(start, end, dtype=np.int64)
            chosen = np.random.choice(window, size=self.args.num_side_cam, replace=False)
            cam_names = [f"cam_{i}" for i in chosen.tolist()]

        cam_images = []
        cam_pointmaps = []
        cam_extrinsics_out = []
        cam_intrinsics_out = []

        # Update render so capture() returns current state.
        self.env.unwrapped.scene.update_render(
            update_sensors=False, update_human_render_cameras=True,
        )

        for cam_name in cam_names:
            camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
            params = camera.get_params()
            # get_params values can be batched (num_envs, ...); squeeze first.
            K_np = np.asarray(params["intrinsic_cv"], dtype=np.float32)
            if K_np.ndim == 3:
                K_np = K_np[0]
            cam2world_gl_np = np.asarray(params["cam2world_gl"], dtype=np.float32)
            if cam2world_gl_np.ndim == 3:
                cam2world_gl_np = cam2world_gl_np[0]

            if self.use_pointmaps:
                camera.capture()
                obs_dict = camera.get_obs(
                    rgb=True, depth=False, position=True,
                    segmentation=False, normal=False, albedo=False,
                )
                rgb_raw = obs_dict["rgb"]  # (1, H, W, 3 or 4) uint8 or float
                if hasattr(rgb_raw, 'detach'):
                    rgb_raw = rgb_raw.detach().cpu().numpy()
                if rgb_raw.ndim == 4:
                    rgb_raw = rgb_raw[0]
                rgb_np = rgb_raw[..., :3]
                if rgb_np.dtype != np.float32:
                    rgb_np = rgb_np.astype(np.float32) / 255.0
                rgb_tensor = einops.rearrange(
                    torch.from_numpy(np.ascontiguousarray(rgb_np)), 'h w c -> c h w'
                ).float().cuda()

                position_texture = obs_dict["position"]
                if position_texture.ndim == 4:
                    position_texture = position_texture[0]
                pointmap_np = self._pointmap_from_position_texture(
                    position_texture, cam2world_gl_np,
                )
            else:
                obs = self.env.unwrapped.render_rgb_array(cam_name)
                rgb_tensor = (obs.permute(0, 3, 1, 2).float() / 255.0)[0]
                pointmap_np = None

            if self.use_plucker:
                pl = self.plucker_embedder(K_np, cam2world_gl_np)["plucker"][0]
                plucker_tensor = einops.rearrange(pl, 'h w c -> c h w')
            else:
                _, H, W = rgb_tensor.shape
                plucker_tensor = torch.zeros(6, H, W, device=rgb_tensor.device)

            if self.use_pointmaps:
                pointmap_tensor = torch.from_numpy(pointmap_np).float().cuda()
                rgb_c = self._paired_crop(rgb_tensor)
                pointmap_c = self._paired_crop(pointmap_tensor)
                plucker_c = self._paired_crop(plucker_tensor)
                top, left = self._paired_crop.offsets()
                K_c = adjust_intrinsic(K_np, top, left)

                img_chw = torch.cat([rgb_c, plucker_c], dim=0)  # (9, dst, dst)
                cam_images.append(img_chw)
                cam_pointmaps.append(pointmap_c)
                cam_extrinsics_out.append(
                    torch.from_numpy(cam2world_gl_np).float().cuda()
                )
                cam_intrinsics_out.append(torch.from_numpy(K_c).float().cuda())
            else:
                img_chw = torch.cat([rgb_tensor, plucker_tensor], dim=0)
                cam_images.append(self.transforms(img_chw))
                cam_extrinsics_out.append(
                    torch.from_numpy(cam2world_gl_np).float().cuda()
                )
                cam_intrinsics_out.append(torch.from_numpy(K_np).float().cuda())

        # Stack per-camera images: [num_cameras, C, H, W]
        image_tensor = torch.stack(cam_images, dim=0)
        cam_extrinsics_stack = torch.stack(cam_extrinsics_out, dim=0)
        cam_intrinsics_stack = torch.stack(cam_intrinsics_out, dim=0)

        # qpos extraction
        st = states[start_ts]
        for key in ("panda", "panda_wristcam", "panda_stick"):
            if key in st["articulations"]:
                robot_qpos = st["articulations"][key][13:22]
                break

        # Match robosuite behavior: optionally drop proprio to zeros with probability prob_drop_proprio
        if np.random.rand() < self.args.prob_drop_proprio:
            robot_qpos = np.zeros_like(robot_qpos)
        actions_seq = actions[start_ts:]

        padded_actions = np.zeros((self.max_seq_length, actions.shape[1]), dtype=np.float32)
        seq_length = min(len(actions_seq), self.max_seq_length)
        padded_actions[:seq_length] = actions_seq[:seq_length]

        is_pad = np.zeros(self.max_seq_length, dtype=np.bool_)
        is_pad[seq_length:] = True

        state_normalized = (robot_qpos - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        actions_normalized = (padded_actions - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        out = {
            'image': image_tensor,
            'qpos': torch.from_numpy(state_normalized).float().cuda(),
            'actions': torch.from_numpy(actions_normalized).float().cuda(),
            'is_pad': torch.from_numpy(is_pad).cuda(),
            'eef_xyz': torch.from_numpy(eef_xyz).float().cuda(),
            'cam_extrinsics_full': cam_extrinsics_stack,  # (num_cams, 4, 4) c2w (GL)
            'cam_intrinsics_full': cam_intrinsics_stack,  # (num_cams, 3, 3)
        }
        if self.use_pointmaps:
            out['pointmap'] = torch.stack(cam_pointmaps, dim=0)  # (num_cams, 3, H, W)
        return out

# --- Data Loading Function ---

def load_data(args, env, val_split=0.1):
    json_path = os.path.join(args.dataset_path, 'trajectory.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    available_demos = len(json_data['episodes'])

    assert args.num_episodes + 10 <= available_demos, "Not enough demos to split"

    train_indices = list(range(args.num_episodes))
    val_indices = list(range(args.num_episodes, args.num_episodes + 10))
    
    print("Computing normalization statistics...")
    norm_stats = get_norm_stats(args.dataset_path, num_demos=args.num_episodes, policy_class=args.policy_class)
    print("Normalization statistics computed.")
    
    print("Loading training dataset...")
    train_dataset = EpisodicDataset(
        train_indices, 
        norm_stats, 
        args,
        transform=args.transform,
        env=env
    )
    
    print("Loading validation dataset...")
    val_dataset = EpisodicDataset(
        val_indices, 
        norm_stats,
        args,
        transform="id",  # Use simpler transform for validation
        env=env
    )
    print("Datasets loaded.")
    
    max_seq_length = train_dataset.max_seq_length
    print(f"Using max sequence length: {max_seq_length}")

    train_dataset.max_seq_length = max_seq_length
    val_dataset.max_seq_length = max_seq_length

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_dataloader, val_dataloader, norm_stats
