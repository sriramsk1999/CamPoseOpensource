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

from eval import to_mp4

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _REPO_ROOT)
from policy_common.pointmap import mujoco_metric_depth, backproject, pose_from_pos_ori, c2w_opengl_to_opencv
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


def get_norm_stats(dataset_path, num_demos, policy_class: str = 'dp'):
    """
    Compute normalization statistics for actions and states from robosuite dataset.
    
    Args:
        dataset_path (str): Path to the robosuite HDF5 dataset
        num_demos (int): Number of demonstrations to use for computing stats.
    Returns:
        dict: Dictionary containing normalization statistics
    """
    all_states_data = []
    all_action_data = []
    
    with h5py.File(dataset_path, 'r') as dataset_file:
        demo_keys = [k for k in dataset_file['data'].keys() if k.startswith('demo_')]
        num_demos_available = len(demo_keys)
        num_demos_to_use = min(num_demos, num_demos_available)
            
        print(f"Computing robosuite normalization statistics using {num_demos_to_use} demonstrations from {dataset_path}...")
        
        for i in range(num_demos_to_use):
            demo_key = f'demo_{i}'
            
            # Load states and actions from robosuite format
            states = dataset_file[f'data/{demo_key}/states'][()].astype(np.float32)
            actions = dataset_file[f'data/{demo_key}/actions'][()].astype(np.float32)
            
            # Extract only robot joint positions (first 7 dimensions)
            robot_qpos = states[:, :7]  # Robot joint positions

            all_states_data.append(robot_qpos)
            all_action_data.append(actions)

    states_array = np.concatenate(all_states_data, axis=0)
    actions_array = np.concatenate(all_action_data, axis=0)

    # Use min–max scaling for diffusion policy variants ('dp') so that
    # values lie in approximately [-1, 1], matching the DDPM clip range during sampling.
    if policy_class in ['dp']:
        # Use min–max scaling encoded as mean/std such that
        # (x - mean)/std == 2*(x - min)/(max-min) - 1
        s_min = states_array.min(axis=0)
        s_max = states_array.max(axis=0)
        a_min = actions_array.min(axis=0)
        a_max = actions_array.max(axis=0)

        state_std = np.maximum((s_max - s_min) / 2.0, 1e-6)
        state_mean = (s_min + s_max) / 2.0

        action_std = np.maximum((a_max - a_min) / 2.0, 1e-6)
        action_mean = (a_min + a_max) / 2.0

        print("action max, min:", a_max, a_min)
    elif policy_class == 'pi0':
        # FAST expects per-dimension robust bounds mapped to [-1, 1].
        # Encode via mean/std so that (x - mean)/std reproduces the robust mapping.
        # Actions: use 1st and 99th percentiles per dimension.
        a_q1 = np.quantile(actions_array, 0.01, axis=0)
        a_q99 = np.quantile(actions_array, 0.99, axis=0)
        action_std = np.maximum((a_q99 - a_q1) / 2.0, 1e-6)
        action_mean = (a_q1 + a_q99) / 2.0

        # States: keep z-score (or switch to robust similarly if desired)
        state_mean = np.mean(states_array, axis=0)
        state_std = np.std(states_array, axis=0)
        state_std = np.clip(state_std, 1e-4, np.inf)
        print("[pi0] action q99, q1:", a_q99, a_q1)
    else:
        # Default z-score
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
        "action_std": action_std,
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

# --- Dataset Class ---

class EpisodicDataset(Dataset):
    """
    Dataset for loading episodic data from robosuite demonstration files.
    Renders images on-the-fly with dynamic camera poses and Plucker embeddings.
    """
    def __init__(self, demo_indices, norm_stats, args, camera_poses_file=None,
                 max_seq_length=None, transform="id", env=None):
        """
        Args:
            demo_indices (list): List of demonstration indices to use
            norm_stats (dict): Normalization statistics for actions and states
            args: Arguments object containing dataset_path, camera_names, use_plucker, etc.
            camera_poses_file (str): Path to JSON file with camera poses
            max_seq_length (int, optional): Maximum sequence length for actions
            transform (str): Transform to apply to images - "id", "crop", "jitter", or "crop_jitter"
            env: Pre-created robosuite environment to use for rendering
        """
        super().__init__()
        self.demo_indices = demo_indices
        self.norm_stats = norm_stats
        self.args = args
        self.image_size = 256  # Standard image size
        self.use_plucker = args.use_plucker
        self.num_cameras = args.num_side_cam
        self.use_pointmaps = args.use_pointmaps
        self._paired_crop = PairedRandomCrop(src=self.image_size, dst=224)
        
        if not self.args.default_cam:
            poses_path = os.path.join(self.args.camera_poses_dir, camera_poses_file)
            with open(poses_path, 'r') as f:
                raw = json.load(f)
            self.camera_poses = raw['poses']
            
            print(f"Loaded {len(self.camera_poses)} camera poses (old format) from {poses_path}; num_side_cam={self.num_cameras}")
        else:
            # self.camera_poses = None
            print("Using default agentview camera pose (duplicated if multiple cams)")
        
        if self.use_plucker:
            self.plucker_embedder = PluckerEmbedder(img_size=self.image_size, device='cuda')
        else:
            self.plucker_embedder = None
        
        # Load demonstration data
        self.demo_states = []
        self.demo_actions = []
        self.demo_lengths = []
        
        print(f"Loading robosuite data for {len(demo_indices)} demos from {args.dataset_path}...")
        with h5py.File(args.dataset_path, "r") as dataset_file:
            for idx in self.demo_indices:
                demo_key = f'demo_{idx}'
                
                states = dataset_file[f'data/{demo_key}/states'][()].astype(np.float32)
                actions = dataset_file[f'data/{demo_key}/actions'][()].astype(np.float32)
                
                self.demo_states.append(states)
                self.demo_actions.append(actions)
                demo_len = len(actions)
                self.demo_lengths.append(demo_len)

        print(f"Successfully loaded {len(self.demo_indices)} robosuite demonstrations.")

        self.env = env

        if max_seq_length is None:
            self.max_seq_length = max(self.demo_lengths)
        else:
            self.max_seq_length = max_seq_length

        # Set up image transforms
        if transform == "id":
            self.transforms = T.Resize(self.image_size)
        elif transform == "crop":
            self.transforms = RandomCrop(min_side=int(self.image_size*0.8), max_side=self.image_size, output_size=self.image_size)
        elif transform == "crop_jitter":
            self.transforms = T.Compose([
                RandomCrop(min_side=int(self.image_size*0.8), max_side=self.image_size, output_size=self.image_size),
                RGBJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ])
        else:
            raise ValueError("Invalid transform type. Choose 'id', 'crop', 'jitter', or 'crop_jitter'.")
            
    def __len__(self):
        return len(self.demo_indices)
    
    def _get_camera_intrinsics(self):
        cam_name = "agentview"  # assume same intrinsics
        cam_id = self.env.sim.model.camera_name2id(cam_name)
        
        fovy = self.env.sim.model.cam_fovy[cam_id] * np.pi / 180.0
        width, height = self.image_size, self.image_size
        
        focal_length = height / (2 * np.tan(fovy / 2))
        
        intrinsics = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsics
    
    def _set_camera_pose(self, cam_to_world):
        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)
        
        self.env.sim.model.cam_pos[cam_id] = cam_to_world[:3, 3]
        rotation = Rotation.from_matrix(cam_to_world[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]

        self.env.sim.model.cam_quat[cam_id] = [quat[3], quat[0], quat[1], quat[2]]
    
    def _mujoco_near_far(self):
        extent = self.env.sim.model.stat.extent
        near = self.env.sim.model.vis.map.znear * extent
        far = self.env.sim.model.vis.map.zfar * extent
        return float(near), float(far)

    def _get_eef_xyz_world(self):
        arm = self.env.robots[0].arms[0]
        site_id = self.env.robots[0].eef_site_id[arm]
        return np.array(self.env.sim.data.site_xpos[site_id], dtype=np.float32)

    def __getitem__(self, demo_idx):
        demo_length = self.demo_lengths[demo_idx]
        states = self.demo_states[demo_idx]
        actions = self.demo_actions[demo_idx]
        start_ts = np.random.randint(demo_length)

        self.env.sim.set_state_from_flattened(states[start_ts])
        self.env.sim.forward()
        eef_xyz = self._get_eef_xyz_world()

        # Sample one random crop window and reuse it for all images
        self._paired_crop.sample_offsets()

        cam_images = []
        cam_pointmaps = []
        cam_extrinsics_out = []
        cam_intrinsics_out = []

        if not self.args.default_cam:
            start = self.args.m * demo_idx
            end = start + self.args.n
            window = np.arange(start, end, dtype=np.int64)
            chosen = np.random.choice(window, size=self.args.num_side_cam, replace=False)
            pose_set = [self.camera_poses[i] for i in chosen.tolist()]
        else:
            pose_set = [None] * self.args.num_side_cam

        K_base = self._get_camera_intrinsics()  # (3,3) for the uncropped 256 image

        for cam_pose_raw in pose_set:
            if not self.args.default_cam:
                cam_pose = np.array(cam_pose_raw, dtype=np.float32)  # c2w (OpenCV/MuJoCo)
                self._set_camera_pose(cam_pose)
            else:
                # Read the current agentview pose directly from MuJoCo.
                cam_id = self.env.sim.model.camera_name2id("agentview")
                pos = np.array(self.env.sim.model.cam_pos[cam_id], dtype=np.float32)
                q = self.env.sim.model.cam_quat[cam_id]  # (w,x,y,z)
                R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().astype(np.float32)
                cam_pose = pose_from_pos_ori(pos, R)
            self.env.sim.forward()

            if self.use_pointmaps:
                rgb_img, depth_norm = self.env.sim.render(
                    camera_name="agentview", height=self.image_size,
                    width=self.image_size, depth=True,
                )
                rgb_img = np.flipud(rgb_img).copy()
                depth_norm = np.flipud(depth_norm).copy()
                near, far = self._mujoco_near_far()
                depth_m = mujoco_metric_depth(depth_norm, near, far)
                # Pose files + mujoco cam_quat are GL convention (camera looks
                # down -Z). backproject assumes OpenCV (+Z forward), so convert.
                pointmap_np = backproject(
                    depth_m, K_base, c2w_opengl_to_opencv(cam_pose),
                    invalid_value=0.0, max_depth=far * 0.99,
                )  # (3, H, W) world-frame xyz
            else:
                rgb_img = self.env.sim.render(
                    camera_name="agentview", height=self.image_size,
                    width=self.image_size, depth=False,
                )
                rgb_img = np.flipud(rgb_img).copy()
                pointmap_np = None

            rgb_tensor = einops.rearrange(
                torch.from_numpy(rgb_img).float() / 255.0, 'h w c -> c h w'
            ).cuda()

            if self.use_plucker and not self.args.default_cam:
                intrinsics_tensor = torch.from_numpy(K_base).unsqueeze(0).float().cuda()
                cam_to_world_tensor = torch.from_numpy(cam_pose).unsqueeze(0).float().cuda()
                with torch.no_grad():
                    plucker_data = self.plucker_embedder(intrinsics_tensor, cam_to_world_tensor)
                    plucker_tensor = einops.rearrange(plucker_data['plucker'][0], 'h w c -> c h w')
            else:
                plucker_tensor = torch.zeros(6, rgb_tensor.shape[1], rgb_tensor.shape[2], device='cuda')

            if self.use_pointmaps:
                pointmap_tensor = torch.from_numpy(pointmap_np).float().cuda()
                # Single joint crop — preserves geometric alignment between
                # RGB, pointmap, and Plucker channels.
                rgb_c = self._paired_crop(rgb_tensor)
                pointmap_c = self._paired_crop(pointmap_tensor)
                plucker_c = self._paired_crop(plucker_tensor)
                top, left = self._paired_crop.offsets()
                K_c = adjust_intrinsic(K_base, top, left)

                img_chw = torch.cat([rgb_c, plucker_c], dim=0)  # (9, dst, dst)
                cam_images.append(img_chw)
                cam_pointmaps.append(pointmap_c)
                cam_extrinsics_out.append(torch.from_numpy(cam_pose).float().cuda())
                cam_intrinsics_out.append(torch.from_numpy(K_c).float().cuda())
            else:
                img_chw = torch.cat([rgb_tensor, plucker_tensor], dim=0)
                cam_images.append(self.transforms(img_chw))
                cam_extrinsics_out.append(torch.from_numpy(cam_pose).float().cuda())
                cam_intrinsics_out.append(torch.from_numpy(K_base).float().cuda())

        # Stack per-camera images: [num_cameras, C, H, W]
        image_tensor = torch.stack(cam_images, dim=0)
        cam_extrinsics_stack = torch.stack(cam_extrinsics_out, dim=0)  # (num_cams, 4, 4) c2w
        cam_intrinsics_stack = torch.stack(cam_intrinsics_out, dim=0)  # (num_cams, 3, 3)

        # Legacy "cam_extrinsics" field used by existing policies: always [2, 4, 4]
        # holding c2w for the first two cameras, zero-padded.
        if self.args.use_cam_pose and not self.args.default_cam:
            legacy = []
            for i in range(2):
                if i < len(pose_set) and pose_set[i] is not None:
                    legacy.append(torch.from_numpy(np.array(pose_set[i], dtype=np.float32)).float().cuda())
                else:
                    legacy.append(torch.zeros(4, 4, device='cuda'))
            cam_extrinsics = torch.stack(legacy, dim=0)
        else:
            cam_extrinsics = torch.zeros(2, 4, 4, device='cuda')

        # Normalize and convert to tensors
        robot_qpos = states[start_ts][:7]
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
            'cam_extrinsics': cam_extrinsics,
            # New fields (always emitted; existing policies ignore them)
            'eef_xyz': torch.from_numpy(eef_xyz).float().cuda(),
            'cam_extrinsics_full': cam_extrinsics_stack,  # (num_cams, 4, 4) c2w
            'cam_intrinsics_full': cam_intrinsics_stack,  # (num_cams, 3, 3)
        }
        if self.use_pointmaps:
            out['pointmap'] = torch.stack(cam_pointmaps, dim=0)  # (num_cams, 3, H, W)
        return out

    def __del__(self):
        """Close the environment when the dataset is destroyed."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()

# --- Data Loading Function ---

def load_data(args, env, val_split=0.1):
    with h5py.File(args.dataset_path, 'r') as f:
        available_demos = len([k for k in f['data'].keys() if k.startswith('demo_')])

    assert args.num_episodes + 10 <= available_demos, "Not enough demos to split"

    train_indices = list(range(args.num_episodes))
    val_indices = list(range(args.num_episodes, args.num_episodes + 10))
    
    print("Computing normalization statistics...")
    # Choose normalization style based on policy_class (dp -> min-max-as-mean/std)
    norm_stats = get_norm_stats(args.dataset_path, num_demos=args.num_episodes, policy_class=args.policy_class)
    print("Normalization statistics computed.")
    
    print("Loading training dataset...")
    train_dataset = EpisodicDataset(
        train_indices, 
        norm_stats, 
        args,
        camera_poses_file=args.train_poses_file,
        transform=args.transform,
        env=env
    )
    
    print("Loading validation dataset...")
    val_dataset = EpisodicDataset(
        val_indices, 
        norm_stats,
        args,
        camera_poses_file=args.test_poses_file,
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
        num_workers=0,  # Disable multiprocessing due to robosuite environment
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
