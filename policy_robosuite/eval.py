import os, random, math, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import einops
import json
from contextlib import nullcontext
import h5py
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as TF
from cam_embedding import PluckerEmbedder

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from policy_common.pointmap import mujoco_metric_depth, backproject, pose_from_pos_ori, c2w_opengl_to_opencv
from policy_common.paired_crop import adjust_intrinsic

def to_mp4(save_path, image_list, reward_list=None, success_list=None, info_list=None):
    """
    Save a list of images as an MP4 video with reward and success overlaid using imageio with H264 encoding.
    """
    import imageio
    import os
    
    # Convert images to list of numpy arrays if needed
    if isinstance(image_list, torch.Tensor):
        image_list = image_list.cpu().numpy()
    
    # Ensure the save path has .mp4 extension
    if not save_path.endswith('.mp4'):
        save_path = save_path.replace('.avi', '.mp4')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare frames with overlays
    frames = []
    for i, img in enumerate(image_list):
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Add text overlays if provided
        if reward_list is not None or success_list is not None:
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default()
            
            y_offset = 10
            if reward_list is not None and i < len(reward_list):
                draw.text((10, y_offset), f"Reward: {reward_list[i]:.3f}", fill=(255, 255, 255), font=font)
                y_offset += 30
            
            if success_list is not None and i < len(success_list):
                success_text = "SUCCESS" if success_list[i] else "FAILURE"
                color = (0, 255, 0) if success_list[i] else (255, 0, 0)
                draw.text((10, y_offset), success_text, fill=color, font=font)
            
            img = np.array(img_pil)
        
        frames.append(img)
    
    # Save video with H264 encoding, suppress warnings
    import sys
    
    # Redirect stderr to suppress libx264 warnings
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        
        with imageio.get_writer(save_path, fps=10, codec='h264', ffmpeg_params=['-crf', '23', '-preset', 'medium']) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        sys.stderr = old_stderr

class Evaluator:
    """
    Class to evaluate policies on robosuite environments
    """
    def __init__(self, env, norm_stats, dataset_path, args):

        self.env = env
        self.args = args
        self.norm_stats = {k: torch.tensor(v).float() for k, v in norm_stats.items()}
        self.chunk_size = args.chunk_size
        self.max_steps = args.eval_max_steps
        self.eval_save_n_video = args.eval_save_n_video
        
        self.H = 256
        self.W = 256

        self.success_by_seed = {}
        
        self.plucker_embedder = PluckerEmbedder(img_size=256, device='cuda')
        self.intrinsics = self._get_camera_intrinsics()

        # articubot_dit{,_rgb} feeds center-cropped (crop_dst, crop_dst) tensors
        # to the policy and additionally consumes depth-derived pointmaps.
        # Crop constants are static (center crop), so precompute them once.
        # Single DINO/paired-crop codepath shared by articubot DiT variants and
        # act_dino: center-crop 256→224, adjust K, emit pointmap. act_dino
        # simply ignores the pointmap field.
        self.is_dino = getattr(args, 'policy_class', '') in (
            'articubot_dit', 'articubot_dit_rgb', 'act_dino'
        )
        self.crop_dst = 224
        self.crop_top = (self.H - self.crop_dst) // 2
        self.crop_left = (self.W - self.crop_dst) // 2
        self.K_crop = adjust_intrinsic(self.intrinsics, self.crop_top, self.crop_left)

        camera_poses_dir = args.camera_poses_dir
        self.num_side_cam = int(args.num_side_cam)
        if not args.default_cam:
            self.camera_poses_by_name = {}
            for filename in args.pose_files:
                poses_path = os.path.join(camera_poses_dir, filename)
                with open(poses_path, 'r') as f:
                    raw = json.load(f)
                pose_name = os.path.splitext(filename)[0]
                self.camera_poses_by_name[pose_name] = raw['poses']
                print(f"Loaded {len(raw['poses'])} camera poses (old format) from {poses_path}; key={pose_name}; num_side_cam={self.num_side_cam}")
        else:
            print("Evaluator: default_cam=True; using agentview pose duplicated if needed")

        # Detect action space from dataset metadata
        with h5py.File(dataset_path, 'r') as f:
            action_space_attr = f['data'].attrs['action_space']
        self.action_space = action_space_attr.decode('utf-8') if isinstance(action_space_attr, bytes) else action_space_attr
    
    def _get_camera_intrinsics(self):
        """Extract camera intrinsics from robosuite environment."""
        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)
        
        # Get field of view and image dimensions
        fovy = self.env.sim.model.cam_fovy[cam_id] * np.pi / 180.0
        width, height = 256, 256
        
        # Compute focal length
        focal_length = height / (2 * np.tan(fovy / 2))
        
        # Create intrinsics matrix
        intrinsics = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return intrinsics
    
    def _set_camera_pose(self, cam_to_world):
        """Set camera pose in robosuite environment."""
        
        cam_name = "agentview"
        cam_id = self.env.sim.model.camera_name2id(cam_name)
        
        # Set camera position
        self.env.sim.model.cam_pos[cam_id] = cam_to_world[:3, 3]
        
        # Set camera orientation (convert rotation matrix to quaternion)
        rotation = Rotation.from_matrix(cam_to_world[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        # MuJoCo uses [w, x, y, z] format
        self.env.sim.model.cam_quat[cam_id] = [quat[3], quat[0], quat[1], quat[2]]
    
    def _render_cam_image(self, cam_pose_raw):
        if cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw)
            self._set_camera_pose(cam_pose)
        self.env.sim.forward()
        img = self.env.sim.render(camera_name="agentview", height=256, width=256, depth=False)
        return np.flipud(img).copy()

    def _render_cam_rgbd(self, cam_pose_raw):
        """Render RGB + world-frame pointmap at 256x256. Returns (rgb, pointmap, c2w)."""
        if cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw, dtype=np.float32)
            self._set_camera_pose(cam_pose)
        self.env.sim.forward()
        rgb, depth_norm = self.env.sim.render(
            camera_name="agentview", height=256, width=256, depth=True,
        )
        rgb = np.flipud(rgb).copy()
        depth_norm = np.flipud(depth_norm).copy()
        if cam_pose_raw is None:
            cam_id = self.env.sim.model.camera_name2id("agentview")
            pos = np.array(self.env.sim.model.cam_pos[cam_id], dtype=np.float32)
            q = self.env.sim.model.cam_quat[cam_id]  # (w,x,y,z)
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().astype(np.float32)
            cam_pose = pose_from_pos_ori(pos, R)
        extent = self.env.sim.model.stat.extent
        near = float(self.env.sim.model.vis.map.znear * extent)
        far = float(self.env.sim.model.vis.map.zfar * extent)
        depth_m = mujoco_metric_depth(depth_norm, near, far)
        # Pose files + mujoco cam_quat are GL convention (camera looks down -Z).
        # backproject assumes OpenCV (+Z forward), so convert.
        pointmap = backproject(
            depth_m, self.intrinsics, c2w_opengl_to_opencv(cam_pose),
            invalid_value=0.0, max_depth=far * 0.99,
        )
        return rgb, pointmap, cam_pose

    def _legacy_cam_extrinsics(self, pose_set):
        """Build the [1, 2, 4, 4] cam_extrinsics field ACT/DP/SmolVLA expect."""
        if not (self.args.use_cam_pose and not self.args.default_cam):
            return torch.zeros(1, 2, 4, 4, device='cuda')
        per_cam_T = []
        for i in range(2):
            if i < len(pose_set) and pose_set[i] is not None:
                per_cam_T.append(torch.from_numpy(np.array(pose_set[i], dtype=np.float32)).float().cuda())
            else:
                per_cam_T.append(torch.zeros(4, 4, device='cuda'))
        return torch.stack(per_cam_T, dim=0).unsqueeze(0)

    def _eef_xyz(self):
        return torch.from_numpy(
            np.array(self.env.sim.data.site_xpos[
                self.env.robots[0].eef_site_id[self.env.robots[0].arms[0]]
            ], dtype=np.float32)
        ).unsqueeze(0).cuda()

    def _build_pointmap_batch(self, per_cam_rgb, per_cam_pointmaps, per_cam_poses, pose_set):
        """Inference batch. Center-crop 256→crop_dst with matching K."""
        top, left, dst = self.crop_top, self.crop_left, self.crop_dst
        imgs, pms, extrs = [], [], []
        for rgb_np, pm_np, cam_pose in zip(per_cam_rgb, per_cam_pointmaps, per_cam_poses):
            # rgb | plucker tensor built by the existing helper (no crop).
            # Zero plucker in default_cam mode to match training.
            plu_pose = None if self.args.default_cam else cam_pose
            rgb_plu = self._image_to_tensor(rgb_np, plu_pose)  # (9, 256, 256) cpu
            pm_t = torch.from_numpy(pm_np).float()             # (3, 256, 256) cpu
            imgs.append(rgb_plu[:, top:top + dst, left:left + dst])
            pms.append(pm_t[:, top:top + dst, left:left + dst])
            extrs.append(torch.from_numpy(cam_pose).float())
        n = len(imgs)
        K_crop_t = torch.from_numpy(self.K_crop).float().unsqueeze(0).expand(n, -1, -1)
        return {
            'image': torch.stack(imgs, dim=0).unsqueeze(0).cuda(),
            'pointmap': torch.stack(pms, dim=0).unsqueeze(0).cuda(),
            'cam_extrinsics_full': torch.stack(extrs, dim=0).unsqueeze(0).cuda(),
            'cam_intrinsics_full': K_crop_t.unsqueeze(0).cuda(),
            'eef_xyz': self._eef_xyz(),
            'cam_extrinsics': self._legacy_cam_extrinsics(pose_set),
        }

    def _image_to_tensor(self, cam_img, cam_pose_raw):
        rgb_tensor = einops.rearrange(torch.from_numpy(cam_img).float() / 255.0, 'h w c -> c h w')
        if self.args.use_plucker and cam_pose_raw is not None:
            cam_pose = np.array(cam_pose_raw)
            intrinsics_tensor = torch.from_numpy(self.intrinsics).unsqueeze(0).float().cuda()
            cam_to_world_tensor = torch.from_numpy(cam_pose).unsqueeze(0).float().cuda()
            with torch.no_grad():
                plucker_data = self.plucker_embedder(intrinsics_tensor, cam_to_world_tensor)
                plucker_tensor = einops.rearrange(plucker_data['plucker'][0].cpu(), 'h w c -> c h w')
        else:
            plucker_tensor = torch.zeros(6, rgb_tensor.shape[1], rgb_tensor.shape[2])
        return torch.cat([rgb_tensor, plucker_tensor], dim=0)
    
    def evaluate(self, policy, save_path, video_prefix, pose_name, init_state=None, episode_num=0):
        np.random.seed(episode_num)
        
        # Initialize environment
        if init_state is not None:
            self.env.reset()
            self.env.sim.set_state_from_flattened(init_state)
        else:
            self.env.reset()

        if self.action_space in ('eef_delta', 'joint_delta'):
            self.env.set_init_action()

        camera_frames, success_labels, rewards, success = [], [], [], []
        done = False
        step = 0
        has_succeeded = False

        if self.args.default_cam:
            pose_set = [None] * self.num_side_cam
        else:
            poses_list = self.camera_poses_by_name[pose_name]
            if self.num_side_cam == 1:
                pose_set = [poses_list[episode_num]]
            else:
                pose_set = [poses_list[2 * episode_num], poses_list[2 * episode_num + 1]]
        
        while not done and step < self.max_steps:
            if self.is_dino:
                per_cam = [self._render_cam_rgbd(p) for p in pose_set]
                per_cam_images = [x[0] for x in per_cam]
                per_cam_pointmaps = [x[1] for x in per_cam]
                per_cam_poses = [x[2] for x in per_cam]
            else:
                per_cam_images = [self._render_cam_image(p) for p in pose_set]
                per_cam_pointmaps = None
                per_cam_poses = None

            camera_frame = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
            camera_frames.append(camera_frame)
            success_labels.append(has_succeeded)

            if self.is_dino:
                batch = self._build_pointmap_batch(
                    per_cam_images, per_cam_pointmaps, per_cam_poses, pose_set,
                )
            else:
                per_cam_tensors = [self._image_to_tensor(img, p) for img, p in zip(per_cam_images, pose_set)]
                image_tensor = torch.stack(per_cam_tensors, dim=0).unsqueeze(0).cuda()
                batch = {
                    'image': image_tensor,
                    'cam_extrinsics': self._legacy_cam_extrinsics(pose_set),
                }

            state_vector = self.env.sim.data.qpos[:7]
            if np.random.rand() < self.args.prob_drop_proprio:
                state_vector = np.zeros_like(state_vector)
            normalized_state = (state_vector - self.norm_stats["state_mean"].cpu().numpy()) / self.norm_stats["state_std"].cpu().numpy()
            state_tensor = einops.rearrange(torch.tensor(normalized_state, device="cuda").float(), 'd -> 1 d')
            batch['qpos'] = state_tensor

            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if self.args.use_fp16 else nullcontext()):
                action_chunk = policy(batch)
            action_chunk = action_chunk[0].float().cpu().numpy() * self.norm_stats["action_std"].cpu().numpy() + self.norm_stats["action_mean"].cpu().numpy()
            
            # Execute action chunk
            for i in range(action_chunk.shape[0]):                
                if done or step >= self.max_steps:
                    break

                next_obs, reward, done, info = self.env.step(action_chunk[i])
                
                current_success = (reward == 1)
                has_succeeded = has_succeeded or current_success
                rewards.append(float(reward))
                success.append(current_success)
                step += 1
                
                if episode_num < self.eval_save_n_video and i < action_chunk.shape[0] - 1:
                    per_cam_images = [self._render_cam_image(p) for p in pose_set]
                    camera_frame = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
                    camera_frames.append(camera_frame)
                    success_labels.append(has_succeeded)
        
        final_success = any(success)
        print(f"Episode {episode_num}: Success = {final_success}")
        
        self.success_by_seed[episode_num] = bool(final_success)
        
        if episode_num < self.eval_save_n_video:
            camera_video_path = os.path.join(save_path, f"{video_prefix}_{pose_name}_success_{final_success}.mp4")
            to_mp4(camera_video_path, camera_frames, success_list=success_labels)
        
        results = {
            "success_rate": float(final_success),
            "mean_episode_length": float(step),
            "max_rewards": rewards
        }
        
        return results, float(final_success), step