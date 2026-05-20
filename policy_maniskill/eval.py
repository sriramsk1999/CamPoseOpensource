import os, random, math, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import einops
import torchvision.transforms.functional as TF
from cam_embedding import PluckerEmbedder
import gymnasium as gym
from contextlib import nullcontext

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from policy_common.paired_crop import adjust_intrinsic
from policy_common.canonical_view import fuse_and_render_from_world_points
from policy_common.pointmap import c2w_opengl_to_opencv, invert_pose


def _to_numpy(x):
    """np.asarray-equivalent that handles CUDA torch tensors. See
    policy_maniskill/utils.py for rationale."""
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)

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

        # Convert to PIL for overlay drawing
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()

        # Upper-right corner: current step index
        step_text = f"Step {i}"
        draw.text((img_pil.width - 10, 10), step_text, fill=(255, 255, 255), font=font, anchor='rt')

        # Upper-left overlays (optional): reward and success
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
    Class to evaluate policies on ManiSkill environments
    """
    def __init__(self, env, norm_stats, args):

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
        self.num_side_cam = args.num_side_cam

        # All DiT-family wrappers consume {qpos, image, eef_xyz} and feed
        # 224×224 imagery into the encoder. RoPE4D + 3DFA additionally read
        # pointmap (and RoPE4D reads extrinsics + intrinsics). Mirrors the
        # is_articubot flag in policy_maniskill/utils.py so train/eval
        # batches stay structurally identical. dit_dino_cv routes here only
        # for the 224 center crop — the wrapper doesn't read pointmap.
        self.is_articubot = args.policy_class in (
            'dit_rope4d_dino_cv', 'dit_dino_sv', 'dit_dino_cv',
            'flow_matching_3dfa',
        )
        # Center crop 256 → 224 for the is_articubot path (deterministic
        # mirror of the dataloader's PairedRandomCrop at eval time).
        self.crop_dst = 224
        self.crop_top = (self.H - self.crop_dst) // 2
        self.crop_left = (self.W - self.crop_dst) // 2

        # Canonical-view setup. The first num_side_cam render cameras serve
        # as canonical target viewpoints. Deterministic per env seed (same
        # as dataloader). Stays None if --use_canonical_views=0.
        self.use_canonical_views = bool(getattr(args, 'use_canonical_views', False))
        self.canonical_c2ws_gl = None
        self.canonical_w2cs_cv = None
        self.canonical_Ks = None
        if self.use_canonical_views:
            self.canonical_c2ws_gl = []
            self.canonical_w2cs_cv = []
            self.canonical_Ks = []
            for i in range(self.num_side_cam):
                cam = env.unwrapped.scene.human_render_cameras[f'cam_{i}']
                params = cam.get_params()
                K = _to_numpy(params["intrinsic_cv"]).astype(np.float32)
                if K.ndim == 3:
                    K = K[0]
                c2w_gl = _to_numpy(params["cam2world_gl"]).astype(np.float32)
                if c2w_gl.ndim == 3:
                    c2w_gl = c2w_gl[0]
                self.canonical_c2ws_gl.append(c2w_gl)
                self.canonical_Ks.append(K)
                self.canonical_w2cs_cv.append(
                    invert_pose(c2w_opengl_to_opencv(c2w_gl))
                )
        
    
    def _get_camera_intrinsics(self, cam_name):
        camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
        return camera.get_params()["intrinsic_cv"].cpu()

    def _get_cam2world(self, cam_name):
        camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
        return camera.get_params()["cam2world_gl"].cpu()

    def _pointmap_from_position_texture(self, position_texture, cam2world_gl):
        """SAPIEN camera-frame position texture -> world-frame (3, H, W).
        Mirror of utils.py:EpisodicDataset._pointmap_from_position_texture."""
        if hasattr(position_texture, 'detach'):
            pos = position_texture.detach().cpu().numpy()
        else:
            pos = np.asarray(position_texture)
        if pos.dtype == np.int16:
            pos = pos.astype(np.float32) * 0.001
        else:
            pos = pos.astype(np.float32)
        H, W, _ = pos.shape
        invalid = np.all(pos == 0.0, axis=-1)
        cam2world_gl = np.asarray(cam2world_gl, dtype=np.float32)
        ones = np.ones((H, W, 1), dtype=np.float32)
        pts_h = np.concatenate([pos, ones], axis=-1)
        pts_world = (pts_h @ cam2world_gl.T)[..., :3]
        pts_world[invalid] = 0.0
        return pts_world.transpose(2, 0, 1).astype(np.float32)

    def _build_articubot_step(self, cam_names, drop_proprio):
        """Per-step batch construction for is_articubot policies.

        Returns (batch_dict, per_cam_full_rgb_uint8) where:
          - batch_dict matches the dataloader output keys exactly:
              image (1, n_cams, 9, 224, 224)  [rgb + plucker]
              pointmap (1, n_cams, 3, 224, 224) world-frame xyz
              qpos (1, 9) z-scored
              eef_xyz (1, 3) raw
              cam_extrinsics_full (1, n_cams, 4, 4) c2w (GL)
              cam_intrinsics_full (1, n_cams, 3, 3) (adjusted for crop)
          - per_cam_full_rgb_uint8 is the uncropped 256x256 RGB per cam,
            for the video recording.
        """
        top, left, dst = self.crop_top, self.crop_left, self.crop_dst
        # Update render so capture() returns current state.
        self.env.unwrapped.scene.update_render(
            update_sensors=False, update_human_render_cameras=True,
        )

        cam_images, cam_pointmaps, cam_extrs, cam_intrs = [], [], [], []
        per_cam_full_rgb = []
        for cam_name in cam_names:
            camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
            params = camera.get_params()
            K_np = _to_numpy(params["intrinsic_cv"]).astype(np.float32)
            if K_np.ndim == 3:
                K_np = K_np[0]
            c2w_gl_np = _to_numpy(params["cam2world_gl"]).astype(np.float32)
            if c2w_gl_np.ndim == 3:
                c2w_gl_np = c2w_gl_np[0]

            camera.capture()
            obs_dict = camera.get_obs(
                rgb=True, depth=False, position=True,
                segmentation=False, normal=False, albedo=False,
            )
            rgb_raw = obs_dict["rgb"]
            if hasattr(rgb_raw, 'detach'):
                rgb_raw = rgb_raw.detach().cpu().numpy()
            if rgb_raw.ndim == 4:
                rgb_raw = rgb_raw[0]
            rgb_np = rgb_raw[..., :3]
            if rgb_np.dtype != np.float32:
                rgb_uint8 = rgb_np.astype(np.uint8)
                rgb_np = rgb_np.astype(np.float32) / 255.0
            else:
                rgb_uint8 = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
            per_cam_full_rgb.append(rgb_uint8)
            rgb_tensor = einops.rearrange(
                torch.from_numpy(np.ascontiguousarray(rgb_np)), 'h w c -> c h w',
            ).float().cuda()

            position_texture = obs_dict["position"]
            if position_texture.ndim == 4:
                position_texture = position_texture[0]
            pointmap_np = self._pointmap_from_position_texture(
                position_texture, c2w_gl_np,
            )
            pointmap_tensor = torch.from_numpy(pointmap_np).float().cuda()

            if self.args.use_plucker:
                # PluckerEmbedder expects batched torch tensors on GPU.
                intrinsics_t = torch.from_numpy(K_np).unsqueeze(0).float().cuda()
                cam_to_world_t = torch.from_numpy(c2w_gl_np).unsqueeze(0).float().cuda()
                with torch.no_grad():
                    plucker_data = self.plucker_embedder(intrinsics_t, cam_to_world_t)
                    pl = plucker_data["plucker"][0]
                plucker_tensor = einops.rearrange(pl, 'h w c -> c h w')
            else:
                _, H, W = rgb_tensor.shape
                plucker_tensor = torch.zeros(6, H, W, device=rgb_tensor.device)

            # Deterministic center crop 256 -> 224 (mirrors dataloader's
            # PairedRandomCrop but with fixed offsets at eval).
            rgb_c = rgb_tensor[:, top:top + dst, left:left + dst]
            pointmap_c = pointmap_tensor[:, top:top + dst, left:left + dst]
            plucker_c = plucker_tensor[:, top:top + dst, left:left + dst]
            K_c = adjust_intrinsic(K_np, top, left)

            cam_images.append(torch.cat([rgb_c, plucker_c], dim=0))  # (9, 224, 224)
            cam_pointmaps.append(pointmap_c)                          # (3, 224, 224)
            cam_extrs.append(torch.from_numpy(c2w_gl_np).float().cuda())
            cam_intrs.append(torch.from_numpy(K_c).float().cuda())

        # qpos from env state
        st = self.env.unwrapped.get_state_dict()
        for key in ("panda", "panda_wristcam", "panda_stick"):
            if key in st['articulations']:
                qpos = st['articulations'][key][0, 13:22]
                break
        state_vector = qpos.cpu().numpy()
        # eef_xyz from agent tcp
        tcp_pos = self.env.unwrapped.agent.tcp_pos
        if hasattr(tcp_pos, 'detach'):
            tcp_pos = tcp_pos.detach().cpu().numpy()
        eef_xyz = np.asarray(tcp_pos, dtype=np.float32).reshape(-1)[:3]

        if drop_proprio:
            state_vector = np.zeros_like(state_vector)
            eef_xyz = np.zeros_like(eef_xyz)
        norm_state = (
            state_vector - self.norm_stats["state_mean"].cpu().numpy()
        ) / self.norm_stats["state_std"].cpu().numpy()

        batch = {
            'image':              torch.stack(cam_images, dim=0).unsqueeze(0),
            'pointmap':           torch.stack(cam_pointmaps, dim=0).unsqueeze(0),
            'qpos':               torch.from_numpy(norm_state).float().unsqueeze(0).cuda(),
            'eef_xyz':            torch.from_numpy(eef_xyz).float().unsqueeze(0).cuda(),
            'cam_extrinsics_full': torch.stack(cam_extrs, dim=0).unsqueeze(0),
            'cam_intrinsics_full': torch.stack(cam_intrs, dim=0).unsqueeze(0),
        }
        return batch, per_cam_full_rgb

    def _build_canonical_step(self, cam_names, drop_proprio):
        """Per-step batch construction for the canonical-view path.

        Renders each chosen input cam (rgb + position texture), fuses the
        per-pixel world points, and splats into self.num_side_cam fixed
        canonical viewpoints. Returns the same batch dict shape as
        _build_articubot_step but with canonical_rgbs in 'image' (no
        plucker, no real pointmap) and the canonical extrinsics/intrinsics.
        """
        top, left, dst = self.crop_top, self.crop_left, self.crop_dst
        self.env.unwrapped.scene.update_render(
            update_sensors=False, update_human_render_cameras=True,
        )

        input_pts_worlds, input_rgbs_uint8 = [], []
        for cam_name in cam_names:
            camera = self.env.unwrapped.scene.human_render_cameras[cam_name]
            params = camera.get_params()
            c2w_gl_np = _to_numpy(params["cam2world_gl"]).astype(np.float32)
            if c2w_gl_np.ndim == 3:
                c2w_gl_np = c2w_gl_np[0]
            camera.capture()
            obs_dict = camera.get_obs(
                rgb=True, depth=False, position=True,
                segmentation=False, normal=False, albedo=False,
            )
            rgb_raw = obs_dict["rgb"]
            if hasattr(rgb_raw, 'detach'):
                rgb_raw = rgb_raw.detach().cpu().numpy()
            if rgb_raw.ndim == 4:
                rgb_raw = rgb_raw[0]
            rgb_uint8 = (
                rgb_raw[..., :3].astype(np.uint8) if rgb_raw.dtype != np.float32
                else (rgb_raw[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
            )
            position_texture = obs_dict["position"]
            if position_texture.ndim == 4:
                position_texture = position_texture[0]
            pts_world = self._pointmap_from_position_texture(
                position_texture, c2w_gl_np,
            )
            input_pts_worlds.append(pts_world)
            input_rgbs_uint8.append(rgb_uint8)

        canonical_rgbs = fuse_and_render_from_world_points(
            input_pts_worlds, input_rgbs_uint8,
            self.canonical_w2cs_cv, self.canonical_Ks,
            H=self.H, W=self.W,
        )

        cam_images, cam_pointmaps, cam_extrs, cam_intrs = [], [], [], []
        for i, canonical_rgb in enumerate(canonical_rgbs):
            rgb_tensor = einops.rearrange(
                torch.from_numpy(canonical_rgb).float() / 255.0,
                'h w c -> c h w',
            ).cuda()
            rgb_c = rgb_tensor[:, top:top + dst, left:left + dst]
            K_c = adjust_intrinsic(self.canonical_Ks[i], top, left)
            _, Hc, Wc = rgb_c.shape
            zero_plucker = torch.zeros(6, Hc, Wc, device='cuda')
            cam_images.append(torch.cat([rgb_c, zero_plucker], dim=0))
            cam_pointmaps.append(torch.zeros(3, Hc, Wc, device='cuda'))
            cam_extrs.append(
                torch.from_numpy(self.canonical_c2ws_gl[i]).float().cuda()
            )
            cam_intrs.append(torch.from_numpy(K_c).float().cuda())

        # qpos + eef_xyz
        st = self.env.unwrapped.get_state_dict()
        for key in ("panda", "panda_wristcam", "panda_stick"):
            if key in st['articulations']:
                qpos = st['articulations'][key][0, 13:22]
                break
        state_vector = qpos.cpu().numpy()
        tcp_pos = self.env.unwrapped.agent.tcp_pos
        if hasattr(tcp_pos, 'detach'):
            tcp_pos = tcp_pos.detach().cpu().numpy()
        eef_xyz = np.asarray(tcp_pos, dtype=np.float32).reshape(-1)[:3]
        if drop_proprio:
            state_vector = np.zeros_like(state_vector)
            eef_xyz = np.zeros_like(eef_xyz)
        norm_state = (
            state_vector - self.norm_stats["state_mean"].cpu().numpy()
        ) / self.norm_stats["state_std"].cpu().numpy()

        batch = {
            'image':              torch.stack(cam_images, dim=0).unsqueeze(0),
            'pointmap':           torch.stack(cam_pointmaps, dim=0).unsqueeze(0),
            'qpos':               torch.from_numpy(norm_state).float().unsqueeze(0).cuda(),
            'eef_xyz':            torch.from_numpy(eef_xyz).float().unsqueeze(0).cuda(),
            'cam_extrinsics_full': torch.stack(cam_extrs, dim=0).unsqueeze(0),
            'cam_intrinsics_full': torch.stack(cam_intrs, dim=0).unsqueeze(0),
        }
        return batch, canonical_rgbs
    
    def evaluate(self, policy, save_path, video_prefix, pose_name, episode_num=0):
        np.random.seed(episode_num)
        self.env.reset(seed=episode_num)

        if self.args.default_cam:
            cam_names = ["render_camera"] * self.num_side_cam
        elif self.num_side_cam == 1:
            idx = episode_num if pose_name == 'train' else 500 + episode_num
            cam_names = [f'cam_{idx}']
        else:
            base = 0 if pose_name == 'train' else 500
            cam_names = [f'cam_{base + 2 * episode_num}', f'cam_{base + 2 * episode_num + 1}']
        print(f"Episode {episode_num}: Using cameras {cam_names}")

        camera_frames, success_labels, rewards, success = [], [], [], []
        done = False
        step = 0
        has_succeeded = False
        
        while not done and step < self.max_steps:
            drop_proprio = bool(np.random.rand() < self.args.prob_drop_proprio)

            if self.use_canonical_views:
                batch, canonical_rgbs = self._build_canonical_step(cam_names, drop_proprio)
                combined = canonical_rgbs[0] if len(canonical_rgbs) == 1 else np.concatenate(canonical_rgbs, axis=1)
                camera_frames.append(combined)
                success_labels.append(has_succeeded)
            elif self.is_articubot:
                batch, per_cam_full_rgb = self._build_articubot_step(cam_names, drop_proprio)
                combined = per_cam_full_rgb[0] if len(per_cam_full_rgb) == 1 else np.concatenate(per_cam_full_rgb, axis=1)
                camera_frames.append(combined)
                success_labels.append(has_succeeded)
            else:
                per_cam_images = [self.env.unwrapped.render_rgb_array(n).cpu().numpy()[0] for n in cam_names]
                combined = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
                camera_frames.append(combined)
                success_labels.append(has_succeeded)

                per_cam_tensors = []
                for n, img in zip(cam_names, per_cam_images):
                    rgb_tensor = einops.rearrange(torch.from_numpy(img).float() / 255.0, 'h w c -> c h w')
                    if self.args.use_plucker:
                        camera = self.env.unwrapped.scene.human_render_cameras[n]
                        intrinsics_tensor = camera.get_params()["intrinsic_cv"]
                        cam_to_world_tensor = camera.get_params()["cam2world_gl"]
                        with torch.no_grad():
                            plucker_data = self.plucker_embedder(intrinsics_tensor, cam_to_world_tensor)
                            plucker_tensor = einops.rearrange(plucker_data['plucker'][0].cpu(), 'h w c -> c h w')
                    else:
                        plucker_tensor = torch.zeros(6, rgb_tensor.shape[1], rgb_tensor.shape[2])
                    per_cam_tensors.append(torch.cat([rgb_tensor, plucker_tensor], dim=0))

                image_tensor = torch.stack(per_cam_tensors, dim=0).unsqueeze(0).cuda()
                st = self.env.unwrapped.get_state_dict()
                for key in ("panda", "panda_wristcam", "panda_stick"):
                    if key in st['articulations']:
                        qpos = st['articulations'][key][0, 13:22]
                        break
                state_vector = qpos.cpu().numpy()
                if drop_proprio:
                    state_vector = np.zeros_like(state_vector)
                normalized_state = (state_vector - self.norm_stats["state_mean"].cpu().numpy()) / self.norm_stats["state_std"].cpu().numpy()
                state_tensor = einops.rearrange(torch.tensor(normalized_state, device="cuda").float(), 'd -> 1 d')
                batch = {'qpos': state_tensor, 'image': image_tensor}

            with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if self.args.use_fp16 else nullcontext()):
                action_chunk = policy(batch)
            action_chunk = action_chunk[0].float().cpu().numpy() * self.norm_stats["action_std"].cpu().numpy() + self.norm_stats["action_mean"].cpu().numpy()
            
            # Execute action chunk
            for i in range(action_chunk.shape[0]):                
                if done or step >= self.max_steps:
                    break

                res = self.env.step(torch.tensor(action_chunk[i], device='cuda'))
                obs, reward, terminated, truncated, info = res
                done = bool(terminated or truncated)
                current_success = info['success'][0].item() if isinstance(info['success'], torch.Tensor) else bool(info['success'])
                has_succeeded = has_succeeded or current_success
                rewards.append(float(reward))
                success.append(current_success)
                step += 1
                
                if episode_num < self.eval_save_n_video and i < action_chunk.shape[0] - 1:
                    per_cam_images = [self.env.unwrapped.render_rgb_array(n).cpu().numpy()[0] for n in cam_names]
                    combined = per_cam_images[0] if len(per_cam_images) == 1 else np.concatenate([per_cam_images[0], per_cam_images[1]], axis=1)
                    camera_frames.append(combined)
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

def main():
    """Main function to run dataset replay or policy evaluation"""
    dataset_path = '/share/data/ripl/tianchong/projects/CamPoseRobosuite/demos/lift/ph/low_dim_v141.hdf5'
    output_dir = '/share/data/ripl/tianchong/projects/CamPoseRobosuite/evaluation_results'
    # camera_name will be retrieved from args.camera_names[0]
    


if __name__ == '__main__':
    main()
