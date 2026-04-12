import os

import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (32768, 32768))

import json
import time
import numpy as np
import torch
import argparse
import tqdm
import json
from contextlib import nullcontext
import gymnasium as gym
from pathlib import Path

from utils import load_data, compute_dict_mean, set_seed, detach_dict, cosine_schedule, cosine_schedule_with_warmup, constant_schedule, cleanup_ckpt, get_last_ckpt, save_image_batch_as_mp4
from models.act import ACTPolicy
from models.dp import DiffusionPolicy
from models.smolvla import SmolVLAPolicyWrapper
from eval import Evaluator

import sys as _sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_sys.path.insert(0, _REPO_ROOT)
from campose_wrappers.articubot_dit import ArticubotDiTWrapper

import wandb

torch.backends.cuda.enable_flash_sdp(True)
print(torch.backends.cuda.is_flash_attention_available())

def main(args, ckpt=None):
    set_seed(args.seed)
    start_time = time.time()

    # Read env id and control mode from ManiSkill dataset JSON
    dataset_json_path = os.path.join(args.dataset_path, 'trajectory.json')
    with open(dataset_json_path, 'r') as f:
        dataset_meta = json.load(f)
    env_id = dataset_meta['env_info']['env_id']
    control_mode = dataset_meta['env_info']['env_kwargs']['control_mode']
    args.action_dim = 7 if 'ee' in control_mode else 8
    # ManiSkill Panda qpos slice is length 9 (indices 13:22)
    args.obs_dim = 9

    env_kwargs = {
        'obs_mode': None,
        'control_mode': control_mode,
        'render_mode': "rgb_array",
        'sim_backend': 'gpu',
        'max_episode_steps': args.eval_max_steps,
    }

    if 'Rand' in env_id:
        env_kwargs['fixed'] = args.original

    # Build ManiSkill environment
    env = gym.make(env_id, **env_kwargs)
    env.reset()
    
    train_dataloader, val_dataloader, stats = load_data(
        args=args,
        env=env
    )

    # Save stats
    os.makedirs(args.ckpt_dir, exist_ok=True)
    stats_path = os.path.join(args.ckpt_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=4)

    evaluator = Evaluator(env=env, norm_stats=stats, args=args)

    # Create policy and optimizer based on selected policy_class
    if args.policy_class == 'act':
        policy = ACTPolicy(args).cuda()
    elif args.policy_class == 'dp':
        policy = DiffusionPolicy(args).cuda()
    elif args.policy_class == 'smolvla':
        policy = SmolVLAPolicyWrapper(args).cuda()
    elif args.policy_class == 'articubot_dit':
        # Maniskill Panda: eef_xyz(3) + qpos(9) = 12-dim state.
        policy = ArticubotDiTWrapper(
            args=args,
            state_dim=3 + 9,
            action_dim=args.action_dim,
            num_cams=args.num_side_cam,
            image_size=224,
            norm_stats=stats,
        ).cuda()
    else:
        raise ValueError(f"Unsupported policy_class: {args.policy_class}")

    optimizer = policy.configure_optimizers()
    # Learning rate schedule: cosine with warmup for SmolVLA, constant otherwise
    steps_per_epoch = len(train_dataloader)
    total_steps = args.num_epochs * steps_per_epoch
    if args.lr_scheduler == 'cosine':
        warmup_steps = int(0.05 * total_steps)
        scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = constant_schedule(optimizer)
    
    epoch = 0
    if ckpt is not None:
        policy.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch = ckpt['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")
    
    pbar = tqdm.tqdm(total=args.num_epochs, desc="Training")
    pbar.update(epoch)
    
    while epoch < args.num_epochs:
        # Validation
        if epoch % 10 == 0:
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for data in val_dataloader:
                    with torch.autocast("cuda", dtype=torch.bfloat16) if args.use_fp16 else nullcontext():
                        forward_dict = policy(data)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                for k, v in epoch_summary.items():
                    wandb.log({f'val_{k}': v.item()}, step=epoch)

        # Evaluation
        if epoch % args.eval_every == 0:
            policy.eval()
            for pose_name in ['train', 'test']:
                eval_save_path = os.path.join(args.ckpt_dir, f"eval_epoch_{epoch}_{pose_name}")
                os.makedirs(eval_save_path, exist_ok=True)
                evaluator.success_by_seed = {}

                success_rates = []
                for episode_idx in range(50 if epoch > args.eval_start_epoch else args.eval_episodes):
                    with torch.no_grad():
                        _, success_rate, _ = evaluator.evaluate(
                            policy=policy,
                            save_path=eval_save_path,
                            video_prefix=f"epoch_{epoch}_episode_{episode_idx}",
                            pose_name=pose_name,
                            episode_num=episode_idx
                        )
                    success_rates.append(success_rate)
                avg_success_rate = sum(success_rates) / len(success_rates)
                wandb.log({f'success_rate_{pose_name}': avg_success_rate}, step=epoch)
                with open(os.path.join(eval_save_path, 'success_by_seed.json'), 'w') as f:
                    json.dump(evaluator.success_by_seed, f, indent=2)

        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.ckpt_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch, 
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_summary['loss'], 
                'wandb_id': wandb.run.id
            }, checkpoint_path)
            cleanup_ckpt(args.ckpt_dir, keep=3)  # Keep last 3 checkpoints

            # if time.time() - start_time > 7.5 * 60 * 60:
            #     print(f"⏰ Time limit reached ({(time.time() - start_time)/3600:.1f} hours). Exiting...")
            #     break
        # Training
        train_history = []
        policy.train()
        for data in train_dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16) if args.use_fp16 else nullcontext():
                forward_dict = policy(data)
                loss = forward_dict['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            train_history.append(detach_dict(forward_dict))
        
        epoch_summary = compute_dict_mean(train_history)
        for k, v in epoch_summary.items():
            wandb.log({f'train_{k}': v.item()}, step=epoch)

        pbar.update(1)
        epoch += 1

    pbar.close()
    env.close()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train policy for ManiSkill environments (ACT/DP/SmolVLA)")
    def str2bool(v): return str(v) == "1"

    # General config
    parser.add_argument('--name', type=str, default='test_maniskill', help='name for the run')
    parser.add_argument('--dataset_dir', type=str, 
                        default=None,
                        help='Path to demos root directory (absolute). If None, defaults to policy_maniskill/demos')
    parser.add_argument('--dataset_suffix', type=str, default='push_rand', help='dataset subfolder under demos/')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Path to checkpoints directory (absolute). If None, defaults to policy_maniskill/checkpoints/<name>')
    parser.add_argument('--policy_class', type=str, default='act', choices=['dp','act','smolvla','articubot_dit'], help='policy class')
    parser.add_argument('--use_pointmaps', default=False, type=str2bool, help='render pointmaps alongside RGB for RoPE4D policies')
    parser.add_argument('--horizon', default=16, type=int, help='action horizon for flow-matching DiT policies')
    parser.add_argument('--n_action_steps', default=8, type=int, help='number of action steps executed per inference')

    parser.add_argument('--num_episodes', default=200, type=int, help='num_episodes')
    parser.add_argument('--use_plucker', default=True, type=str2bool, help='use Plucker embeddings')

    # Camera pose config
    parser.add_argument('--n', type=int, default=3, help='Number of cameras per window W(i) = [m*i, m*i + n)')
    parser.add_argument('--m', type=int, default=1, help='Stride for camera window W(i) = [m*i, m*i + n)')
    parser.add_argument('--num_side_cam', type=int, default=1, choices=[1,2], help='Number of side cams to use (1 or 2)')
    parser.add_argument('--default_cam', type=str2bool, default=False, help='When true, use default agentview pose for all cameras (duplicate if >1)')

    # Training config
    parser.add_argument('--batch_size', default=70, type=int, help='batch_size')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--num_epochs', default=30_001, type=int, help='num_epochs')
    parser.add_argument('--eval_start_epoch', type=int, default=20_000, help='start evaluating 50 at this epoch')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')
    parser.add_argument('--lr_scheduler', type=str, default='const', help='lr scheduler: const, cosine')
    parser.add_argument('--save_every', type=int, default=10000, help='save checkpoint every N epochs')
    parser.add_argument('--use_fp16', default=True, type=str2bool, help='use mixed precision bf16 training')
    
    # Dataloader config
    parser.add_argument('--transform', type=str, default='crop', choices=['crop', 'id', 'crop_jitter'], 
                        help='Image transformation type')
    parser.add_argument('--prob_drop_proprio', default=1., type=float, help='probability to drop proprio')
    parser.add_argument('--use_cam_pose', default=False, type=bool, help='otherwise mask to 0')
    parser.add_argument('--original', default=False, type=str2bool, help='visually same as original lift')


    # ACT model config
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--latent_drop_prob', type=float, default=0.0, help='drop probability for RGB latents in backbone')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='KL Weight')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk_size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--obs_dim', type=int, default=7, help='observation dimension')
    
    
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--ffn_dim', type=int, default=2048, help='feedforward network dimension')
    parser.add_argument('--enc_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='number of decoder layers')
    parser.add_argument('--pre_norm', type=bool, default=True, help='use pre-normalization')
    parser.add_argument('--activation', default='relu', help='activation function')

    # Backbone config
    parser.add_argument('--backbone', default='late_imagenet', help='backbone: resnet, linear')
    parser.add_argument('--patch_size', type=int, default=8, help='patch size')

    # Evaluation config
    parser.add_argument('--eval_every', type=int, default=1000, help='evaluate every N epochs')
    parser.add_argument('--eval_episodes', type=int, default=10, help='number of evaluation episodes')
    parser.add_argument('--eval_max_steps', type=int, default=300, help='max steps per evaluation episode')
    parser.add_argument('--eval_save_n_video', type=int, default=10, help='save the first n videos for each evaluation epoch')

    # SmolVLA finetuning flags
    parser.add_argument('--freeze_vision_encoder', type=str2bool, default=False, help='freeze the vision encoder (SigLIP)')
    parser.add_argument('--train_expert_only', type=str2bool, default=False, help='train only the action expert; freeze VLM')

    args = parser.parse_args()

    group = args.name[:-7] # remove the seed from the name

    # Anchor paths under the module root (policy_maniskill)
    MODULE_ROOT = Path(__file__).resolve().parent

    # dataset_dir -> dataset_path
    if args.dataset_dir is None:
        args.dataset_dir = str((MODULE_ROOT / "demos").resolve())
    if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
        args.ckpt_dir = str((MODULE_ROOT / "checkpoints" / args.name).resolve())
    args.dataset_path = os.path.join(args.dataset_dir, args.dataset_suffix)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.ckpt_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Check for existing checkpoint to resume
    ckpt_path = get_last_ckpt(args.ckpt_dir)
    
    wandb.init(project='CamPose_training', name=args.name, config=vars(args), group=group)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        main(args, ckpt)
    else:
        print(f"Starting new training run: {args.name}")
        main(args)
