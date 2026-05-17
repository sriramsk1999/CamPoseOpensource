"""Build a `joint_abs` action-space dataset from an existing `eef_delta` one.

Reads the source hdf5 (e.g. ``squarerand_eef_delta.hdf5``) and emits a sibling
``squarerand_joint_abs.hdf5`` where every demo's ``actions`` array carries
**absolute joint-position targets** instead of end-effector deltas:

    new_actions[t] = [ env.sim.data.qpos[arm_joints] at step t+1,
                       eef_delta_actions[t, gripper_dim] ]    # shape (8,)

The arm-joint targets are read directly from the MuJoCo state at step t+1
(the state the policy *was supposed to reach* after action t), giving an
absolute joint trajectory that a JOINT_POSITION controller can track. Gripper
is copied through unchanged from the original action.

We leave ``states`` and ``env_args.env_kwargs`` untouched except for swapping
the controller in env_kwargs to ``JOINT_POSITION`` (mirroring the convention
of the existing ``liftrand_joint_abs.hdf5``). The dataset's ``action_space``
attribute is set to ``joint_abs`` so train.py picks the right env wrapping.

Why we do it this way:
- MolmoBot expects absolute joint_pos actions (paper's "absolute substantially
  outperforms delta" finding from real-world ablations).
- No env replay is required for correctness — we just relabel actions from
  the existing state trajectory. The relabel is exact for any controller that
  tracks joint targets faithfully (which is what JOINT_POSITION does).

Usage:
    python script_robosuite_demos/build_joint_abs_dataset.py \\
        --src policy_robosuite/demos/squarerand_eef_delta.hdf5 \\
        --dst policy_robosuite/demos/squarerand_joint_abs.hdf5
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# Arm-joint indices into the MuJoCo qpos vector. Match
# policy_robosuite/utils.py:get_norm_stats which reads ``states[:, 1:8]``.
ARM_QPOS_SLICE = slice(1, 8)
N_ARM_JOINTS = 7
N_ACTION_DIMS = 8  # 7 arm + 1 gripper


def _load_template_joint_abs_controller():
    """Pull a known-good JOINT_POSITION controller spec from
    ``liftrand_joint_abs.hdf5``. Mirroring an existing joint_abs demo guarantees
    we get the gains / impedance settings that have been validated on the same
    robosuite version we're running.
    """
    demos_dir = Path(__file__).resolve().parent.parent / "policy_robosuite" / "demos"
    template_path = demos_dir / "liftrand_joint_abs.hdf5"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template demo {template_path} not found — needed to source the "
            f"JOINT_POSITION controller config. Either generate a joint_abs demo "
            f"first or hard-code the controller spec."
        )
    with h5py.File(template_path, "r") as f:
        ea = f["data"].attrs["env_args"]
        if isinstance(ea, bytes):
            ea = ea.decode()
        env_config = json.loads(ea)
    return env_config["env_kwargs"]["controller_configs"]


def relabel_demo(actions_eef: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Convert an (T, 7) eef_delta action sequence + (T, ...) state sequence
    into an (T, 8) joint_abs action sequence.

    actions_eef[..., -1] is the gripper command. We copy it through. The first
    7 dims of the new action are the arm joint angles **at the next state** —
    i.e. the joint configuration the policy should achieve after taking this
    action.
    """
    T = actions_eef.shape[0]
    assert states.shape[0] == T, (
        f"action / state length mismatch: actions={T}, states={states.shape[0]}"
    )
    new_actions = np.zeros((T, N_ACTION_DIMS), dtype=np.float32)

    # Joint targets = state at t+1; last step targets stay-where-you-are.
    next_states = np.concatenate([states[1:], states[-1:]], axis=0)
    new_actions[:, :N_ARM_JOINTS] = next_states[:, ARM_QPOS_SLICE].astype(np.float32)
    # Carry the original gripper command through unchanged.
    new_actions[:, N_ARM_JOINTS] = actions_eef[:, -1].astype(np.float32)
    return new_actions


def build_joint_abs_dataset(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination exists, refusing to overwrite: {dst}")

    print(f"Source: {src}")
    print(f"Destination: {dst}")

    joint_abs_controller = _load_template_joint_abs_controller()

    with h5py.File(src, "r") as f_src:
        src_data = f_src["data"]
        ea = src_data.attrs["env_args"]
        if isinstance(ea, bytes):
            ea = ea.decode()
        env_config = json.loads(ea)
        # Swap controller to JOINT_POSITION (rest of env_kwargs unchanged).
        env_config["env_kwargs"]["controller_configs"] = joint_abs_controller
        new_env_args = json.dumps(env_config)

        demo_keys = sorted(
            [k for k in src_data.keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        n_demos = len(demo_keys)
        print(f"Found {n_demos} demos.")

        with h5py.File(dst, "w") as f_dst:
            dst_data = f_dst.create_group("data")
            dst_data.attrs["env_args"] = new_env_args
            dst_data.attrs["action_space"] = "joint_abs"
            for k, v in src_data.attrs.items():
                if k not in ("env_args", "action_space") and k not in dst_data.attrs:
                    dst_data.attrs[k] = v

            for demo_key in tqdm(demo_keys, desc="Relabeling demos"):
                src_demo = src_data[demo_key]
                states = src_demo["states"][()]
                actions_eef = src_demo["actions"][()]
                actions_joint = relabel_demo(actions_eef, states)

                dst_demo = dst_data.create_group(demo_key)
                dst_demo.create_dataset("states", data=states)
                dst_demo.create_dataset("actions", data=actions_joint)
                # Copy any extra per-demo keys (rewards, dones, obs_*) through.
                for k in src_demo.keys():
                    if k in ("states", "actions"):
                        continue
                    src_demo.copy(k, dst_demo)
                for k, v in src_demo.attrs.items():
                    dst_demo.attrs[k] = v

    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to source eef_delta hdf5")
    parser.add_argument("--dst", required=True, help="Path to write joint_abs hdf5")
    args = parser.parse_args()
    build_joint_abs_dataset(args.src, args.dst)


if __name__ == "__main__":
    main()
