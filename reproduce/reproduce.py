import argparse
from pathlib import Path
import shlex
import subprocess

import yaml


REPRODUCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRODUCE_DIR.parent
MANIFEST_PATH = REPRODUCE_DIR / "paper_runs.yaml"


def load_manifest():
    with MANIFEST_PATH.open() as f:
        return yaml.safe_load(f)


def build_command(manifest, paper_item, exp, seed):
    experiment_id = manifest["paper_items"][paper_item][exp]
    experiment = manifest["experiments"][experiment_id]
    experiment["seeds"].index(seed)
    name = f"{paper_item}_{exp}_seed_{seed}"
    command = ["python", experiment["entrypoint"], "--name", name, "--seed", str(seed)]
    for key, value in experiment["overrides"].items():
        command.extend([f"--{key}", str(value)])
    return command


def main():
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description="Run exactly one paper experiment.")
    parser.add_argument("--paper_item", required=True, choices=manifest["paper_items"])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    command = build_command(manifest, args.paper_item, args.exp, args.seed)
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
