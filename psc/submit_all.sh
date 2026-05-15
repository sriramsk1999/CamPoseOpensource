#!/bin/bash
# Convenience: submit all 6 baselines from the psc/ directory.
# Usage: bash psc/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")"
sbatch act_dino_sv.slurm
sbatch dit_dino_sv_plucker.slurm
sbatch dit_dino_sv_canon.slurm
sbatch flow_matching_3dfa.slurm
sbatch dit_maniwhere_sv.slurm
sbatch dit_rope4d_dino_cv.slurm
echo "All 6 jobs submitted. Logs will land in psc/logs/."
