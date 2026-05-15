#!/bin/zsh
# Single-case CPU run via the scipy/numpy reference solver.
# Edit the python invocation below for your case.
#SBATCH --job-name=qg2c_cpu
#SBATCH --partition=huce_cascade
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/cpu_%j.out
#SBATCH --error=outputs/logs/cpu_%j.err

source ~/.zshrc
mamba activate neuralgcm

set -eo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "== node:   $(hostname)"
echo "== start:  $(date -Iseconds)"

python -u src/run_single.py \
    --amp 4e4 --r 0.10 --r1 0.10 --r2 0.10 \
    --delta 0.2 --Rd 40 --beta 1.728e-3 \
    --seed 0 \
    --nx 128 --ny 64 \
    --tmax 500 --dt 0.0078125 --dt-save-fields 5.0 \
    --backend scipy \
    --detailed --analyze --force \
    --run-dir outputs/runs \
    --label run_cpu_baseline

echo "== done $(date -Iseconds)"
