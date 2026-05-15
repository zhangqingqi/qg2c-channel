#!/bin/zsh
# Single-case GPU run via the JAX backend.
# Edit the python invocation below for your case.
#SBATCH --job-name=qg2c_jax
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue
#SBATCH --output=outputs/logs/jax_%j.out
#SBATCH --error=outputs/logs/jax_%j.err

source ~/.zshrc
mamba activate neuralgcm

set -eo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "== node:   $(hostname)"
echo "== gpu:    $(/usr/bin/nvidia-smi -L 2>&1 | head -1)"
echo "== start:  $(date -Iseconds)"

python -u src/run_single.py \
    --amp 4e4 --r 0.10 --r1 0.04 --r2 0.04 \
    --delta 0.5 --Rd 20 --beta 8.64e-4 \
    --seed 1 --init-style matlab-sin-noise --noise-amp 1e-3 \
    --nx 512 --ny 256 \
    --tmax 500 --dt 0.00048828125 --dt-save-fields 2.0 \
    --backend jax \
    --detailed --analyze --force \
    --run-dir outputs/runs \
    --label run_filament_Rd20

echo "== done $(date -Iseconds)"
