#!/bin/bash
#SBATCH --job-name=nccl_perf
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=logs/nccl_fix_%Nn_%j.out
#SBATCH --exclusive

mkdir -p logs
set -euo pipefail

# 1. Load System Stack
module purge
module load PrgEnv-nvidia
module load cudatoolkit/24.11_12.6
module load craype-accel-nvidia90
module load craype-network-ofi
module load cray-mpich
module load brics/nccl
module load brics/aws-ofi-nccl

# 2. Critical MPI/Network Exports
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_CXI_DEFAULT_VNI=$((SLURM_JOB_ID % 10000))
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=2

# 3. Run
BIN_PATH="$(pwd)/nccl-tests/build/all_reduce_perf"

echo "====================================================="
echo "Running with --mpi=cray_shasta to fix Rank=1 issue"
echo "====================================================="

# Try 'cray_shasta' first. If that fails, 'pmi2' is usually the backup.
srun --label --mpi=cray_shasta \
    $BIN_PATH \
    -b 512M -e 8G -f 2 \
    -g 1 \
    -c 1 \
    -n 50
