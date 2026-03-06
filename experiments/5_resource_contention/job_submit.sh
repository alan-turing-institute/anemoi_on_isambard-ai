#!/bin/bash
#SBATCH --job-name=o96_contention
#SBATCH --nodes=1
##SBATCH --nodelist=nid011191
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load cuda/12.6
module load gcc-native/13.2
module load brics/nccl
export NCCL_CROSS_NIC=1

eval "$(mamba shell hook --shell bash)"
FORGE_ENV_PATH="/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/conda_env/aifs_gh200_base"
mamba activate "${FORGE_ENV_PATH}"

export CC=$(which gcc)
export CXX=$(which g++)
export PYTHONUNBUFFERED=1
export FI_CXI_DEFAULT_VNI=$((SLURM_JOB_ID % 10000))
export HYDRA_FULL_ERROR=1

export SENTINEL="/tmp/contention_done_${SLURM_JOB_ID}"
export SUBMIT_DIR="$PWD"
rm -f "$SENTINEL"


srun bash -c '
# Capture original SLURM identity BEFORE any overrides.
# GPU_ID and PROC_ID are still needed for port assignment and output dirs.
GPU_ID=$SLURM_LOCALID
PROC_ID=$SLURM_PROCID

# Pin to physical GPU. CUDA_VISIBLE_DEVICES must be set before SLURM_LOCALID
# is overridden, otherwise the remapped cuda:0 points to the wrong device.
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Override ALL SLURM/torch rank vars to zero so Lightning treats every
# process as an independent single-GPU rank-0 trainer.
# Critical: SLURM_PROCID must also be 0 — Lightning reads it directly
# for internal rank detection, causing non-zero ranks to block before
# moving the model to GPU even when RANK=0 and WORLD_SIZE=1.
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1
export SLURM_NPROCS=1
export SLURM_PROCID=0
export SLURM_LOCALID=0
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + GPU_ID))

cd "$SUBMIT_DIR/pretraining"
TRAIN_START=$(date +%s)
$HOME/myutils/entrypoint.sh \
    python -X faulthandler -m anemoi.training profile \
    --config-name=pretraining_o96.yaml \
    hydra.run.dir="outputs/rank_${PROC_ID}" \
    hardware.num_gpus_per_node=1
TRAIN_END=$(date +%s)
echo "Proc $PROC_ID TRAINING TIME: $((TRAIN_END - TRAIN_START)) seconds"
'

rm -f "$SENTINEL"
echo "Contention test complete."
