#!/bin/bash
#SBATCH --job-name=o96_throttle
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

# Sentinel file: rank 0 touches this when training completes; dummy ranks poll it to exit.
export SENTINEL="/tmp/throttle_done_${SLURM_JOB_ID}"
export SUBMIT_DIR="$PWD"
rm -f "$SENTINEL"

# Single srun step: no resource contention between steps.
# CUDA_VISIBLE_DEVICES=$SLURM_LOCALID pins each rank to its own GPU (0-3).
# Rank 0: waits 30s for dummies to reach steady state, then runs 1-GPU training.
# Ranks 1-3: run continuous BF16 matmul until rank 0 signals completion.
srun bash -c '
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

if [[ "$SLURM_PROCID" == "0" ]]; then
    echo "Rank 0: sleeping 30s for dummy loads to reach thermal steady state..." >&2
    sleep 30
    echo "Rank 0: starting 1-GPU training on GPU $SLURM_LOCALID" >&2

    # Override all env vars Lightning/NCCL use to detect multi-GPU mode
    export WORLD_SIZE=1
    export LOCAL_WORLD_SIZE=1
    export SLURM_NTASKS=1
    export SLURM_NTASKS_PER_NODE=1
    export SLURM_NPROCS=1
    export RANK=0
    export LOCAL_RANK=0
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500

    cd "$SUBMIT_DIR/pretraining"

    TRAIN_START=$(date +%s)
    $HOME/myutils/entrypoint.sh \
        python -X faulthandler -m anemoi.training profile \
        --config-name=pretraining_o96.yaml
    TRAIN_END=$(date +%s)
    echo "⏱ TRAINING TIME: $((TRAIN_END - TRAIN_START)) seconds"

    touch "$SENTINEL"

else
    echo "Rank $SLURM_PROCID: dummy load on GPU $SLURM_LOCALID" >&2
    python -c "
import torch, os
a = torch.ones(16384, 4096, dtype=torch.bfloat16, device=\"cuda\")
b = torch.ones(4096, 16384, dtype=torch.bfloat16, device=\"cuda\")
sentinel = os.environ[\"SENTINEL\"]
print(\"Dummy GPU load active rank \" + os.environ[\"SLURM_PROCID\"], flush=True)
while not os.path.exists(sentinel):
    torch.mm(a, b)
    torch.cuda.synchronize()
print(\"Dummy rank \" + os.environ[\"SLURM_PROCID\"] + \" done.\", flush=True)
"
fi
'

rm -f "$SENTINEL"
echo "Throttle test complete."
