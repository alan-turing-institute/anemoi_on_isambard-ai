#!/bin/bash
#SBATCH --job-name=compile_nccl
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/compile_nccl.out

set -euo pipefail

# 1. Clean environment
module purge

# 2. Load Compiler & Wrappers
module load PrgEnv-nvidia
module load cudatoolkit/24.11_12.6
module load craype-accel-nvidia90

# 3. Load Network & MPI
module load craype-network-ofi
module load cray-mpich

# 4. Load NCCL (THE MISSING PIECE)
# We load the default brics module which contains the library and headers
module load brics/nccl

# 5. Path Detection
# We need to tell the Makefile where NCCL lives. 
# Brics modules usually set NCCL_DIR or NCCL_HOME. We unify them.
if [ -n "${NCCL_DIR:-}" ]; then
    export NCCL_HOME=$NCCL_DIR
elif [ -n "${NCCL_ROOT:-}" ]; then
    export NCCL_HOME=$NCCL_ROOT
else
    # Fallback: Try to find where libnccl.so lives from the loaded module
    LIB_PATH=$(dirname $(find $(dirname $(dirname $(which nvcc))) -name "libnccl.so" | head -n 1))
    export NCCL_HOME=$(dirname $LIB_PATH)
fi

echo "✅ NCCL_HOME detected at: $NCCL_HOME"

# 6. Compile
rm -rf nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# We pass NCCL_HOME explicitly so the linker finds -lnccl
make MPI=1 CC=CC CXX=CC NCCL_HOME=$NCCL_HOME -j 8

echo "✅ Compilation complete. Binary is at $(pwd)/build/all_reduce_perf"
