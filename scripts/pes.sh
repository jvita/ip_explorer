#!/bin/bash
#BSUB -J "pain_LL_4gpus_41"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# conda init bash
# conda activate conda-development
source /g/g20/vita1/venv-ruby/bin/activate

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)

LOG_DIR='/g/g20/vita1/ws/logs/ip_explorer/schnet/painn'
PREFIX='node_'
DIMENSIONS=3

sheap -v -hs -rs -1 -p 20 -st 0.4 -dim $DIMENSIONS < "${LOG_DIR}/${PREFIX}representations.xyz" > "${LOG_DIR}/${PREFIX}sheap-${DIMENSIONS}d.xyz"

python3 -m ip_explorer.pes \
    --load-dir '/g/g20/vita1/ws/logs/ip_explorer/schnet/painn/' \
    --prefix ${PREFIX} \
    --n-components ${DIMENSIONS} \
    --scale 0.1 \
