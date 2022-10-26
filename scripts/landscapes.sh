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
source /g/g20/vita1/venv-development/bin/activate

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)

MODEL_FOLDER=MODEL_DIRECTORY

jsrun -n 1 python3 -m ip_explorer.landscapes \
    --seed 1123 \
    --num-nodes 1 \
    --gpus-per-node 4 \
    --batch-size 32 \
    --loss-type 'both' \
    --distance 0.5 \
    --steps 5 \
    --model-type "nequip" \
    --model-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/debug/" \
    --database-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/debug/" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/nequip/debug" \
    # --model-type "schnet" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/logs/runs/painn/4114101-painn_initial-cutoff=7.0-n_atom_basis=30-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000/' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/schnet/painn" \
    # --additional-kwargs "cutoff:7.0" \
    # --prefix '4gpus_'
    # --no-compute-initial-losses
    # --overwrite \
