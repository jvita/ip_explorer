#!/bin/bash
#BSUB -J "JOB_NAME"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 6:00

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
    --gpus-per-node 1 \
    --model-type "nequip" \
    --database-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/debug/" \
    --model-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/debug/" \
    --save-dir "/g/g20/vita1/ws/projects/ip_explorer/logs/debug" \
    --batch-size 32 \
    --cutoff 7.0 \
    --loss-type 'both' \
    --distance 0.5 \
    --steps 21 \
    --overwrite
    # --model-path "/g/g20/vita1/ws/projects/schnet/logs/runs/4085986-al_al_bondwise_long-cutoff=7.0-n_atom_basis=30-n_interactions=0-n_rbf=20-n_layers=8-n_hidden=30-Ew=0.01-Fw=0.99-lr=0.005-epochs=2000" \
