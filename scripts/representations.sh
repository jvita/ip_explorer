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

jsrun -n 1 python3 -m ip_explorer.representations \
    --seed 1123 \
    --num-nodes 1 \
    --gpus-per-node 1 \
    --batch-size 4 \
    --overwrite \
    --prefix "node_" \
    --model-type "nequip" \
    --model-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/no_rescale/" \
    --database-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/no_rescale/" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/nequip/no_rescale" \
    --additional-kwargs "representation_type:node" \
    # -model-type "painn" \
    # -database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # -model-path '/usr/workspace/vita1/logs/runs/painn/4121168-painn_lr_sched-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # -save-dir "/g/g20/vita1/ws/logs/ip_explorer/schnet/painn" \
    # -additional-kwargs "cutoff:7.0 representation_type:node" \
    # --prefix '4gpus_'
    # --no-compute-initial-losses
