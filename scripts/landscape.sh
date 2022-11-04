#!/bin/bash
#BSUB -J "mace_LL_4node"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 4
#BSUB -W 12:00

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# Activate conda environment
source /usr/workspace/vita1/programs/anaconda/bin/activate
conda activate opence-1.7.2-cuda-11.4

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=`hostname ${SLURM_NODELIST} | head -n1`


# Runtime settings
NUM_NODES=2
GPUS_PER_NODE=4

# -n: number of "resource sets"
# -a: number of "tasks" per resource set (default is 1 task = 1 process)
# -c: number of CPUs per resource set
# -g: number of GPUs per resource set
jsrun -n $NUM_NODES -a $GPUS_PER_NODE -c $GPUS_PER_NODE -g $GPUS_PER_NODE python3 -m ip_explorer.landscape \
    --port 4739 \
    --seed 1123 \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --batch-size 2 \
    --loss-type 'both' \
    --distance 0.5 \
    --steps 3 \
    --model-type "test" \
    --save-dir 'test' \
    --database-path 'test' \
    --model-path 'test' \
    --additional-kwargs "m:7 b:3" \
    --no-compute-initial-losses
    # --model-type "mace" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/projects/mace/corr3_128/checkpoints/'\
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/mace/debug" \
    # --additional-kwargs "cutoff:5.0" \
    # --model-type "allegro" \
    # --model-path "/g/g20/vita1/ws/projects/allegro/results/AL_Al/initial/" \
    # --database-path "/g/g20/vita1/ws/projects/allegro/results/AL_Al/initial/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/allegro/initial" \
    # --model-type "schnet" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/projects/schnet/logs/runs/4085987-al_al_atomwise_long-cutoff=7.0-n_atom_basis=30-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=2000/' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/debug" \
    # --additional-kwargs "cutoff:7.0" \
    # --model-type "ace" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/projects/ace/senary' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/ace/senary" \
    # --additional-kwargs "cutoff:7.0" \
    # --model-type "painn" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/logs/runs/painn/4121168-painn_lr_sched-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/schnet/painn" \
    # --additional-kwargs "cutoff:7.0" \
    # --prefix '4gpus_'
    # --overwrite \
