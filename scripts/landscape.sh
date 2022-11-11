#!/bin/bash
#BSUB -J "schnet_al_redo_LL_8node_debug"
#BSUB -o "/usr/workspace/vita1/logs/lsf/%J.out"
#BSUB -e "/usr/workspace/vita1/logs/lsf/%J.err"
#BSUB -G c02red
#BSUB -q pbatch
#BSUB -nnodes 8
#BSUB -W 2:00

# Environment setup
module load gcc/8.3.1
module load cuda/11.6.1

# Activate conda environment
source /usr/workspace/vita1/programs/anaconda/bin/activate
conda activate opence-1.7.2-cuda-11.4

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# export MASTER_ADDR=`hostname ${SLURM_NODELIST} | head -n1`

# just to record each node we're using in the job output
jsrun -r 1 hostname
 
# get hostname of node that jsrun considers to be first (where rank 0 will run)
firsthost=`jsrun --nrs 1 -r 1 hostname`
echo "first host: $firsthost"

# set MASTER_ADDR to hostname of first compute node in allocation
# set MASTER_PORT to any used port number
export MASTER_ADDR=$firsthost
# export MASTER_PORT=23456

# Runtime settings
NUM_NODES=8
GPUS_PER_NODE=4
CPUS_PER_GPU=1
CPUS_PER_NODE=$(( $GPUS_PER_NODE*$CPUS_PER_GPU ))

# -r: number of resource sets per node
# -a: number of "tasks" per resource set (default is 1 task = 1 process)
# -c: number of CPUs per resource set
# -g: number of GPUs per resource set
# --bind=none: allow each task to use all of its allocated cpus
jsrun -r 1 -a $GPUS_PER_NODE -c $CPUS_PER_NODE -g $GPUS_PER_NODE --bind=none python3 -m ip_explorer.landscape \
    --port 4757 \
    --num-nodes $NUM_NODES \
    --gpus-per-node $GPUS_PER_NODE \
    --batch-size 16 \
    --loss-type 'both' \
    --distance 0.5 \
    --steps 41 \
    --model-type 'schnet' \
    --model-path '/g/g20/vita1/ws/projects/schnet/logs/runs/4085987-al_al_atomwise_long-cutoff=7.0-n_atom_basis=30-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=2000/' \
    --additional-kwargs "cutoff:7.0" \
    --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/schnet/atomwise" \
    --no-compute-initial-losses \
    # --seed 1123 \
    # --model-type "painn" \
    # --model-path '/g/g20/vita1/ws/logs/runs/schnet/painn/4166174-schnet_aspirin-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/aspirin" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/aspirin/schnet/atomwise" \
    # --additional-kwargs "cutoff:7.0" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/aspirin" \
    # --model-type "allegro" \
    # --model-path "/g/g20/vita1/ws/projects/allegro/results/AL_Al/initial/" \
    # --database-path "/g/g20/vita1/ws/projects/allegro/results/AL_Al/initial/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/allegro/initial" \
    # --model-type "mace" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al/" \
    # --model-path '/g/g20/vita1/ws/projects/mace/corr3_128/checkpoints/'\
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/mace/corr3_equi" \
    # --additional-kwargs "cutoff:5.0" \
    # --model-type "test" \
    # --database-path "test" \
    # --model-path 'test' \
    # --save-dir "test" \
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
