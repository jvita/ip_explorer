#!/bin/bash
#BSUB -J "schnet_ethanol_II"
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
# source /g/g20/vita1/venv-development/bin/activate
source /usr/workspace/vita1/programs/anaconda/bin/activate
conda activate opence-1.7.2-cuda-11.4

# Used for PyTorch Lightning
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)

MODEL_NAME='corr2_128_inv'

jsrun -n 1 python3 -m ip_explorer.information_imbalance \
    --seed 1123 \
    --num-nodes 1 \
    --gpus-per-node 1 \
    --batch-size 32 \
    --overwrite \
    --model-type "schnet" \
    --additional-kwargs "cutoff:7.0 representation_type:node remove_offsets:False" \
    --vgop-kwargs "min_cut:1.0 max_cut:4.0 num_cutoffs:10 elements:['C','H','O'] interactions:'all' pad_atoms:True" \
    --model-path '/g/g20/vita1/ws/logs/runs/schnet/atomwise/4165865-schnet_ethanol-cutoff=7.0-n_atom_basis=128-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=5000' \
    --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/ethanol" \
    --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/ethanol/schnet/atomwise" \
    # --model-path '/g/g20/vita1/ws/projects/schnet/logs/runs/4085987-al_al_atomwise_long-cutoff=7.0-n_atom_basis=30-n_interactions=3-n_rbf=20-n_layers=2-n_hidden=None-Ew=0.01-Fw=0.99-lr=0.005-epochs=2000' \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --prefix 'actually_al_data_' \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/rmd17/ethanol/schnet/atomwise" \
    # --model-type "vgop" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/vgop" \
    # --additional-kwargs "cutoffs:[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] elements:['Al'] interactions:'all'" \
    #--additional-kwargs "cutoffs:[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] elements:['C','H','O','N'] interactions:'all' pad_atoms:True" \
    # --database-path "/g/g20/vita1/ws/projects/data/rMD17/rmd17/downsampled/salicylic" \
    # --model-type "mace" \
    # --model-path "/g/g20/vita1/ws/projects/mace/results/${MODEL_NAME}/checkpoints/" \
    # --database-path "/g/g20/vita1/ws/projects/data/AL_Al" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/AL_Al/mace/${MODEL_NAME}" \
    # --additional-kwargs "cutoff:5.0" \
    # --model-type "nequip" \
    # --model-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/no_rescale/" \
    # --database-path "/g/g20/vita1/ws/projects/nequip/results/AL_Al/no_rescale/" \
    # --save-dir "/g/g20/vita1/ws/logs/ip_explorer/nequip/no_rescale" \
    # --additional-kwargs "representation_type:both" \
    # --no-compute-initial-losses
