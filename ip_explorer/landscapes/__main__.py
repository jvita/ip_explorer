#!/usr/bin/env python
# coding: utf-8

# Imports

import random
import numpy as np

import torch
import torchmetrics

import os
import logging
import argparse
import numpy as np

# from ase import Atoms
# from ase.io import read, write

import pytorch_lightning as pl

import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper

from ip_explorer.loaders.datamodules import get_datamodule_wrapper
from ip_explorer.loaders.models import get_model_wrapper
from ip_explorer.landscapes.loss import SchNetLoss, NequIPLoss

parser = argparse.ArgumentParser(
    description="Generate loss landscapes"
)

# Add CLI arguments
parser.add_argument( '--seed', type=int, help='The random seed to use', dest='seed', default=None, required=False,)
parser.add_argument( '--num-nodes', type=int, help='The number of nodes available for training', dest='num_nodes', required=True,) 
parser.add_argument( '--gpus-per-node', type=int, help='The number of GPUs per node to use', dest='gpus_per_node', default=1, required=False,) 

parser.add_argument( '--save-dir', type=str, help='Directory in which to save the results. Created if does not exist', dest='save_dir', required=True)
parser.add_argument( '--overwrite', help='Allows save_directory to be overwritten', action='store_true')
parser.add_argument( '--no-ovewrite', action='store_false', dest='overwrite')
parser.set_defaults(overwrite=True)
parser.add_argument( '--model-type', type=str, help='Type of model being used.  Must be one of the supported types from ip_explorer', dest='model_type', required=True)
parser.add_argument( '--database-path', type=str, help='Path to formatted schnetpack.data.ASEAtomsData database', dest='database_path', required=True)
parser.add_argument( '--model-path', type=str, help='Full path to model checkpoint file', dest='model_path', required=True,) 

parser.add_argument( '--batch-size', type=int, help='Batch size for data loaders', dest='batch_size', default=128, required=False,)

parser.add_argument( '--loss-type', type=str, help='"energy", "force" or None', dest='loss_type', default=None, required=False,) 
parser.add_argument( '--cutoff', type=float, help='Pair distance cutoff. Only needed for SchNet', dest='cutoff', required=True)
parser.add_argument( '--distance', type=float, help='Fractional distance in parameterspace', dest='distance', required=True,) 
parser.add_argument( '--steps', type=int, help='Number of grid steps in each direction in parameter space', dest='steps', required=True,) 

args = parser.parse_args()

print('ALL ARGUMENTS:')
print(args)

# Seed RNGs
if args.seed is None:
    args.seed = np.random.randint()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    # Setup
    # if args.model_type == 'schnet':
    #     datamodule = load_datamodule(
    #         args.model_type,
    #         args.database_path,
    #         args.batch_size,
    #         cutoff=args.cutoff,
    #     )

    #     # model = load_model(args.model_type, args.model_path, copy_to_cwd=True)
    # elif args.model_type == 'nequip':
    #     datamodule = load_datamodule(
    #         args.model_type,
    #         args.database_path,
    #         args.batch_size,
    #     )

    #     # model = load_model(args.model_type, args.model_path, copy_to_cwd=True)

    datamodule  = get_datamodule_wrapper(args.model_type)(args.database_path, args.batch_size)
    model       = get_model_wrapper(args.model_type)(args.model_path, copy_to_cwd=True)

    model.eval()
    model_final = SimpleModelWrapper(model)
    start_point = model_final.get_module_parameters()

    logger = pl.loggers.CSVLogger(save_dir=args.save_dir)

    # TODO: use devices=1 for train/test/val verification to avoid duplicating
    # data, as suggested on this page:
    # https://pytorch-lightning.readthedocs.io/en/stable/common/evaluation_intermediate.html

    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=4,
        accelerator='cuda',
        strategy='ddp',
    )

    print('Computing training errors with devices=1', flush=True)
    trainer.test(model, dataloaders=datamodule.train_dataloader())
    print('Computing validation errors with devices=1', flush=True)
    trainer.test(model, dataloaders=datamodule.val_dataloader())
    print('Computing testing errors with devices=1', flush=True)
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    # # Generate random directions
    # dir_one = rand_u_like(start_point)
    # dir_two = orthogonal_to(start_point)

    # # Filter normalize
    # dir_one.filter_normalize_(start_point)
    # dir_two.filter_normalize_(start_point)

    # # scale to match steps and total distance; used for shifting models on each # rank
    # scaling_one = ((start_point.model_norm() * args.distance) / args.steps) / dir_one.model_norm()
    # scaling_two = ((start_point.model_norm() * args.distance) / args.steps) / dir_two.model_norm()
    # dir_one.mul_(scaling_one)
    # dir_two.mul_(scaling_two)

    # # Shift starting points to correct positions depending upon rank
    # steps_per_worker_along_dim1 = int(np.floor(args.steps/(world_size*args.gpus_per_node)))

    # if world_rank == world_size-1:
    #     # Assign any leftovers to rank 0
    #     steps_per_worker_along_dim1 = args.steps - steps_per_worker_along_dim1*world_size*args.gpus_per_node

    # steps_along_dir1 = world_comm.allgather(steps_per_worker_along_dim1)

    # print(f'RANK {world_rank} steps: {steps_along_dir1}')

    # # Move back along dir_one depending upon rank
    # dir_one.mul_(steps_per_worker_along_dim1)
    # start_point.sub_(dir_one)
    # dir_one.truediv_(steps_per_worker_along_dim1)

    # # Move to beginning of dir_two
    # dir_one.mul_(args.steps)
    # start_point.sub_(dir_one)
    # dir_one.truediv_(steps_per_worker_along_dim1)

    # # Un-do temporary scaling now that the model has been shifted
    # dir_one.truediv_(scaling_one)
    # dir_two.truediv_(scaling_two)

    raise RuntimeError

    loss_data_fin = loss_landscapes.random_plane(
        model_final,
        metrics['train'],
        distance=args.distance,  # maximum distance in parameter space
        steps=args.steps,    # number of steps
        normalization=None,  # they've already been filter-normalized
        deepcopy_model=True,
        n_loss_terms=2,
        dir_one=dir_one,
        dir_two=dir_two,
        shift_to_center=False,
    )

    parent_folder = os.path.split(args.model_path)[0]

    print("Saving results in:", parent_folder)

    errors = np.array([
        [train_eloss, train_floss],
        [test_eloss, test_floss],
        [val_eloss, val_floss],
    ])

    full_path = os.path.join(parent_folder, '_'.join([args.prefix, f'errors']))

    np.savetxt(
        full_path,
        errors,
        header='col=[E_RMSE, F_RMSE] (eV/atom, eV/Ang); row=[train, test, val]'
    )

    save_name = 'L={}_d={:.2f}_s={}'.format('energy', args.distance, args.steps)
    full_path = os.path.join(parent_folder, save_name)
    np.save(full_path, loss_data_fin[0])

    save_name = 'L={}_d={:.2f}_s={}'.format('forces', args.distance, args.steps)
    full_path = os.path.join(parent_folder, save_name)
    np.save(full_path, loss_data_fin[1])


if __name__ == '__main__':
    os.environ['GPUS_PER_NODE'] = str(args.gpus_per_node)
    main()
