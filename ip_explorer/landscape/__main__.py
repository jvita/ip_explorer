#!/usr/bin/env python
# coding: utf-8

# Imports

import random
import numpy as np

import torch
import torchmetrics

import os
import argparse
import numpy as np

import pytorch_lightning as pl

import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper

from ip_explorer.datamodules import get_datamodule_wrapper
from ip_explorer.models import get_model_wrapper
from ip_explorer.landscape.loss import EnergyForceLoss

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(
    description="Generate loss landscapes"
)

# Add CLI arguments
parser.add_argument( '--seed', type=int, help='The random seed to use', dest='seed', default=None, required=False,)
parser.add_argument( '--num-nodes', type=int, help='The number of nodes available for training', dest='num_nodes', required=True,) 
parser.add_argument( '--gpus-per-node', type=int, help='The number of GPUs per node to use', dest='gpus_per_node', default=1, required=False,) 

parser.add_argument( '--prefix', type=str, help='Prefix to add to the beginning of logged files', dest='prefix', default='', required=False)
parser.add_argument( '--save-dir', type=str, help='Directory in which to save the results. Created if does not exist', dest='save_dir', required=True)
parser.add_argument( '--overwrite', help='Allows save_directory to be overwritten', action='store_true')
parser.add_argument( '--no-ovewrite', action='store_false', dest='overwrite')
parser.set_defaults(overwrite=False)

parser.add_argument( '--model-type', type=str, help='Type of model being used.  Must be one of the supported types from ip_explorer', dest='model_type', required=True)
parser.add_argument( '--database-path', type=str, help='Path to formatted schnetpack.data.ASEAtomsData database', dest='database_path', required=True)
parser.add_argument( '--model-path', type=str, help='Full path to model checkpoint file', dest='model_path', required=True,) 

parser.add_argument( '--compute-initial-losses', help='Computes and logs the train/test/val losses of the inital model', action='store_true')
parser.add_argument( '--no-compute-initial-losses', action='store_false', dest='compute_initial_losses')
parser.set_defaults(compute_initial_losses=True)

parser.add_argument( '--batch-size', type=int, help='Batch size for data loaders', dest='batch_size', default=128, required=False,)

parser.add_argument( '--loss-type', type=str, help='"energy", "force" or None', dest='loss_type', default=None, required=False,) 
parser.add_argument( '--distance', type=float, help='Fractional distance in parameterspace', dest='distance', required=True,) 
parser.add_argument( '--steps', type=int, help='Number of grid steps in each direction in parameter space', dest='steps', required=True,) 

parser.add_argument( '--additional-kwargs', type=str, help='A string of additional key-value argument pairs that will be passed to the model and datamodule wrappers. Format: "key1:value1 key2:value2"', dest='additional_kwargs', required=False, default='') 

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
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    os.chdir(args.save_dir)
    print("Saving results in:", args.save_dir)

    additional_kwargs = {}
    for kv_pair in args.additional_kwargs.split():
        k, v = kv_pair.split(':')
        additional_kwargs[k] = v

    datamodule = get_datamodule_wrapper(args.model_type)(
        args.database_path,
        batch_size=args.batch_size,
        num_workers=int(np.floor(int(os.environ['LSB_MAX_NUM_PROCESSORS'])/int(os.environ['GPUS_PER_NODE']))),
        **additional_kwargs,
    )
    model = get_model_wrapper(args.model_type)(
        model_dir=args.model_path,
        copy_to_cwd=True,
        **additional_kwargs,
    )

    model.eval()

    if args.compute_initial_losses:

        # TODO: use devices=1 for train/test/val verification to avoid
        # duplicating data, as suggested on this page:
        # https://pytorch-lightning.readthedocs.io/en/stable/common/evaluation_intermediate.html

        # Compute initial train/val/test losses
        trainer = pl.Trainer(
            num_nodes=1,
            devices=1,
            accelerator='cuda',
        )

        print('Computing training errors with devices=1 to avoid batch padding errors', flush=True)
        trainer.test(model, dataloaders=datamodule.train_dataloader())
        train_eloss, train_floss = model.results['e_rmse'], model.results['f_rmse']
        print('Computing validation errors with devices=1 to avoid batch padding errors', flush=True)
        trainer.test(model, dataloaders=datamodule.val_dataloader())
        val_eloss, val_floss = model.results['e_rmse'], model.results['f_rmse']
        print('Computing testing errors with devices=1 to avoid batch padding errors', flush=True)
        trainer.test(model, dataloaders=datamodule.test_dataloader())
        test_eloss, test_floss = model.results['e_rmse'], model.results['f_rmse']

        print('E_RMSE (eV/atom), F_RMSE (eV/Ang)')
        print(f'\tTrain:\t{train_eloss}, \t{train_floss}')
        print(f'\tTest:\t{test_eloss}, \t{test_floss}')
        print(f'\tVal:\t{val_eloss}, \t{val_floss}')

        errors = np.array([
            [train_eloss, train_floss],
            [test_eloss, test_floss],
            [val_eloss, val_floss],
        ])

        np.savetxt(
            os.path.join(args.save_dir, args.prefix+'errors'),
            errors,
            header='col=[E_RMSE, F_RMSE] (eV/atom, eV/Ang); row=[train, test, val]'
        )

    # Switch to using a distributed model. Note that this means there will be
    # some noise in the generated landscapes due to batch padding.
    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        accelerator='cuda',
        strategy='ddp',
        enable_progress_bar=False,
    )

    metric = EnergyForceLoss(
        evaluation_fxn = trainer.test,
        data_loader=datamodule.train_dataloader()
    )

    model_final = SimpleModelWrapper(model)  # needed for loss_landscapes

    loss_data_fin = loss_landscapes.random_plane(
        model_final,
        metric,
        distance=args.distance,     # maximum distance in parameter space
        steps=args.steps,           # number of steps
        normalization='filter',
        deepcopy_model=True,
        n_loss_terms=2,
    )

    save_name = 'L={}_d={:.2f}_s={}'.format('energy', args.distance, args.steps)
    full_path = os.path.join(args.save_dir, args.prefix+save_name)
    np.save(full_path, loss_data_fin[0])

    save_name = 'L={}_d={:.2f}_s={}'.format('forces', args.distance, args.steps)
    full_path = os.path.join(args.save_dir, args.prefix+save_name)
    np.save(full_path, loss_data_fin[1])

    print('Done generating loss landscape!')


if __name__ == '__main__':
    os.environ['GPUS_PER_NODE'] = str(args.gpus_per_node)
    main()
