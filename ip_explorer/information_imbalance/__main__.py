#!/usr/bin/env python
# coding: utf-8

# Imports

import random
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import torch
import torchmetrics

import os
import argparse
import numpy as np

from ase import Atoms
from ase.io import write

import pytorch_lightning as pl

from ip_explorer.datamodules import get_datamodule_wrapper
from ip_explorer.models import get_model_wrapper

from dadapy.metric_comparisons import MetricComparisons

import logging
# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(
    description="Generate PES"
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
parser.add_argument( '--model-path', type=str, help='Full path to model checkpoint file', dest='model_path', default='.') 

parser.add_argument( '--batch-size', type=int, help='Batch size for data loaders', dest='batch_size', default=128, required=False,)
parser.add_argument( '--slice', type=int, help='Step size to use when reducing data size via slicing', default=1, dest='slice', required=False,)

parser.add_argument( '--additional-kwargs', type=str, help='A string of additional key-value argument pairs that will be passed to the model and datamodule wrappers. Format: "key1:value1 key2:value2"', dest='additional_kwargs', required=False, default='') 
parser.add_argument( '--vgop-kwargs', type=str, help='A string of key-value pairs that will be passed to the VGOP model and datamodule wrappers. Format: "key1:value1 key2:value2"', dest='vgop_kwargs', required=False, default='') 

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

    additional_kwargs = {}
    for kv_pair in args.additional_kwargs.split():
        k, v = kv_pair.split(':')
        additional_kwargs[k] = v

    vgop_kwargs = {}
    for kv_pair in args.vgop_kwargs.split():
        k, v = kv_pair.split(':')
        vgop_kwargs[k] = v

    # Load VGOP model
    vgop = get_model_wrapper('vgop')(
        model_dir=None,
        values_to_compute=('structure_representations',),
        copy_to_cwd=False,
        **vgop_kwargs,
    )

    vgop.eval()

    vgop_datamodule = get_datamodule_wrapper('vgop')(
        args.database_path,
        batch_size=args.batch_size,
        num_workers=int(np.floor(int(os.environ['LSB_MAX_NUM_PROCESSORS'])/int(os.environ['GPUS_PER_NODE']))),
        **vgop_kwargs,
    )


    # Load learned model
    model = get_model_wrapper(args.model_type)(
        model_dir=args.model_path,
        values_to_compute=('structure_representations',),
        copy_to_cwd=True,
        **additional_kwargs,
    )

    model.eval()

    model_datamodule = get_datamodule_wrapper(args.model_type)(
        args.database_path,
        batch_size=args.batch_size,
        num_workers=int(np.floor(int(os.environ['LSB_MAX_NUM_PROCESSORS'])/int(os.environ['GPUS_PER_NODE']))),
        **additional_kwargs,
    )

    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        accelerator='cuda',
        strategy='ddp',
        # enable_progress_bar=False,
    )

    trainer.test(vgop, dataloaders=vgop_datamodule.train_dataloader())
    original = vgop.results['representations'].detach().cpu().numpy()
    original_eng = vgop.results['representations_energies'].detach().cpu().numpy()
    print('VGOP REPRESENTATIONS SHAPE:', original.shape)

    trainer.test(model, dataloaders=model_datamodule.train_dataloader())
    learned = model.results['representations'].detach().cpu().numpy()
    learned_eng = model.results['representations_energies'].detach().cpu().numpy()
    print('MODEL REPRESENTATIONS SHAPE:', learned.shape)

    nsamples = original.shape[0]

    # Sort by energy to try to deal with the fact that orders will be wrong
    original = original[np.argsort(original_eng)]
    learned = learned[np.argsort(learned_eng)]

    original_metric = MetricComparisons(original)
    learned_metric  = MetricComparisons(learned)

    original_metric.compute_distances(maxk=nsamples-1, metric='euclidean')
    learned_metric.compute_distances(maxk=nsamples-1, metric='euclidean')

    original_ranks = original_metric.dist_indices
    learned_ranks  = learned_metric.dist_indices

    learned_imbalances = learned_metric.return_inf_imb_target_selected_coords(
        target_ranks=original_ranks,
        coord_list=[list(range(learned.shape[1]))]
    )

    np.save(
        os.path.join(args.save_dir, args.prefix+'information-imbalance'),
        learned_imbalances,
    )

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(learned_imbalances[0, 0], learned_imbalances[1, 0], label=args.model_type)

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_aspect('equal')

    ax.legend()

    ax.set_title("Information imbalance", fontsize=18, pad=20)

    ax.set_ylabel(r"$\Delta($VGOP $\rightarrow$ model$) $", fontsize=14)
    ax.set_xlabel(r"$\Delta($VGOP $\rightarrow$ model$) $", fontsize=14)

    plt.savefig(
        os.path.join(args.save_dir, args.prefix+'information-imbalance.png'),
        bbox_inches='tight'
    )

    print("Saving results in:", args.save_dir)


if __name__ == '__main__':
    os.environ['GPUS_PER_NODE'] = str(args.gpus_per_node)
    main()