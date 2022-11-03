from .base import PLDataModuleWrapper

import os
import numpy as np
from ase.io import read

from mace.tools import torch_geometric, get_atomic_number_table_from_zs
from mace.data import AtomicData, config_from_atoms_list


class MACEDataModule(PLDataModuleWrapper):
    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                Path to a directory containing the following files:

                    * `train.xyz`: an ASE-readable list of atoms
                    * `test.xyz`: an ASE-readable list of atoms
                    * `val.xyz`: an ASE-readable list of atoms
        """

        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for SchNetDataModule. Use --additional-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """

        train   = config_from_atoms_list(read(os.path.join(stage, 'train.xyz'), format='extxyz', index=':'))
        test    = config_from_atoms_list(read(os.path.join(stage, 'test.xyz'), format='extxyz', index=':'))
        val     = config_from_atoms_list(read(os.path.join(stage, 'val.xyz'), format='extxyz', index=':'))

        z_table = get_atomic_number_table_from_zs(
            z
            for configs in (train, test, val)
            for config in configs
            for z in config.atomic_numbers
        )

        self.train_dataset  = [AtomicData.from_config(c, z_table=z_table, cutoff=self.cutoff) for c in train]
        self.test_dataset   = [AtomicData.from_config(c, z_table=z_table, cutoff=self.cutoff) for c in test]
        self.val_dataset    = [AtomicData.from_config(c, z_table=z_table, cutoff=self.cutoff) for c in val]


    def get_dataloader(self, dataset):
        return torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
