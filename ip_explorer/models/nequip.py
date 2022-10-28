from .base import PLModelWrapper

import os
import torch
import shutil

from nequip.data import AtomicData, AtomicDataDict
from nequip.train import Trainer, Metrics
from nequip.model import model_from_config
from nequip.utils import Config, instantiate


class NequIPModelWrapper(PLModelWrapper):
    """
    A wrapper to a nequip model. Assumes that `traindir` contains a
    configuration file with the name 'config.yaml', and a model checkpoint with
    the name 'best_model.pth'.

    Note that the 'config.yaml' file should include the following keys:

        ```
        - - forces
          - rmse
        - - total_energy
          - rmse
          - PerAtom: true
        ```

    """
    def __init__(self, model_dir, **kwargs):
        if 'representation_type' in kwargs:
            self.representation_type = kwargs['representation_type']
        else:
            self.representation_type = 'node'

        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, traindir):

        config = Config.from_file(
            os.path.join(traindir, 'config.yaml'),
        )

        # self.model, _ = Trainer.load_model_from_training_session(traindir=traindir)
        self.model = model_from_config(config, initialize=False)

        if 'structure_representations' in self.values_to_compute:
            self._register_representations_hook()

        metrics_components = config.get("metrics_components", None)
        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=config,
        )

        self.metrics = metrics


    def compute_loss(self, batch):
        self.metrics.reset()

        batch_dict = AtomicData.to_AtomicDataDict(batch)

        out = self.model.forward(batch_dict)

        with torch.no_grad():
            self.metrics(out, batch)

        results, _ = self.metrics.flatten_metrics(self.metrics.current_result())

        if 'e/N_rmse' not in results:
            raise RuntimeError("e/N_rmse not found in metrics dictionary. Make sure the config file stored in self.model_path has the correct settings")

        if 'f_rmse' not in results:
            raise RuntimeError("f_rmse not found in metrics dictionary. Make sure the config file stored in self.model_path has the correct settings")

        return {
            'energy': results['e/N_rmse']**2,  # mse isn't implemented yet in nequip
            'force':  results['f_rmse']**2,
            'batch_size': int(max(batch_dict[AtomicDataDict.BATCH_KEY])+1),
            'natoms': batch_dict[AtomicDataDict.FORCE_KEY].shape[0],
        }


    def _register_representations_hook(self):
        """Add hook for extracting the output of the final convolution layer"""
        def hook(model, inputs):
            inputs[0]['node_representations'] = inputs[0][AtomicDataDict.NODE_FEATURES_KEY]
            inputs[0]['edge_representations'] = inputs[0][AtomicDataDict.EDGE_EMBEDDING_KEY]

        for name, module in self.model.named_modules():
            # if name.split('.')[-1] == 'conv_to_output_hidden':
            if name.split('.')[-1] == 'output_hidden_to_scalar':
                module.register_forward_pre_hook(hook)


    def compute_structure_representations(self, batch):
        batch_dict = AtomicData.to_AtomicDataDict(batch)

        out = self.model.forward(batch_dict)

        with torch.no_grad():
            # z = out[AtomicDataDict.NODE_FEATURES_KEY]
            z = out['node_representations']
            per_atom_representations = z.new_zeros(z.shape)

            if self.representation_type in ['node', 'both']:
                per_atom_representations += z

            if self.representation_type in ['edge', 'both']:
                raise NotImplementedError("Still trying to figure out how to use the edge representations properly")

                idx_i = out[AtomicDataDict.EDGE_INDEX_KEY][0, :]
                idx_j = out[AtomicDataDict.EDGE_INDEX_KEY][1, :]

                per_atom_representations.index_add_(0, idx_i, out['edge_representations'])
                per_atom_representations.index_add_(0, idx_j, out['edge_representations'])

        splits = torch.unique(out['batch'], return_counts=True)[1]
        splits = splits.detach().cpu().numpy().tolist()

        return {
            'representations': per_atom_representations,
            'representations_splits': splits,
            'representations_energy': batch_dict[AtomicDataDict.TOTAL_ENERGY_KEY],
        }


    def copy(self, traindir):
        shutil.copyfile(
            os.path.join(traindir, 'best_model.pth'),
            os.path.join(os.getcwd(), 'best_model.pth'),
        )

        shutil.copyfile(
            os.path.join(traindir, 'config.yaml'),
            os.path.join(os.getcwd(), 'config.yaml'),
        )
