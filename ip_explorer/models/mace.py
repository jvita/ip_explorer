from .base import PLModelWrapper

import os
import glob
import shutil
import torch
import numpy as np

class MACEModelWrapper(PLModelWrapper):
    """
    A wrapper for a MACE model.

    """
    def __init__(self, model_dir, **kwargs):
        """
        Arguments:

            model_dir (str):
                The path to a folder containing the following files:

                    * 'results/template.model': a model template that can be loaded
                    using `torch.load`; intended as a workaround for having to
                    call the model constructor with all hyperparameters. Note
                    that the mace/scripts/run_train.py script may have needed to
                    have been modified to save this template at the beginning of
                    training.
                    * 'results/*epoch-*.pt' checkpoints with the "model" key; used for loading
                    the state dict of a trained model
        """
        if 'representation_type' in kwargs:
            self.representation_type = kwargs['representation_type']
        else:
            self.representation_type = 'node'


        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = torch.load(os.path.join(model_path, 'checkpoints', 'template.model'))

        # For handling the possible existence of multiple checkpoints
        checkpoint_files = glob.glob(os.path.join(model_path, 'checkpoints', '*.pt'))
        checkpoint_epochs = [
            int(path.split('epoch-')[-1].split('.pt')[0])
            for path in checkpoint_files
        ]

        state_dict = torch.load(checkpoint_files[np.argmax(checkpoint_epochs)])['model']

        self.model.load_state_dict(state_dict)

        if 'structure_representations' in self.values_to_compute:
            self._node_representations = []
            self._register_representations_hook()

    def random_model(self, model_path):
        return torch.load(os.path.join(model_path, 'checkpoints', 'template.model'))


    def compute_loss(self, batch):

        natoms = torch.unique(batch.batch, return_counts=True)[1]

        true_eng = batch.energy/natoms
        true_fcs = batch.forces

        results = self.model(batch)

        pred_eng = results['energy']/natoms
        pred_fcs = results['forces']

        ediff = (pred_eng - true_eng).detach().cpu().numpy()
        fdiff = (pred_fcs - true_fcs).detach().cpu().numpy()

        return {
            'energy': np.mean(ediff**2),
            'force':  np.mean(fdiff**2),
            'batch_size': batch.energy.shape[0],
            'natoms': sum(natoms).detach().cpu().numpy(),
        }


    def _register_representations_hook(self):
        """Add hook for extracting the output of the final convolution layer"""
        def hook(model, inputs):
            self._node_representations.append(inputs[0])

        # There are readout layers for each interaction block, which are then
        # summed. This is Eq. 4 in the MACE paper.
        for module in self.model.readouts:
            module.register_forward_pre_hook(hook)


    def compute_structure_representations(self, batch):
        out = self.model.forward(batch)

        with torch.no_grad():
            representations = []

            if self.representation_type in ['node', 'both']:
                representations.append(torch.cat(self._node_representations, dim=1))

            if self.representation_type in ['edge', 'both']:
                raise NotImplementedError("Edge interactions are not supported for MACE yet")

                if isinstance(self.model.representation, SchNet):
                    x = batch['scalar_representation']
                    z = x.new_zeros((batch['scalar_representation'].shape[0], self.model.radial_basis.n_rbf))

                    idx_i = batch[structure.idx_i]
                    idx_j = batch[structure.idx_j]

                    z.index_add_(0, idx_i, batch['distance_representation'])
                    z.index_add_(0, idx_j, batch['distance_representation'])
                else:  # PaiNN
                    z = batch['vector_representation']
                    z = torch.mean(z, dim=1)  # average over cartesian dimension

                representations.append(z)

        self._node_representations = []  # reset logged representations

        representations = torch.cat(representations, dim=1)
        splits = torch.unique(batch.batch, return_counts=True)[1]

        return {
            'representations': representations,
            'representations_splits': splits.detach().cpu().numpy().tolist(),
            'representations_energy': batch.energy/splits,
        }



    def copy(self, model_path):

        shutil.copyfile(
            os.path.join(model_path,  'checkpoints/template.model'),
            os.path.join(os.getcwd(), 'template.model')
        )

        checkpoint_files = glob.glob(os.path.join(model_path, 'checkpoints/*.pt'))
        checkpoint_epochs = [
            int(path.split('epoch-')[-1].split('.pt')[0])
            for path in checkpoint_files
        ]

        best_file = checkpoint_files[np.argmax(checkpoint_epochs)]
        best_file = os.path.split(best_file)[-1]

        shutil.copyfile(
            os.path.join(model_path,  'checkpoints/'+best_file),
            os.path.join(os.getcwd(), best_file)
        )
