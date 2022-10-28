from .base import PLModelWrapper

import os
import shutil
import torch
import numpy as np

import tensorflow as tf
from tensorpotential.potential.ace import ACE


class ACETorchWrapper(ACE):
    """
    This class wraps a TensorPotential implementation of ACE using
    setters/getters on specific class attributes so that underlying Tensorflow
    Variables are exposed as PyTorch Parameters.
    """
    def __init__(self, tflow_model):
        self._tflow_model = tflow_model

        # self._fit_coefs = torch.nn.Parameter(self._tflow_model.fit_coefs.numpy(), requires_grad=True)

        self._basis_coefs = torch.nn.Parameter(self._tflow_model.fit_coefs[self.total_num_crad:]

    def set_coefs(self, coefs):
        self.fit_coefs.assign(tf.Variable(coefs, dtype=tf.float64))

    def get_coefs(self):
        return self.fit_coefs

    # @parameter
    # def fit_coefs(self):
    #     return self._fit_coefs

    # @fit_coefs.setter
    # def fit_coefs(self, fit_coefs):
    #     self._fit_coefs = fit_coefs
    #     self._tflow_model.fit_coefs = tf.Variable(fit_coefs.detach().cpu().numpy(), dtype=tf.float64, name='adjustable_coefs')

class ACEModelWrapper(PLModelWrapper):
    """
    A wrapper for an ACE model. This wrapper is
    """
    def __init__(self, model_dir, **kwargs):
        if 'representation_type' in kwargs:
            self.representation_type = kwargs['representation_type']
        else:
            self.representation_type = 'node'


        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = torch.load(
            os.path.join(model_path, 'best_model'),
            map_location=torch.device('cpu')
        )


    def compute_loss(self, batch):
        true_eng = batch['energy']/batch['_n_atoms']
        true_fcs = batch['forces']

        results = self.model.forward(batch)

        pred_eng = results['energy']/batch['_n_atoms']
        pred_fcs = results['forces']

        ediff = (pred_eng - true_eng).detach().cpu().numpy()
        fdiff = (pred_fcs - true_fcs).detach().cpu().numpy()

        return {
            'energy': np.mean(ediff**2),
            'force':  np.mean(fdiff**2),
            'batch_size': batch['energy'].shape[0],
            'natoms': sum(batch['_n_atoms']).detach().cpu().numpy(),
        }


    def compute_structure_representations(self, batch):
        out = self.model.forward(batch)

        with torch.no_grad():
            representations = []

            if self.representation_type in ['node', 'both']:
                representations.append(batch['scalar_representation'])

            if self.representation_type in ['edge', 'both']:

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

        representations = torch.cat(representations, dim=1)

        return {
            'representations': representations,
            'representations_splits': batch['_n_atoms'].detach().cpu().numpy().tolist(),
            # 'representations_energy': batch_dict[AtomicDataDict.TOTAL_ENERGY_KEY],
            'representations_energy': batch['energy'],
        }



    def copy(self, model_path):
        shutil.copyfile(
            os.path.join(model_path, 'best_model'),
            os.path.join(os.getcwd(), 'best_model'),
        )


