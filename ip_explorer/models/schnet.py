from .base import PLModelWrapper

import os
import shutil
import torch
import numpy as np


class SchNetModelWrapper(PLModelWrapper):
    """
    A wrapper for a SchNet model. Assumes that `model_path` contains a model
    checkpoint file with the name 'best_model'
    """
    def __init__(self, model_dir, **kwargs):
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
            'natoms': sum(batch['_n_atoms']),
        }


    def copy(self, model_path):
        shutil.copyfile(
            os.path.join(model_path, 'best_model'),
            os.path.join(os.getcwd(), 'best_model'),
        )


