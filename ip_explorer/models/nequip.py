from .base import PLModelWrapper

import os
import torch
import shutil

from nequip.data import AtomicData
from nequip.train import Trainer, Metrics
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
    def __init__(self, traindir, copy_to_cwd=False):
        super().__init__(model_dir=traindir, copy_to_cwd=copy_to_cwd)


    def load_model(self, traindir):

        self.model, _ = Trainer.load_model_from_training_session(traindir=traindir)
        metrics_config = Config.from_file(
            os.path.join(traindir, 'config.yaml'),
        )
        metrics_components = metrics_config.get("metrics_components", None)
        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=metrics_config,
        )

        self.metrics = metrics


    def loss_fxn(self, batch):
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
            'batch_size': max(batch_dict['batch'])+1,
            'natoms': batch_dict['forces'].shape[0],
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
