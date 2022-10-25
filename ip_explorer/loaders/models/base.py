import torch
import numpy as np
import pytorch_lightning as pl


class PLModelWrapper(pl.LightningModule):
    """
    A model wrapper for facilitating distributed execution via Pytorch
    Lightning. When implementing a new model that sub-classes from PLModelWrapper,
    the three functions that should be implemented are `load_model`, `loss_fxn`,
    and `copy`.

    This wrapper utilizes the `pl.Trainer.test()` function as a workaround for
    distributed prediction. After calling `pl.Trainer.test()`, the RMSE values
    will be stored under the `PLModelWrapper.rmse_eng` and `PLModelWrapper.rmse_fcs`
    variables (a workaround for the fact that the `test_epoch_end()` function
    can't return anything yet).

    NOTE: When doing distributed evaluation, Pytorch Lightning may pad the
    batches to ensure that the batch size is consistent across devices. This can
    lead to slightly incorrect statistics. Therefore, `devices=1` should be
    passed to the pl.Trainer class when exact statistics are required using this
    model.
    """

    def __init__(self, model_dir, copy_to_cwd=False):
        """
        Arguments:

            model_dir (str):
                The path to a folder containing the information necessary to
                load the model. Will be passed to `PLModelWrapper.load_model` and
                `PLModelWrapper.copy()`

            copy_to_cwd (bool):
                If True, calls the `PLModelWrapper.copy()` function during
                instantiation.

        """
        super().__init__()

        self.model = None
        self.load_model(model_dir)

        if self.model is None:
            raise RuntimeError("Failed to load model. Make sure to implement `load_model()` and assign `self.model`")

        if copy_to_cwd:
            self.copy(model_dir)

        self.model_dir = model_dir


    def load_model(self, model_dir):
        """
        Populates the `self.model` variable with a callable PyTorch module. This
        is also where any additional model-specific class variables should be
        populated.
        """
        raise NotImplementedError


    def loss_fxn(self, batch):
        """
        A function for computing the energy and force MSE of the model.

        Arguments:

            batch:
                An object that can be passed directly to the `model.forward()`
                function.

        Returns:

            results (dict):
                A dictionary with four required keys:

                    * `'energy'`: the batch energy MSE (units of energy per atom)
                    * `'force'`: the batch force MSE (units of energy per distance)
                    * `'batch_size'`: the number of structures in the batch.
                        Used for computing weighted averages of energy errors.
                    * `'natoms'`: the number of atoms in the batch.
                        Used for computing weighted averages of force errors.
        """
        raise NotImplementedError


    def copy(self, model_path):
        """
        Copies all files necessary for model construction to the current
        working directory.

        Args:

            model_path (str):
                The path to a folder containing the information necessary to
                load the model.

        """
        raise NotImplementedError


    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        results = self.loss_fxn(batch)

        # loss_fxn MUST return the MSE or MAE so weighted aggregation is correct
        results['energy'] *= results['batch_size']
        results['force']  *= results['natoms']

        for k,v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu()#.numpy()

        return results

    def test_epoch_end(self, step_outputs):
        n_e_tot = sum([s['batch_size'] for s in step_outputs])
        n_f_tot = sum([s['natoms'] for s in step_outputs])

        rmse_eng = np.sqrt(np.sum(([s['energy'] for s in step_outputs]))/n_e_tot)
        rmse_fcs = np.sqrt(np.sum(([s['force'] for s in step_outputs]))/n_f_tot)

        rmse_eng = np.average(self.all_gather(rmse_eng).detach().cpu().numpy())
        rmse_fcs = np.average(self.all_gather(rmse_fcs).detach().cpu().numpy())

        print('RMSE:', rmse_eng, rmse_fcs)

        return rmse_eng, rmse_fcs


