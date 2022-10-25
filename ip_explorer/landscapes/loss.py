import numpy as np

import loss_landscapes
from loss_landscapes.model_interface.model_wrapper import ModelWrapper

from nequip.data import AtomicData


class EnergyForceLoss(loss_landscapes.metrics.Metric):
    """
    A wrapper for computing the energy/loss RMSE values for a given model and
    dataset. This class is specifically inteneded to interface with the
    loss_landscapes package.
    """

    data_loader    = None
    evaluation_fxn = None
    loss_type      = None

    def __init__(
        self,
        data_loader,
        evaluation_fxn=None,
        loss_type='both',
        device='cpu'
        ):
        """
        Arguments:

            data_loader (iterable):
                An iterable that returns data batches that can be passed
                directly to `evaluation_fxn`. This will typically be something
                like a PyTorch DataLoader instance.

            evaluation_fxn (callable):
                A function that takes as arguments `(model, data_loader)`,
                and returns a dictionary of values corresponding to chosen terms
                in the loss function. The only required dictionary keys are
                'energy' and 'force'.

            loss_type (str, default='both'):
                One of ['energy', 'force', 'both'].

            device (torch.device):
                The device to which to send the data
        """
        super().__init__()
        self.data_loader    = data_loader
        self.evaluation_fxn = evaluation_fxn
        self.loss_type      = loss_type
        self.device         = device


    def __call__(self, model_wrapper: ModelWrapper) -> tuple:
        """
        Returns the computed loss terms. Returned units depend upon the
        specified `evaluation_fxn`, but are expected to be in eV/atom and
        eV/Ang.

        Returns:
            (energy_rmse,), (force_rmse,) or (energy_rmse, force_rmse)
        """
        results_dict = self.evaluation_fxn(
            model_wrapper.modules[0], self.data_loader, self.device
        )

        if self.loss_type == 'energy':
            return results_dict['energy'],
        elif self.loss_type == 'force':
            return results_dict['force'],
        else:
            return results_dict['energy'], results_dict['force']


class SchNetLoss(EnergyForceLoss):

    def __init__(self, data_loader, loss_type, device):
        super().__init__(
            data_loader=data_loader,
            evaluation_fxn=self._evaluation_fxn,
            loss_type=loss_type,
            device=device
        )

    @staticmethod
    def _batch_eval(model, batch):
        return model.forward(batch)

    @staticmethod
    def _evaluation_fxn(model, data_loader, device) -> dict:
        true_energies = []
        pred_energies = []

        true_forces = []
        pred_forces = []

        for batch in data_loader:
            for k,v in batch.items():
                batch[k] = v.to(device)

            true_energies.append((batch['energy']/batch['_n_atoms']).detach().cpu().numpy())
            true_forces.append(batch['forces'].detach().cpu().numpy())

            results = self._batch_eval(model, batch)

            pred_energies.append((results['energy']/batch['_n_atoms']).detach().cpu())
            pred_forces.append(results['forces'].detach().cpu())

            for k,v in batch.items():
                batch[k] = v.to('cpu')

        pred_energies = np.concatenate(pred_energies)
        true_energies = np.concatenate(true_energies)

        pred_forces = np.concatenate(pred_forces)
        true_forces = np.concatenate(true_forces)

        loss_eng = np.sqrt(np.mean((pred_energies - true_energies)**2))
        loss_fcs = np.sqrt(np.mean((pred_forces   - true_forces)**2))

        return {
            'energy': loss_eng,
            'force':  loss_fcs,
        }


class NequIPLoss(EnergyForceLoss):
    def __init__(self, data_loader, metrics, loss_type, device):
        """
        Arguments:

            data_loader (iterable):
                An iterable that returns data batches that can be passed
                directly to `evaluation_fxn`. This will typically be something
                like a PyTorch DataLoader instance.

            metrics (nequip.train.Metrics):
                The NequIP Metrics logger for tracking running averages

            loss_type (str, default='both'):
                One of ['energy', 'force', 'both'].
        """
        super().__init__(
            data_loader=data_loader,
            evaluation_fxn=self._evaluation_fxn(metrics),
            loss_type=loss_type,
            device=device
        )


    @staticmethod
    def _evaluation_fxn(metrics, device) -> float:
        def f(model, data_loader):
            metrics.reset()

            for batch in data_loader:
                batch.to(device)
                out = model.forward(AtomicData.to_AtomicDataDict(batch))

                with torch.no_grad():
                    metrics(out, batch)

                batch.to('cpu')

            results, _ = metrics.flatten_metrics(self.metrics.current_result())

            return {
                'energy': results['e/N_rmse'],
                'force':  results['f_rmse'],
            }


class SNNPLoss(EnergyForceLoss):
    def __init__(self, data_loader, metrics, loss_type):
        super().__init__(
            data_loader=data_loader,
            evaluation_fxn=self._evaluation_fxn(metrics),
            loss_type=loss_type
        )


    @staticmethod
    def _evaluation_fxn(model, data_loader, device) -> float:
        true_energies = []
        pred_energies = []

        true_forces = []
        pred_forces = []

        for batch in data_loader:
            batch['species'] = batch['species'].to(device)
            batch['cell'] = batch['cell'].to(device)
            batch['coordinates'] = batch['coordinates'].to(device)
            batch['pbc'] = batch['pbc'].to(device)

            forces = torch.zeros_like(batch['true_forces'])

            energies, forces, embeddings = model.forward(batch, calc_forces=True)
            energies = energies.sum()/batch['true_forces'].shape[0]

            true_energies.append(batch['true_energies'].cpu().numpy())
            pred_energies.append(energies.detach().cpu().numpy())

            true_forces.append(batch['true_forces'].cpu().numpy()[0])
            pred_forces.append(forces.detach().cpu().numpy()[0])

            batch['coordinates'] = batch['coordinates'].cpu()
            batch['species'] = batch['species'].cpu()
            batch['cell'] = batch['cell'].cpu()
            batch['pbc'] = batch['pbc'].cpu()

        loss_eng = np.sqrt(np.mean((np.array(pred_energies) - np.concatenate(true_energies))**2))
        loss_fcs = np.sqrt(np.mean((np.concatenate(pred_forces)   - np.concatenate(true_forces))**2))

        return {
            'energy': loss_eng,
            'force':  loss_fcs,
        }
