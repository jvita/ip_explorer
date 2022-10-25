from .schnet import SchNetDataModule
from .nequip import NequIPDataModule


implemented_datamodules = {
    'schnet': SchNetDataModule,
    'nequip': NequIPDataModule,
}

def get_datamodule_wrapper(datamodule_type):
    global implemented_datamodules

    if datamodule_type not in implemented_datamodules:
        raise RuntimeError("Wrapper for datamodule type '{}' has not been implemented.".format(datamodule_type))

    return implemented_datamodules[datamodule_type]
