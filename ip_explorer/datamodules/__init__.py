from .schnet import SchNetDataModule
from .nequip import NequIPDataModule
from .ace import ACEDataModule
from .mace import MACEDataModule
from .ase import ASEDataModule


implemented_datamodules = {
    'schnet':       SchNetDataModule,
    'painn':        SchNetDataModule,
    'nequip':       NequIPDataModule,
    'allegro':      NequIPDataModule,
    'ace':          ACEDataModule,
    'mace':         MACEDataModule,
    'valle-oganov': ASEDataModule,
}

def get_datamodule_wrapper(datamodule_type):
    global implemented_datamodules

    if datamodule_type not in implemented_datamodules:
        raise RuntimeError("Wrapper for datamodule type '{}' has not been implemented.".format(datamodule_type))

    return implemented_datamodules[datamodule_type]
