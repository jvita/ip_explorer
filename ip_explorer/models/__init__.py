from .schnet import SchNetModelWrapper
from .nequip import NequIPModelWrapper
from .ace import ACEModelWrapper
from .valle_oganov import ValleOganovModelWrapper

implemented_wrappers = {
    'schnet':       SchNetModelWrapper,
    'painn':        SchNetModelWrapper,
    'nequip':       NequIPModelWrapper,
    'allegro':      NequIPModelWrapper,
    'ace':          ACEModelWrapper,
    'valle-oganov': ValleOganovModelWrapper,
}

def get_model_wrapper(model_type):
    global implemented_wrappers

    if model_type not in implemented_wrappers:
        raise RuntimeError("Wrapper for model type '{}' has not been implemented.".format(model_type))

    return implemented_wrappers[model_type]
