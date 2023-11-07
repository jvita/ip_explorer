""" Class used to define interface to complex models """

import abc
import itertools
import torch.nn
from loss_landscapes.model_interface.model_parameters import ModelParameters


class ModelWrapper(abc.ABC):
    def __init__(self, modules: list, layers=None):
        self.modules = modules
        self.layers  = layers

    def get_modules(self) -> list:
        return self.modules

    def get_module_parameters(self) -> ModelParameters:
        # return ModelParameters([p for module in self.modules for p in module.parameters()])

        return ModelParameters([
            p for module in self.modules for k, p in module.named_parameters()
            if self.layers is None or k in self.layers
        ])

    def train(self, mode=True) -> 'ModelWrapper':
        for module in self.modules:
            module.train(mode)
        return self

    def eval(self) -> 'ModelWrapper':
        return self.train(False)

    def requires_grad_(self, requires_grad=True) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                p.requires_grad = requires_grad
        return self

    def zero_grad(self) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        return self

    def parameters(self):
        # return itertools.chain([module.parameters() for module in self.modules])
        return itertools.chain([
            p for module in self.modules for k, p in module.named_parameters()
            if self.layers is None or k in self.layers
        ])

    def named_parameters(self):
        # return itertools.chain([module.named_parameters() for module in self.modules])
        return itertools.chain([
            (k,p) for module in self.modules for k, p in module.named_parameters()
            if self.layers is None or k in self.layers
        ])

    @abc.abstractmethod
    def forward(self, x):
        pass


class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module, layers):
        super().__init__([model], layers)

    def forward(self, x):
        return self.modules[0](x)


class GeneralModelWrapper(ModelWrapper):
    def __init__(self, model, modules: list, forward_fn):
        super().__init__(modules)
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)


def wrap_model(model):
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(False)
    elif isinstance(model, torch.nn.Module):
        return SimpleModelWrapper(model).requires_grad_(False)
    else:
        raise ValueError('Only models of type torch.nn.modules.module.Module can be passed without a wrapper.')
