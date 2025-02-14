import torch
from torch import nn


class Joint(nn.Module):
    def __init__(self, branch_modules: list[nn.Module]):
        super().__init__()
        self.branch_modules = branch_modules
        for i, module in enumerate(branch_modules):
            self.register_module(str(i), module)

    def forward(self, x):
        # TODO: provide other ways of joining branches
        return torch.cat(
            list((module(x) for module in self.branch_modules)),
            dim=1,
        )


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
