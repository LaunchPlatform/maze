import typing
from functools import reduce

import torch
from torch import nn
from torch.nn import functional

from ..gene import pipeline
from ..gene.symbols import JointType

JOINT_OP_FUNC_MAP: dict[JointType, typing.Callable] = {
    JointType.ADD: torch.add,
    JointType.SUB: torch.sub,
    JointType.MUL: torch.mul,
}


class Joint(nn.Module):
    def __init__(self, branch_modules: list[nn.Module], joint_type: JointType):
        super().__init__()
        self.joint_type = joint_type
        self.branch_modules = branch_modules
        for i, module in enumerate(branch_modules):
            self.register_module(str(i), module)

    def forward(self, x):
        tensors = list((module(x) for module in self.branch_modules))
        if self.joint_type == JointType.CONCAT:
            return torch.cat(
                tensors,
                dim=1,
            )
        else:
            max_length = max(*map(lambda t: t.size(1), tensors))
            padded_tensors = []
            pad_value = 0
            if self.joint_type == JointType.MUL:
                # pad 1 for mul to keep values from the other branches
                pad_value = 1
            for tensor in tensors:
                if tensor.size(1) == max_length:
                    padded_tensors.append(tensor)
                    continue
                delta = max_length - tensor.size(1)
                padded_tensors.append(
                    functional.pad(tensor, (0, delta), "constant", pad_value)
                )
            op_func = JOINT_OP_FUNC_MAP[self.joint_type]
            return reduce(op_func, padded_tensors[1:], padded_tensors[0])


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


def build_pipeline(module: pipeline.Module) -> nn.Module:
    match module:
        case pipeline.ReLU():
            return nn.ReLU()
        case pipeline.LeakyReLU():
            return nn.LeakyReLU()
        case pipeline.Tanh():
            return nn.Tanh()
        case pipeline.Softmax():
            return nn.Softmax()
        case pipeline.Flatten():
            return nn.Flatten()
        case pipeline.Reshape(output_shape=output_shape):
            return Reshape(*output_shape)
        case pipeline.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        ):
            return nn.Linear(
                bias=bias,
                in_features=in_features,
                out_features=out_features,
            )
        case pipeline.AdaptiveMaxPool1d(out_features=out_features):
            return nn.AdaptiveMaxPool1d(out_features)
        case pipeline.AdaptiveAvgPool1d(out_features=out_features):
            return nn.AdaptiveAvgPool1d(out_features)
        case pipeline.Sequential(modules=modules):
            return nn.Sequential(*map(build_pipeline, modules))
        case pipeline.Joint(branches=branches, joint_type=joint_type):
            return Joint(
                branch_modules=list(map(build_pipeline, branches)),
                joint_type=joint_type,
            )
        case _:
            raise ValueError(f"Unknown module type {type(module)}")
