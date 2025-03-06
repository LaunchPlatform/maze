import functools
import math
import typing
from functools import reduce

import torch
from torch import nn
from torch.nn import functional as fn

from ..gene import pipeline
from ..gene.symbols import JointType
from ..gene.symbols import SymbolType

JOINT_OP_FUNC_MAP: dict[JointType, typing.Callable] = {
    JointType.ADD: torch.add,
    JointType.SUB: torch.sub,
    JointType.MUL: torch.mul,
}


SIMPLE_MODULES_MAP = {
    # Non-linear Activations (weighted sum, nonlinearity)
    SymbolType.ELU: nn.ELU,
    SymbolType.HARDSHRINK: nn.Hardshrink,
    SymbolType.HARDSIGMOID: nn.Hardsigmoid,
    SymbolType.HARDTANH: nn.Hardtanh,
    SymbolType.HARDSWISH: nn.Hardswish,
    SymbolType.LEAKY_RELU: nn.LeakyReLU,
    SymbolType.LOGSIGMOID: nn.LogSigmoid,
    SymbolType.PRELU: nn.PReLU,
    SymbolType.RELU: nn.ReLU,
    SymbolType.RELU6: nn.ReLU6,
    SymbolType.RRELU: nn.RReLU,
    SymbolType.SIGMOID: nn.SELU,
    SymbolType.SOFTPLUS: nn.Softplus,
    SymbolType.SOFTSHRINK: nn.Softshrink,
    SymbolType.SOFTSIGN: nn.Softsign,
    SymbolType.TANH: nn.Tanh,
    SymbolType.TANHSHRINK: nn.Tanhshrink,
    # TODO:
    # SymbolType.THRESHOLD: nn.Threshold,
    # Non-linear Activations (other)
    SymbolType.SOFTMAX: nn.Softmax,
    SymbolType.SOFTMIN: nn.Softmin,
    SymbolType.LOGSOFTMAX: nn.Softmax,
}


class Joint(nn.Module):
    def __init__(self, branch_modules: list[nn.Module], joint_type: JointType):
        super().__init__()
        self.joint_type = joint_type
        self.branch_modules = branch_modules
        for i, module in enumerate(branch_modules):
            self.register_module(str(i), module)

    def extra_repr(self) -> str:
        return f"joint_type={self.joint_type.value}"

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
                padded_tensors.append(fn.pad(tensor, (0, delta), "constant", pad_value))
            op_func = JOINT_OP_FUNC_MAP[self.joint_type]
            return reduce(op_func, padded_tensors[1:], padded_tensors[0])


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def extra_repr(self) -> str:
        return f"shape={self.shape}"

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


def build_pipeline(
    module: pipeline.Module,
    module_learning_parameters: list[dict[str, typing.Any]] | None = None,
) -> nn.Module:
    match module:
        case pipeline.SimpleModule(symbol_type=symbol_type):
            return SIMPLE_MODULES_MAP[symbol_type]()
        case pipeline.Flatten():
            return nn.Flatten()
        case pipeline.Reshape(output_shape=output_shape):
            return Reshape(*output_shape)
        case pipeline.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            learning_parameters=learning_parameters,
        ):
            result = nn.Linear(
                bias=bias,
                in_features=in_features,
                out_features=out_features,
            )
            if module_learning_parameters is not None:
                module_learning_parameters.append(
                    dict(
                        params=result.parameters(),
                        lr=learning_parameters.lr,
                        momentum=learning_parameters.momentum,
                        dampening=learning_parameters.dampening,
                        weight_decay=learning_parameters.weight_decay,
                    )
                )
            return result
        case pipeline.Dropout(probability=probability):
            return nn.Dropout(probability)
        case pipeline.AdaptiveMaxPool1d(out_features=out_features):
            return nn.AdaptiveMaxPool1d(out_features)
        case pipeline.AdaptiveAvgPool1d(out_features=out_features):
            return nn.AdaptiveAvgPool1d(out_features)
        case pipeline.BatchNorm1d(
            input_shape=input_shape, eps=eps, momentum=momentum, affine=affine
        ):
            return nn.BatchNorm1d(
                num_features=math.prod(input_shape),
                eps=eps,
                momentum=momentum,
                affine=affine,
            )
        case pipeline.InstanceNorm1d(
            input_shape=input_shape, eps=eps, momentum=momentum, affine=affine
        ):
            return nn.InstanceNorm1d(
                num_features=math.prod(input_shape),
                eps=eps,
                momentum=momentum,
                affine=affine,
            )
        case pipeline.Sequential(modules=modules):
            return nn.Sequential(
                *map(
                    functools.partial(
                        build_pipeline,
                        module_learning_parameters=module_learning_parameters,
                    ),
                    modules,
                )
            )
        case pipeline.Joint(branches=branches, joint_type=joint_type):
            return Joint(
                branch_modules=list(
                    map(
                        functools.partial(
                            build_pipeline,
                            module_learning_parameters=module_learning_parameters,
                        ),
                        branches,
                    )
                ),
                joint_type=joint_type,
            )
        case _:
            raise ValueError(f"Unknown module type {type(module)}")
