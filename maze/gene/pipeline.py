import dataclasses

from .symbols import JointType
from .symbols import LearningParameters


@dataclasses.dataclass(frozen=True)
class Module:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Sequential(Module):
    modules: list[Module]


@dataclasses.dataclass(frozen=True)
class Flatten(Module):
    pass


@dataclasses.dataclass(frozen=True)
class Reshape(Module):
    pass


@dataclasses.dataclass(frozen=True)
class Joint(Module):
    branches: list[Module]
    joint_type: JointType


@dataclasses.dataclass(frozen=True)
class ReLU(Module):
    pass


@dataclasses.dataclass(frozen=True)
class LeakyReLU(Module):
    pass


@dataclasses.dataclass(frozen=True)
class Tanh(Module):
    pass


@dataclasses.dataclass(frozen=True)
class Softmax(Module):
    pass


@dataclasses.dataclass(frozen=True)
class Linear(Module):
    bias: bool
    in_features: int
    out_features: int
    learning_parameters: LearningParameters


@dataclasses.dataclass(frozen=True)
class AdaptiveMaxPool1d(Module):
    out_features: int


@dataclasses.dataclass(frozen=True)
class AdaptiveAvgPool1d(Module):
    out_features: int
