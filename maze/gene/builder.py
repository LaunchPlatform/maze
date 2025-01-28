from torch import nn

from .symbols import Symbol


def build_models(symbols: list[Symbol]) -> list[nn.Module]:
    modules = []
    for item in symbols:
        match item:
            case (Symbol.REPEAT_START, times):
                pass
            case (Symbol.LINEAR, output_features):
                modules.append(nn.LazyLinear(out_features=output_features))
            case Symbol.RELU:
                modules.append(nn.ReLU())
            case Symbol.LEAKY_RELU:
                modules.append(nn.LeakyReLU())
            case Symbol.TANH:
                modules.append(nn.Tanh())
    return modules
