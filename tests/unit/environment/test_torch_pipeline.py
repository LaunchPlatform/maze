import typing

import pytest
from torch import nn

from maze.environment import torch_pipeline
from maze.environment.torch_pipeline import build_pipeline
from maze.gene import pipeline


def module_type_kwargs(module: nn.Module) -> (typing.Type, dict):
    module_type = type(module)
    match module_type:
        case nn.ReLU | nn.LeakyReLU | nn.Tanh | nn.Flatten | nn.Softmax:
            return module_type, {}
        case torch_pipeline.Reshape:
            return module_type, dict(
                shape=module.shape,
            )
        case nn.Linear:
            return module_type, dict(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        case nn.AdaptiveAvgPool1d | nn.AdaptiveMaxPool1d:
            return module_type, dict(
                output_size=module.output_size,
            )
        case nn.Sequential:
            return module_type, dict(
                modules=list(map(module_type_kwargs, list(module.modules())[1:]))
            )
        case _:
            raise ValueError(f"Unexpected module type {module_type}")


@pytest.mark.parametrize(
    "module, expected",
    [
        (
            pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28)),
            nn.ReLU(),
        ),
        (
            pipeline.LeakyReLU(input_shape=(28, 28), output_shape=(28, 28)),
            nn.LeakyReLU(),
        ),
        (
            pipeline.Tanh(input_shape=(28, 28), output_shape=(28, 28)),
            nn.Tanh(),
        ),
        (
            pipeline.Softmax(input_shape=(28, 28), output_shape=(28, 28)),
            nn.Softmax(),
        ),
        (
            pipeline.Flatten(input_shape=(28, 28), output_shape=(28 * 28,)),
            nn.Flatten(),
        ),
        (
            pipeline.Reshape(input_shape=(28, 28), output_shape=(28 * 28,)),
            torch_pipeline.Reshape(
                28 * 28,
            ),
        ),
        (
            pipeline.AdaptiveAvgPool1d(
                input_shape=(123,), output_shape=(456,), out_features=456
            ),
            nn.AdaptiveAvgPool1d(456),
        ),
        (
            pipeline.AdaptiveMaxPool1d(
                input_shape=(123,), output_shape=(456,), out_features=456
            ),
            nn.AdaptiveMaxPool1d(456),
        ),
        (
            pipeline.Linear(
                input_shape=(28 * 28,),
                output_shape=(123,),
                in_features=28 * 28,
                out_features=123,
                bias=True,
            ),
            nn.Linear(in_features=28 * 28, out_features=123, bias=True),
        ),
        (
            pipeline.Sequential(
                input_shape=(28, 28),
                output_shape=(28, 28),
                modules=[
                    pipeline.ReLU(input_shape=(28, 28), output_shape=(28, 28)),
                    pipeline.Tanh(input_shape=(28, 28), output_shape=(28, 28)),
                ],
            ),
            nn.Sequential(nn.ReLU(), nn.Tanh()),
        ),
    ],
)
def test_build_pipeline(module: pipeline.Module, expected: nn.Module):
    torch_module = build_pipeline(module)
    assert module_type_kwargs(torch_module) == module_type_kwargs(expected)
