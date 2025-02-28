import pytest

from maze.gene import pipeline
from maze.gene.dag import build_dag
from maze.gene.dag import DAG
from maze.gene.dag import Edge
from maze.gene.dag import Node
from maze.gene.symbols import JointType
from maze.gene.symbols import LearningParameters


@pytest.mark.parametrize(
    "module, expected_nodes, expected_edges",
    [
        (
            pipeline.Sequential(
                input_shape=(28, 28),
                output_shape=(28 * 28,),
                modules=[
                    pipeline.Flatten(
                        input_shape=(28, 28),
                        output_shape=(28 * 28,),
                    ),
                    pipeline.Linear(
                        input_shape=(28 * 28,),
                        output_shape=(123,),
                        in_features=28 * 28,
                        out_features=123,
                        bias=True,
                        learning_parameters=LearningParameters(
                            lr=0.01,
                            momentum=0.02,
                            dampening=0.03,
                            weight_decay=0.04,
                        ),
                    ),
                    pipeline.Tanh(
                        input_shape=(123,),
                        output_shape=(123,),
                    ),
                    pipeline.ReLU(
                        input_shape=(123,),
                        output_shape=(123,),
                    ),
                ],
            ),
            [
                Node(name="INPUT"),
                Node(name="Flatten"),
                Node(
                    name="Linear",
                    attributes=[
                        (
                            "in_features",
                            "784",
                        ),
                        (
                            "out_features",
                            "123",
                        ),
                        (
                            "bias",
                            "True",
                        ),
                        (
                            "lr",
                            "1.00E-02",
                        ),
                        (
                            "momentum",
                            "2.00E-02",
                        ),
                        (
                            "dampening",
                            "3.00E-02",
                        ),
                        (
                            "weight_decay",
                            "4.00E-02",
                        ),
                    ],
                ),
                Node(name="Tanh"),
                Node(name="ReLU"),
            ],
            [
                Edge(src=0, dest=1, label=repr((28, 28))),
                Edge(src=1, dest=2, label=repr((784,))),
                Edge(src=2, dest=3, label=repr((123,))),
                Edge(src=3, dest=4, label=repr((123,))),
            ],
        ),
        (
            pipeline.Joint(
                joint_type=JointType.ADD,
                input_shape=(28, 28),
                output_shape=(28, 28),
                branches=[
                    pipeline.Sequential(
                        input_shape=(28, 28),
                        output_shape=(28 * 28,),
                        modules=[
                            pipeline.Flatten(
                                input_shape=(28, 28),
                                output_shape=(28 * 28,),
                            ),
                            pipeline.Linear(
                                input_shape=(28 * 28,),
                                output_shape=(123,),
                                in_features=28 * 28,
                                out_features=123,
                                bias=True,
                                learning_parameters=LearningParameters(
                                    lr=0.01,
                                    momentum=0.02,
                                    dampening=0.03,
                                    weight_decay=0.04,
                                ),
                            ),
                        ],
                    ),
                    pipeline.Sequential(
                        input_shape=(28, 28),
                        output_shape=(28, 28),
                        modules=[],
                    ),
                ],
            ),
            [
                Node(name="INPUT"),
                Node(
                    name="Joint",
                    attributes=[
                        (
                            "joint_type",
                            "ADD",
                        ),
                    ],
                ),
                Node(name="Flatten"),
                Node(
                    name="Linear",
                    attributes=[
                        (
                            "in_features",
                            "784",
                        ),
                        (
                            "out_features",
                            "123",
                        ),
                        (
                            "bias",
                            "True",
                        ),
                        (
                            "lr",
                            "1.00E-02",
                        ),
                        (
                            "momentum",
                            "2.00E-02",
                        ),
                        (
                            "dampening",
                            "3.00E-02",
                        ),
                        (
                            "weight_decay",
                            "4.00E-02",
                        ),
                    ],
                ),
            ],
            [
                Edge(src=0, dest=2, label=repr((28, 28))),
                Edge(src=2, dest=3, label=repr((784,))),
                Edge(src=3, dest=1, label=repr((784,))),
                Edge(src=0, dest=1, label=repr((28, 28))),
            ],
        ),
    ],
)
def test_build_dag(
    module: pipeline.Module, expected_nodes: list[Node], expected_edges: list[Edge]
):
    dag = DAG()
    input_node = dag.add_node(Node(name="INPUT"))
    build_dag(module=module, prev_node=input_node, dag=dag)
    assert dag.nodes == expected_nodes
    assert frozenset(dag.edges) == frozenset(expected_edges)
