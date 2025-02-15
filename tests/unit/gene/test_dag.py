import pytest

from maze.gene import pipeline
from maze.gene.dag import build_dag
from maze.gene.dag import DAG
from maze.gene.dag import Edge
from maze.gene.dag import Node


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
                Node(name="Linear"),
                Node(name="Tanh"),
                Node(name="ReLu"),
            ],
            [
                Edge(src=0, dest=1),
                Edge(src=1, dest=2),
                Edge(src=2, dest=3),
                Edge(src=3, dest=4),
            ],
        ),
        (
            pipeline.Joint(
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
                Node(name="Joint"),
                Node(name="Flatten"),
                Node(name="Linear"),
            ],
            [
                Edge(src=0, dest=2),
                Edge(src=2, dest=3),
                Edge(src=3, dest=1),
                Edge(src=0, dest=1),
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
