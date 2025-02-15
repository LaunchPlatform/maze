import pytest

from maze.gene import pipeline
from maze.web.dag import build_dag
from maze.web.dag import DAG
from maze.web.dag import Edge
from maze.web.dag import Node


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
        )
    ],
)
def test_build_dag(
    module: pipeline.Module, expected_nodes: list[Node], expected_edges: list[Edge]
):
    dag = DAG()
    input_node = dag.add_node(Node(name="INPUT"))
    build_dag(module=module, prev_node=input_node, dag=dag)
    assert dag.nodes == expected_nodes
    assert dag.edges == expected_edges
