import dataclasses

from ..gene import pipeline


@dataclasses.dataclass(frozen=True)
class Node:
    name: str


@dataclasses.dataclass(frozen=True)
class Edge:
    src: int
    dest: int


class DAG:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> int:
        sn = len(self.nodes)
        self.nodes.append(node)
        return sn

    def add_edge(self, edge: Edge):
        self.edges.append(edge)


def build_dag(module: pipeline.Module, prev_node: int, dag: DAG) -> int:
    match module:
        case pipeline.ReLU():
            new_node = dag.add_node(Node(name="ReLu"))
        case pipeline.LeakyReLU():
            new_node = dag.add_node(Node(name="LeakyReLu"))
        case pipeline.Tanh():
            new_node = dag.add_node(Node(name="Tanh"))
        case pipeline.Softmax():
            new_node = dag.add_node(Node(name="Softmax"))
        case pipeline.Flatten():
            new_node = dag.add_node(Node(name="Flatten"))
        case pipeline.Reshape(output_shape=output_shape):
            new_node = dag.add_node(Node(name="Reshape"))
        case pipeline.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        ):
            new_node = dag.add_node(Node(name="Linear"))
            # TODO:
        case pipeline.AdaptiveMaxPool1d(out_features=out_features):
            new_node = dag.add_node(Node(name="AdaptiveMaxPool1d"))
            # TODO:
        case pipeline.AdaptiveAvgPool1d(out_features=out_features):
            new_node = dag.add_node(Node(name="AdaptiveAvgPool1d"))
        # TODO:
        case pipeline.Sequential(modules=modules):
            for module in modules:
                prev_node = build_dag(module=module, prev_node=prev_node, dag=dag)
            return prev_node
        case pipeline.Joint(branches=branches):
            new_node = dag.add_node(Node(name="Joint"))
            for branch_module in branches:
                dag.add_edge(
                    Edge(
                        src=build_dag(
                            module=branch_module, prev_node=prev_node, dag=dag
                        ),
                        dest=new_node,
                    )
                )
        case _:
            raise ValueError(f"Unknown module type {type(module)}")
    dag.add_edge(Edge(src=prev_node, dest=new_node))
    return new_node
