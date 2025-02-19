import dataclasses

from . import pipeline


@dataclasses.dataclass(frozen=True)
class Node:
    name: str
    attributes: list[tuple[str, str]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Edge:
    src: int
    dest: int
    label: str | None = None


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


def format_int(value: int) -> str:
    return f"{value:,}"


def extract_attrs(module: pipeline.Module) -> list[tuple[str, str]]:
    attrs = []
    match module:
        case (
            pipeline.ReLU()
            | pipeline.LeakyReLU()
            | pipeline.Tanh()
            | pipeline.Softmax()
            | pipeline.Flatten()
            | pipeline.Reshape()
        ):
            pass
        case pipeline.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        ):
            attrs.append(("in_features", format_int(in_features)))
            attrs.append(("out_features", format_int(out_features)))
            attrs.append(("bias", str(bias)))
        case pipeline.AdaptiveMaxPool1d(out_features=out_features):
            attrs.append(("out_features", format_int(out_features)))
        case pipeline.AdaptiveAvgPool1d(out_features=out_features):
            attrs.append(("out_features", format_int(out_features)))
        case pipeline.Joint(joint_type=joint_type):
            attrs.append(("joint_type", joint_type.value))

        case _:
            raise ValueError(f"Unknown module type {type(module)}")
    return attrs


def build_dag(module: pipeline.Module, prev_node: int, dag: DAG) -> int:
    match module:
        case (
            pipeline.ReLU()
            | pipeline.LeakyReLU()
            | pipeline.Tanh()
            | pipeline.Softmax()
            | pipeline.Flatten()
            | pipeline.Reshape()
            | pipeline.Linear()
            | pipeline.AdaptiveAvgPool1d()
            | pipeline.AdaptiveMaxPool1d()
        ):
            new_node = dag.add_node(
                Node(name=module.__class__.__name__, attributes=extract_attrs(module))
            )
        case pipeline.Sequential(modules=modules):
            for module in modules:
                prev_node = build_dag(module=module, prev_node=prev_node, dag=dag)
            return prev_node
        case pipeline.Joint(branches=branches):
            new_node = dag.add_node(
                Node(name="Joint", attributes=extract_attrs(module))
            )
            for branch_module in branches:
                dag.add_edge(
                    Edge(
                        src=build_dag(
                            module=branch_module, prev_node=prev_node, dag=dag
                        ),
                        dest=new_node,
                        label=repr(branch_module.output_shape),
                    )
                )
            return new_node
        case _:
            raise ValueError(f"Unknown module type {type(module)}")
    dag.add_edge(Edge(src=prev_node, dest=new_node, label=repr(module.input_shape)))
    return new_node
