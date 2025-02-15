import dataclasses

from .... import models


@dataclasses.dataclass(frozen=True)
class ZoneNode:
    id: str
    parent_env_id: str
    name: str


@dataclasses.dataclass(frozen=True)
class EnvironmentNode:
    id: str
    name: str


@dataclasses.dataclass(frozen=True)
class Edge:
    src: str
    dest: str


@dataclasses.dataclass
class DAG:
    zone_nodes: list[ZoneNode] = dataclasses.field(default_factory=list)
    env_nodes: list[EnvironmentNode] = dataclasses.field(default_factory=list)
    edges: list[Edge] = dataclasses.field(default_factory=list)


def build_dag(experiment: models.Experiment) -> DAG:
    dag = DAG()
    # TODO: we only have linear environment for now, let's support free form of env connections in the future
    prev_env = None
    for environment in experiment.environments:
        dag.env_nodes.append(
            EnvironmentNode(id=str(environment.id), name=environment.name)
        )
        for zone in environment.zones:
            dag.zone_nodes.append(
                ZoneNode(
                    id=str(zone.id),
                    name=f"Zone {zone.index}",
                    parent_env_id=str(environment.id),
                )
            )
        if prev_env is not None:
            dag.edges.append(Edge(src=str(prev_env.id), dest=str(environment.id)))
        prev_env = environment
    return dag
