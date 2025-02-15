import dataclasses

from .... import models


@dataclasses.dataclass(frozen=True)
class Zone:
    index: str
    progress: float | None


@dataclasses.dataclass(frozen=True)
class EnvironmentNode:
    id: str
    name: str
    zones: list[Zone]


@dataclasses.dataclass(frozen=True)
class Edge:
    src: str
    dest: str


@dataclasses.dataclass
class DAG:
    nodes: list[EnvironmentNode] = dataclasses.field(default_factory=list)
    edges: list[Edge] = dataclasses.field(default_factory=list)


def build_dag(experiment: models.Experiment) -> DAG:
    dag = DAG()
    # TODO: we only have linear environment for now, let's support free form of env connections in the future
    prev_env = None
    for environment in experiment.environments:
        zones = []
        for zone in environment.zones:
            progress = None
            total_avatars = zone.current_alive_avatars + zone.current_dead_avatars
            if total_avatars > 0:
                progress = zone.current_dead_avatars / total_avatars
            zones.append(Zone(index=zone.index, progress=progress))
        dag.nodes.append(
            EnvironmentNode(
                id=str(environment.id),
                name=environment.name,
                zones=zones,
            )
        )
        if prev_env is not None:
            dag.edges.append(Edge(src=str(prev_env.id), dest=str(environment.id)))
        prev_env = environment
    return dag
