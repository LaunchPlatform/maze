import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException

from .. import deps
from ... import models
from ...gene import pipeline
from ...gene.builder import build_models
from ...gene.builder import ExceedBuildBudgetError
from ...gene.builder import ExceedOperationBudgetError
from ...gene.builder import ModelCost
from ...gene.dag import build_dag
from ...gene.dag import DAG
from ...gene.dag import Edge
from ...gene.dag import Node

router = APIRouter(tags=["agent"])


@router.get("/agents/{id}")
def view_agent(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    agent: models.Agent | None = db.get(models.Agent, id)
    if agent is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    agent_data = agent.agent_data
    dag = None
    try:
        # TODO: cache these
        model = build_models(
            symbols=iter(agent_data.symbols),
            input_shape=agent_data.input_shape,
            budget=ModelCost(operation=100_000_000, build=1_000),
        )
        dag = DAG()
        input_node = dag.add_node(Node(name="INPUT"))
        tail_node = build_dag(
            module=pipeline.Sequential(
                input_shape=agent_data.input_shape,
                output_shape=model.output_shape,
                modules=model.modules,
            ),
            prev_node=input_node,
            dag=dag,
        )
        output_node = dag.add_node(Node(name="OUTPUT"))
        dag.add_edge(Edge(src=tail_node, dest=output_node))
    except ExceedOperationBudgetError | ExceedBuildBudgetError:
        pass

    return templates.TemplateResponse(
        "agent/view_agent.html",
        dict(
            request=request,
            agent=agent,
            dag=dag,
        ),
    )
