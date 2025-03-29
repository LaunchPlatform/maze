import dataclasses
import itertools
import uuid

from fastapi import APIRouter
from fastapi import Request
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.orm import aliased
from sqlalchemy.orm import joinedload
from sqlalchemy.orm import undefer

from .. import deps
from ... import models

router = APIRouter(tags=["zone"])


@dataclasses.dataclass
class PeriodStats:
    period: models.Period
    alive_avatars: int
    dead_avatars: int


@router.get("/zones/{id}")
def view_zone(
    request: Request,
    templates: deps.Jinja2TemplatesDep,
    db: deps.SessionDeps,
    id: uuid.UUID,
):
    zone = db.get(models.Zone, id)
    if zone is None:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    aliased_avatar = aliased(models.Avatar)
    alive_avatar_count = (
        select(func.count())
        .select_from(aliased_avatar)
        .where(
            and_(
                aliased_avatar.period_id == models.Period.id,
                aliased_avatar.zone_id == zone.id,
                aliased_avatar.status == models.AvatarStatus.ALIVE,
            )
        )
    ).scalar_subquery()

    aliased_avatar = aliased(models.Avatar)
    dead_avatar_count = (
        select(func.count())
        .select_from(aliased_avatar)
        .where(
            and_(
                aliased_avatar.period_id == models.Period.id,
                aliased_avatar.zone_id == zone.id,
                aliased_avatar.status != models.AvatarStatus.ALIVE,
            )
        )
    ).scalar_subquery()

    period_avatars_query = (
        db.query(models.Period, alive_avatar_count, dead_avatar_count, models.Avatar)
        .select_from(models.Period)
        .options(undefer(models.Avatar.credit), joinedload(models.Avatar.agent))
        .join(models.Experiment, models.Period.experiment_id == models.Experiment.id)
        .join(
            models.Environment, models.Environment.experiment_id == models.Experiment.id
        )
        .join(
            models.Zone,
            and_(
                models.Zone.environment_id == models.Environment.id,
                models.Zone.id == zone.id,
            ),
        )
        .join(
            models.Avatar,
            and_(
                models.Avatar.period_id == models.Period.id,
                models.Avatar.zone_id == zone.id,
            ),
        )
        .order_by(
            models.Period.index.desc(),
            models.Avatar.credit.desc().nullslast(),
            models.Avatar.status,
        )
    )

    period_avatars = [
        (period_stats, list(map(lambda item: item[1], group)))
        for period_stats, group in itertools.groupby(
            (
                (
                    PeriodStats(
                        period=period,
                        alive_avatars=alive_count,
                        dead_avatars=dead_count,
                    ),
                    avatar,
                )
                for period, alive_count, dead_count, avatar in period_avatars_query
            ),
            key=lambda item: item[0],
        )
    ]

    return templates.TemplateResponse(
        "zone/view_zone.html",
        dict(
            request=request,
            zone=zone,
            period_avatars=period_avatars,
        ),
    )
