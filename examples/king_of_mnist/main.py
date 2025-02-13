import dataclasses
import logging
import random
import typing

from sqlalchemy.orm import object_session
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from maze import models
from maze.environment.templates import LinearEnvironment
from maze.environment.vehicle import Vehicle
from maze.environment.zone import EpochReport
from maze.environment.zone import eval_agent
from maze.environment.zone import OutOfCreditError
from maze.gene.freq_table import build_lookup_table
from maze.gene.freq_table import gen_freq_table
from maze.gene.freq_table import random_lookup
from maze.gene.merge import JiterConfig
from maze.gene.merge import merge_gene
from maze.gene.symbols import generate_gene
from maze.gene.symbols import SymbolParameterRange
from maze.gene.symbols import symbols_adapter
from maze.gene.symbols import SymbolType

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


@dataclasses.dataclass(frozen=True)
class Arguments:
    epoch: int
    initial_credit: int
    basic_cost: int
    reward: int


def format_number(value: int) -> str:
    return f"{value:,}"


def to_millions(l: list[int]) -> list[int]:
    return list(map(lambda x: x * 1_000_000, l))


class KingOfMnist(LinearEnvironment):
    count = 5
    group = "king-of-mnist"
    experiment = "king-of-mnist"

    def make_zones(self, index: int) -> list[models.Zone]:
        zone_count = [10, 5, 2, 1, 1][index]
        return [
            models.Zone(agent_slots=30, index=zone_index)
            for zone_index in range(zone_count)
        ]

    def make_arguments(self, index: int) -> dict | None:
        mi = 1_000_000
        basic_cost = to_millions([1, 2, 3, 4, 5])[index]
        reward = to_millions([100, 90, 80, 70, 60])[index]
        return dataclasses.asdict(
            Arguments(
                epoch=[10, 30, 50, 70, 90][index],
                initial_credit=100 * mi,
                basic_cost=basic_cost,
                reward=reward,
            )
        )

    def initialize_zone(self, zone: models.Zone, period: models.Period):
        if zone.environment.index != 0:
            # we only want to populate first environment
            return
        db = object_session(zone)
        for _ in range(zone.agent_slots):
            # TODO: do we really need this?
            symbol_table = gen_freq_table(
                symbols=list(SymbolType), random_range=(1, 1024)
            )
            gene_length = random.randint(5, 100)
            symbols = list(
                generate_gene(
                    symbol_table=symbol_table,
                    length=gene_length,
                    param_range=SymbolParameterRange(),
                )
            )
            agent = models.Agent(
                gene=symbols_adapter.dump_python(symbols, mode="json"),
                symbol_table={},
                input_shape=[28, 28],
            )
            db.add(agent)
            avatar = models.Avatar(
                period=period,
                agent=agent,
                zone=zone,
            )
            db.add(avatar)

    def run_avatar(
        self, avatar: models.Avatar
    ) -> typing.Generator[EpochReport, None, None]:
        db = object_session(avatar)
        logger = logging.getLogger(__name__)
        args = Arguments(**avatar.zone.environment.arguments)

        vehicle = Vehicle(
            agent=avatar.agent.agent_data,
            loss_fn=nn.CrossEntropyLoss(),
        )
        vehicle.build_models()

        # TODO: DRY these?
        avatar.agent.op_cost = vehicle.model.cost.operation
        avatar.agent.build_cost = vehicle.model.cost.build
        avatar.agent.parameters_count = vehicle.parameter_count()
        db.add(avatar.agent)
        logger.info(
            "Built avatar %s model with build_cost=%s, op_cost=%s, parameters_count=%s",
            avatar.id,
            format_number(avatar.agent.build_cost),
            format_number(avatar.agent.op_cost),
            format_number(avatar.agent.parameters_count),
        )
        logger.info("Avatar %s PyTorch Model:\n%r", avatar.id, vehicle.torch_model)

        credit = args.initial_credit
        avatar.credit = args.initial_credit
        db.add(avatar)

        epoch_cost = avatar.agent.op_cost + args.basic_cost
        if epoch_cost > credit:
            raise OutOfCreditError("Cost exceeds initial credit")
        for epoch in eval_agent(
            vehicle=vehicle,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=args.epoch,
        ):
            epoch.cost = epoch_cost
            epoch.income = int(
                args.reward * ((epoch.test_correct_count / epoch.test_total_count) ** 5)
            )
            credit += epoch.income - epoch.cost
            logger.info("Avatar remaining credit: %s", format_number(credit))
            if credit < 0:
                raise OutOfCreditError("Out of credit")
            yield epoch

    def breed_agents(
        self,
        zone: models.Zone,
        period: models.Period,
    ) -> list[models.Agent]:
        db = object_session(zone)

        agent_credits = (
            db.query(models.Agent, models.Avatar.credit)
            .select_from(models.Agent)
            .join(models.Avatar, models.Avatar.agent_id == models.Agent.id)
            .filter(models.Avatar.zone == zone)
            .filter(models.Avatar.status == models.AvatarStatus.DEAD)
            .filter(models.Avatar.period == period)
            .filter(models.Avatar.credit > 0)
        ).all()
        if len(agent_credits) <= 1:
            return []

        lookup_table = build_lookup_table(agent_credits)
        total_slots = zone.agent_slots
        offspring_slots = total_slots * 0.7

        offspring_agents = []
        for _ in range(int(offspring_slots)):
            lhs = random_lookup(lookup_table)

            # TODO: well, this is not the most performant way to do it. could find a time to improve it later
            excluded_agent_credits = list(
                filter(lambda agent: agent != lhs, agent_credits)
            )
            excluded_lookup_table = build_lookup_table(excluded_agent_credits)
            rhs = random_lookup(excluded_lookup_table)
            lhs_gene = lhs.agent_data.symbols
            rhs_gene = rhs.agent_data.symbols
            gene = list(merge_gene(lhs_gene, rhs_gene, jiter_config=JiterConfig()))
            # TODO: mutations
            new_agent = models.Agent(
                lhs_parent=lhs,
                rhs_parent=rhs,
                gene=symbols_adapter.dump_python(gene, mode="json"),
                input_shape=lhs.input_shape,
                # TODO: remove this?
                symbol_table={},
            )
            offspring_agents.append(new_agent)
        return offspring_agents

    def promote_agents(
        self,
        from_env: models.Environment | None,
        to_env: models.Environment,
        period: models.Period,
        agent_count: int,
    ) -> list[models.Agent]:
        db = object_session(from_env)
        agents = []
        if from_env is None:
            # no source env, it means this is the first env.
            # let's fill the slots with random new ones
            for _ in range(agent_count):
                # TODO: do we really need this?
                symbol_table = gen_freq_table(
                    symbols=list(SymbolType), random_range=(1, 1024)
                )
                gene_length = random.randint(5, 100)
                symbols = list(
                    generate_gene(
                        symbol_table=symbol_table,
                        length=gene_length,
                        param_range=SymbolParameterRange(),
                    )
                )
                agent = models.Agent(
                    gene=symbols_adapter.dump_python(symbols, mode="json"),
                    symbol_table={},
                    input_shape=[28, 28],
                )
                agents.append(agent)
            return agents
        return (
            (
                db.query(models.Agent)
                .join(models.Avatar, models.Avatar.agent_id == models.Agent.id)
                .join(models.Zone, models.Avatar.zone_id == models.Zone.id)
                .filter(models.Zone.environment == from_env)
                .filter(models.Avatar.credit > 0)
                .order_by(models.Avatar.credit.desc())
            )
            .limit(agent_count)
            .all()
        )
