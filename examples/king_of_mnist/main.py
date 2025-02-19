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
from maze.gene.freq_table import random_lookup
from maze.gene.merge import merge_gene
from maze.gene.merge import merge_parameter_dict
from maze.gene.mutation import decide_mutations
from maze.gene.mutation import mutate
from maze.gene.mutation import MutationType
from maze.gene.symbols import generate_gene
from maze.gene.symbols import SymbolParameterRange
from maze.gene.symbols import symbols_adapter

JITTER = 0.1
DEFAULT_MUTATION_PROBABILITIES = {
    MutationType.DUPLICATE: 0.01,
    MutationType.DELETE: 0.01,
    MutationType.REVERSE: 0.01,
}
MUTATION_LENGTH_RANGE = {
    MutationType.DUPLICATE: [1, 3],
    MutationType.DELETE: [1, 3],
    MutationType.REVERSE: [1, 3],
}

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
    reward_difficulty: int


def format_number(value: int) -> str:
    return f"{value:,}"


def to_millions(l: list[int]) -> list[int]:
    return list(map(lambda x: x * 1_000_000, l))


def enum_key_to_str(value: dict) -> dict:
    return {key.value: value for key, value in value.items()}


class KingOfMnistV2(LinearEnvironment):
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
        reward = to_millions([100, 100, 100, 100, 100])[index]
        reward_difficulty = [5, 6, 7, 8, 9][index]
        epoch = [10, 25, 50, 75, 100][index]
        return dataclasses.asdict(
            Arguments(
                epoch=epoch,
                initial_credit=100 * mi,
                basic_cost=basic_cost,
                reward=reward,
                reward_difficulty=reward_difficulty,
            )
        )

    def initialize_zone(self, zone: models.Zone, period: models.Period):
        if zone.environment.index != 0:
            # we only want to populate first environment
            return
        db = object_session(zone)
        for _ in range(zone.agent_slots):
            gene_length = random.randint(5, 100)
            symbols = list(
                generate_gene(
                    length=gene_length,
                    param_range=SymbolParameterRange(),
                )
            )
            agent = models.Agent(
                gene=symbols_adapter.dump_python(symbols, mode="json"),
                input_shape=[28, 28],
                mutation_probabilities=enum_key_to_str(DEFAULT_MUTATION_PROBABILITIES),
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
        avatar.initial_credit = args.initial_credit
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
                args.reward
                * (
                    (epoch.test_correct_count / epoch.test_total_count)
                    ** args.reward_difficulty
                )
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

        lookup_table = build_lookup_table(
            [(agent.id, credit) for agent, credit in agent_credits]
        )
        total_slots = zone.agent_slots
        offspring_slots = total_slots * 0.6

        offspring_agents = []
        for _ in range(int(offspring_slots)):
            lhs = random_lookup(lookup_table)

            # TODO: well, this is not the most performant way to do it. could find a time to improve it later
            excluded_agent_credits = list(
                filter(lambda agent: agent != lhs, agent_credits)
            )
            excluded_lookup_table = build_lookup_table(
                [(agent.id, credit) for agent, credit in excluded_agent_credits]
            )
            rhs = random_lookup(excluded_lookup_table)

            lhs = db.get(models.Agent, lhs)
            rhs = db.get(models.Agent, rhs)

            lhs_gene = lhs.agent_data.symbols
            rhs_gene = rhs.agent_data.symbols

            gene = list(merge_gene(lhs_gene, rhs_gene, jitter=JITTER))
            mutation_probabilities = merge_parameter_dict(
                lhs=lhs.enum_mutation_probabilities,
                rhs=rhs.enum_mutation_probabilities,
                jitter=JITTER,
            )
            mutation_types = decide_mutations(
                probabilities=mutation_probabilities,
                gene_length=len(gene),
            )
            mutation_records, mutated_gene = mutate(
                symbols=gene,
                mutations=mutation_types,
                length_ranges=MUTATION_LENGTH_RANGE,
            )
            new_agent = models.Agent(
                lhs_parent=lhs,
                rhs_parent=rhs,
                gene=symbols_adapter.dump_python(mutated_gene, mode="json"),
                input_shape=lhs.input_shape,
                mutation_probabilities={
                    key.value: value for key, value in mutation_probabilities.items()
                },
                mutations=[
                    models.Mutation(
                        order=index,
                        type=record.type,
                        position=record.position,
                        length=record.length,
                    )
                    for index, record in enumerate(mutation_records)
                ],
            )
            offspring_agents.append(new_agent)
        return offspring_agents

    def promote_agents(
        self,
        environment: models.Environment | None,
        period: models.Period,
        agent_count: int,
    ) -> list[models.Agent]:
        db = object_session(period)
        agents = []
        if environment is None:
            # no source env, it means this is the first env.
            # let's fill the slots with random new ones
            for _ in range(agent_count):
                gene_length = random.randint(5, 100)
                symbols = list(
                    generate_gene(
                        length=gene_length,
                        param_range=SymbolParameterRange(),
                    )
                )
                agent = models.Agent(
                    gene=symbols_adapter.dump_python(symbols, mode="json"),
                    input_shape=[28, 28],
                    mutation_probabilities=enum_key_to_str(
                        DEFAULT_MUTATION_PROBABILITIES
                    ),
                )
                agents.append(agent)
            return agents
        return (
            (
                db.query(models.Agent)
                .join(models.Avatar, models.Avatar.agent_id == models.Agent.id)
                .join(models.Zone, models.Avatar.zone_id == models.Zone.id)
                .filter(models.Zone.environment == environment)
                .filter(models.Avatar.period == period)
                .filter(models.Avatar.credit > 0)
                .filter(models.Avatar.status == models.AvatarStatus.DEAD)
                .order_by(models.Avatar.credit.desc())
            )
            .limit(agent_count)
            .all()
        )
