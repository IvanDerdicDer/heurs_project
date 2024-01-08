import math
import random
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from functools import partial, lru_cache, cache
from itertools import pairwise
from typing import Self, Iterable, Optional

from instance_parser import Customer, Instance


@dataclass(frozen=True, eq=True)
class ExtendedCustomer:
    customer: Customer
    time: int
    capacity: int


@dataclass()
class Ant:
    ttl: int
    path: list[ExtendedCustomer] = field(default_factory=list)
    time: int = 0
    capacity: int = 0
    distance: float = 0


@dataclass
class Solution:
    routes: list[Ant]
    vehicle_count: int

    def edges(self) -> Iterable[tuple[Customer, Customer]]:
        old_edges: list[tuple[Customer, Customer]] = []
        for route in self.routes:
            for edge in pairwise(route.path):
                if edge not in old_edges:
                    yield edge
                else:
                    old_edges.append(edge)


class Pheromones:
    def __init__(
            self,
            decay_rate: float,
            cost_function: Callable[[Solution], float],
            reinforcement_function: Callable[[float], float],
    ):
        self.decay_rate = decay_rate
        self.cost_function = cost_function
        self.reinforcement_function = reinforcement_function
        self.pheromones: dict[tuple[Customer, Customer], float] = dict()

    def __getattr__(self, item: tuple[Customer, Customer]) -> float:
        return self.pheromones.get(item, 1)

    def evaporation(self) -> Self:
        for edge in self.pheromones:
            self.pheromones[edge] = (1 - self.decay_rate) * self.pheromones.get(edge, 1)

        return self

    def reinforce(self, solution: Solution) -> Self:
        for edge in solution.edges():
            self.pheromones[edge] = self.reinforcement_function(self.cost_function(solution)) + self.pheromones.get(
                edge, 1)

        return self


def cost_function(
        solution: Solution,
        instance: Instance
) -> float:
    theoretical_min = math.ceil(sum(i.demand for i in instance.customers) / instance.vehicle_capacity)

    cost = theoretical_min / len(solution.routes)

    max_distance = max(i.distance for i in solution.routes)

    cost -= sum(i.distance / max_distance for i in solution.routes) / len(solution.routes)

    return cost


@cache
def reinforcement_function(cost: float) -> float:
    return 1 / cost


@cache
def calculate_distance(
        c1: Customer,
        c2: Customer
) -> float:
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def get_viable_next(
        instance: Instance,
        ant: Ant
) -> list[Customer]:
    to_return = []

    for customer in instance.customers:
        if (
                (
                        customer.ready_time
                        <= ant.time
                        + math.ceil(calculate_distance(ant.path[-1].customer, customer))
                        <= customer.due_date
                )
                and customer not in ant.path
                and (
                ant.time
                + math.ceil(calculate_distance(ant.path[-1].customer, customer))
                + math.ceil(calculate_distance(ant.path[0].customer, customer))
                <= ant.path[0].customer.due_date
        )
        ):
            to_return.append(customer)

    return to_return


@cache
def customer_weight(
        start_customer: Customer,
        end_customer: Customer
) -> float:
    return (
            1 / calculate_distance(start_customer, end_customer)
            + end_customer.service_time
    )


def ant_colony(
        instance: Instance,
        timeout_minutes: int = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
):
    pheromones = Pheromones(
        0.3,
        partial(cost_function, instance=instance),
        reinforcement_function
    )

    customers = instance.customers.copy()
    depot = customers.pop(0)

    start_time = time.time()

    ants: list[Ant] = []
    while time.time() - start_time <= timeout_minutes:
        ant: Ant = Ant(depot.due_date, [ExtendedCustomer(depot, 0, 0)])
        while viable_next := get_viable_next(instance, ant):
            next_customers = random.sample([
                i
                for i in viable_next
                if random.random() <= (
                    pheromones[(ant.path[-1].customer, i)] ** alpha
                    * customer_weight(ant.path[-1].customer, i) ** beta
                    / sum(pheromones[j] ** alpha * customer_weight(*j) ** beta for j in pheromones.pheromones)
                )
            ], 1)
            if not next_customers:
                next_customer = depot
            else:
                next_customer = next_customers[0]




