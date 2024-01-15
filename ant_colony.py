import math
import random
from time import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial, cache
from itertools import pairwise, combinations
from typing import Self, Iterable, Sequence

from instance_parser import Customer, Instance, parse_instance


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

    def pretty_path(self) -> str:
        return '->'.join(f"{i.customer.number}({i.time})" for i in self.path)


@dataclass
class Solution:
    routes: list[Ant]
    vehicle_count: int

    def edges(self) -> Iterable[tuple[Customer, Customer]]:
        old_edges: list[tuple[Customer, Customer]] = []
        for route in self.routes:
            for edge in pairwise(i.customer for i in route.path):
                if edge not in old_edges:
                    yield edge
                else:
                    old_edges.append(edge)

    def pretty_str(self) -> str:
        to_return = f'{self.vehicle_count}\n'

        for i, route in enumerate(self.routes):
            to_return += f'{i + 1}: {route.pretty_path()}\n'

        to_return += f"{sum(i.distance for i in self.routes):.2f}"

        return to_return


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

    def __getitem__(self, item: tuple[Customer, Customer]) -> float:
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

    if not max_distance:
        max_distance = 1

    cost -= sum(i.distance / max_distance for i in solution.routes) / len(solution.routes)

    return abs(cost)


@cache
def reinforcement_function(cost: float) -> float:
    return 1 / (1 - cost)


@cache
def calculate_distance(
        c1: Customer,
        c2: Customer
) -> float:
    return math.sqrt(math.pow(c1.x - c2.x, 2) + math.pow(c1.y - c2.y, 2))


def get_viable_next(
        customers: Iterable[Customer],
        ant: Ant,
        max_capacity: int
) -> list[Customer]:
    to_return = []

    for customer in customers:
        if (
                (
                        customer.ready_time * bool(ant.time)
                        <= ant.time
                        + math.ceil(calculate_distance(ant.path[-1].customer, customer))
                        <= customer.due_date
                )
                and (
                ant.time
                + math.ceil(calculate_distance(ant.path[-1].customer, customer))
                + customer.service_time
                + math.ceil(calculate_distance(ant.path[0].customer, customer))
                <= ant.path[0].customer.due_date
        )
                and ant.capacity + customer.demand <= max_capacity
        ):
            to_return.append(customer)

    return to_return


@cache
def customer_weight(
        start_customer: Customer,
        end_customer: Customer,
        depot: Customer,
        max_distance: float
) -> float:
    return (
            1 - calculate_distance(start_customer, end_customer) / max_distance
            + 1 - end_customer.ready_time / depot.due_date
            + 1 - (end_customer.due_date - end_customer.ready_time) / depot.due_date
            + calculate_distance(depot, end_customer) / max_distance
    )


def ant_colony(
        instance: Instance,
        timeout_minutes: float = 1,
        alpha: float = 0.5,
        beta: float = 0.5,
        decay_rate: float = 0.3
):
    pheromones = Pheromones(
        decay_rate,
        partial(cost_function, instance=instance),
        reinforcement_function
    )

    customers = instance.customers.copy()
    depot = customers.pop(0)
    iteration_count = 0

    max_distance = max(calculate_distance(*i) for i in combinations(customers, 2))

    start_time = time()
    solutions: list[Solution] = []

    while (
            (
                    timeout_minutes
                    and time() - start_time <= timeout_minutes * 60
            )
            or (
                    solutions
                    and not timeout_minutes
                    and (
                            (cost_function(solutions[-2], instance) - cost_function(solutions[-1], instance))
                            / cost_function(solutions[-2], instance) >= 0.025
                    ) if len(solutions) >= 2 else True
            )
            or (
                    not solutions
                    and not timeout_minutes
            )
    ):
        ants: list[Ant] = []
        customers_to_use = customers.copy()
        for _ in range(instance.vehicle_count):
            ant: Ant = Ant(depot.due_date, [ExtendedCustomer(depot, 0, 0)])
            # while viable_next := get_viable_next(customers_to_use, ant):
            while True:
                viable_next = get_viable_next(customers_to_use, ant, instance.vehicle_capacity)
                if not viable_next:
                    next_customer = depot
                else:
                    next_customer = max(viable_next, key=lambda i: (
                            pheromones[(ant.path[-1].customer, i)] ** alpha
                            * customer_weight(ant.path[-1].customer, i, depot, max_distance) ** beta
                            / sum(
                        (
                            pheromones[(ant.path[-1].customer, j)] ** alpha
                            * customer_weight(ant.path[-1].customer, j, depot, max_distance) ** beta
                            for j in viable_next
                        ),
                        start=1
                    )
                    ))

                if ant.path[-1].customer == next_customer and next_customer == depot:
                    break

                ant.time += math.ceil(
                    calculate_distance(ant.path[-1].customer, next_customer)
                )

                if ant.time < next_customer.ready_time:
                    ant.path[0] = ExtendedCustomer(depot, next_customer.ready_time - ant.time, 0)
                    ant.time += (next_customer.ready_time - ant.time)

                ant.time += next_customer.service_time

                ant.capacity += next_customer.demand
                ant.distance += calculate_distance(ant.path[-1].customer, next_customer)
                if next_customer != depot:
                    customers_to_use.remove(next_customer)
                ant.path.append(ExtendedCustomer(next_customer, ant.time - next_customer.service_time, ant.capacity))
                if next_customer == depot:
                    break
            if not ant.time:
                break
            ants.append(ant)

        solution = Solution(ants, len(ants))
        solutions.append(solution)
        pheromones.evaporation().reinforce(solution)
        iteration_count += 1

    best_solution = max(solutions, key=partial(cost_function, instance=instance))

    customers_visited = sum(len(i.path) - 2 for i in best_solution.routes)
    customers_total = len(customers)

    assert customers_total == customers_visited, f"Not all customers visited. {customers_visited}/{customers_total}"

    return best_solution


def main() -> None:
    instance = parse_instance("instances/inst1.TXT")
    solution = ant_colony(instance, 1, alpha=2, beta=1, decay_rate=0.5)
    print(solution.pretty_str())


if __name__ == "__main__":
    main()
