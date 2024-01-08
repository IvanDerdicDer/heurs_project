import math
from copy import copy
from dataclasses import dataclass
from itertools import pairwise
from typing import Optional

from instance_parser import Instance, Customer


@dataclass
class CustomerExtra:
    customer: Customer
    time: int
    capacity: int


@dataclass
class Route:
    route: list[CustomerExtra]
    distance: float
    time: int
    capacity: int


@dataclass
class Solution:
    routes: list[Route]
    vehicle_count: int

    def __contains__(self, item: Customer) -> bool:
        return any(item in i.route for i in self.routes)


class TooManyVehiclesException(Exception):
    pass


def calculate_distance(
        c1: Customer,
        c2: Customer
) -> float:
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def greedy_solver(
        instance: Instance
) -> Solution:
    customers = copy(instance.customers)

    depot = customers.pop(0)

    customers = sorted(
        customers,
        key=lambda x: (
            depot.due_date - x.ready_time,
            x.due_date,
            calculate_distance(x, depot),
        ),
        reverse=False
    )

    solution: list[Route] = []
    while customers:
        next_customer: Optional[Customer] = None
        route_time = 0
        route_capacity = 0
        route: list[CustomerExtra] = [CustomerExtra(depot, 0, 0)]

        while next_customer is None or (
                customers
                and route_time + next_customer.service_time + math.ceil(
                    calculate_distance(next_customer, route[-1].customer)
                ) + math.ceil(
                    calculate_distance(next_customer, depot)
                ) <= depot.due_date
                and route_capacity + next_customer.demand <= instance.vehicle_capacity
        ):
            possible_next = sorted([
                i
                for i in customers
                if (
                        i.ready_time
                        <= route_time
                        + math.ceil(calculate_distance(next_customer, route[-1].customer))
                        + next_customer.service_time
                        + math.ceil(calculate_distance(i, route[-1].customer))
                        <= i.due_date
                )
            ],
                key=lambda x: (
                    # depot.due_date - x.ready_time,
                    x.ready_time,
                    x.due_date,
                    calculate_distance(x, route[-1].customer),
                ),
                reverse=False
            ) if next_customer else []

            if possible_next:
                next_customer = possible_next.pop()
                customers.remove(next_customer)
            else:
                customers = sorted(
                    customers,
                    key=lambda x: (
                        depot.due_date - x.ready_time,
                        x.due_date,
                        calculate_distance(x, route[-1].customer),
                    ),
                    reverse=False
                )
                next_customer = customers.pop()

            if (
                    route_time + next_customer.service_time + math.ceil(
                    calculate_distance(next_customer, route[-1].customer)
                    ) + math.ceil(
                        calculate_distance(next_customer, depot)
                    ) <= depot.due_date
                    and route_capacity + next_customer.demand <= instance.vehicle_capacity
            ):
                route_capacity += next_customer.demand
                route_time += next_customer.service_time + math.ceil(calculate_distance(next_customer, route[-1].customer))
                route.append(CustomerExtra(next_customer, route_time - next_customer.service_time, route_capacity))
            else:
                customers.append(next_customer)

        route_capacity += depot.demand
        route_time += depot.service_time + math.ceil(calculate_distance(depot, route[-1].customer))
        route = route + [CustomerExtra(depot, route_time, route_capacity)]
        solution.append(Route(
            route,
            sum(calculate_distance(*i) for i in pairwise(j.customer for j in route)),
            route_time,
            route_capacity,
        ))

    return Solution(
        solution,
        len(solution)
    )
