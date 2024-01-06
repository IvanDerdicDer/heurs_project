import math
from collections import deque
from copy import copy
from dataclasses import dataclass
from itertools import pairwise
from typing import Optional

from instance_parser import Instance, Customer


@dataclass
class Route:
    route: list[Customer]
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
            x.ready_time,
            calculate_distance(x, depot),
            x.due_date,
        ),
        reverse=True
    )

    solution: list[Route] = []
    while customers:
        # next_customer: Customer = customers.pop()
        next_customer: Optional[Customer] = None
        route_time = 0
        route_capacity = 0
        route: list[Customer] = [depot]

        # route_capacity += next_customer.demand
        # route_time += next_customer.service_time + math.ceil(calculate_distance(next_customer, route[-1]))
        # route.append(next_customer)
        while next_customer is None or (
                customers
                and route_time + next_customer.service_time + math.ceil(
                    calculate_distance(next_customer, route[-1])
                ) + math.ceil(
                    calculate_distance(next_customer, depot)
                ) <= depot.due_date
                and route_capacity + next_customer.demand <= instance.vehicle_capacity
        ):
            possible_next = sorted([
                    i
                    for i in customers
                    if i.ready_time <= route_time + math.ceil(calculate_distance(next_customer, route[-1])) <= i.due_date
                ],
                key=lambda x: (
                    depot.due_date - x.ready_time,
                    calculate_distance(x, route[-1]),
                    x.due_date,
                ),
                reverse=True
            ) if next_customer else []

            if possible_next:
                next_customer = possible_next.pop()
                customers.remove(next_customer)
            else:
                customers = sorted(
                    customers,
                    key=lambda x: (
                        x.ready_time,
                        calculate_distance(x, route[-1]),
                        x.due_date,
                    ),
                    reverse=True
                )
                next_customer = customers.pop()

            route_capacity += next_customer.demand
            route_time += next_customer.service_time + math.ceil(calculate_distance(next_customer, route[-1]))
            route.append(next_customer)

        route = route + [depot]
        solution.append(Route(
            route,
            sum(calculate_distance(*i) for i in pairwise(route)),
            sum(i.service_time for i in route) + sum(calculate_distance(*i) for i in pairwise(route)),
            sum(i.demand for i in route),
        ))

    return Solution(
        solution,
        len(solution)
    )
