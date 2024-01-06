import math
from collections import deque
from copy import copy
from dataclasses import dataclass

from instance_parser import Instance, Customer


@dataclass
class Route:
    route: list[Customer]


@dataclass
class Solution:
    routes: list[Route]


class TooManyVehiclesException(Exception):
    pass


def greedy_solver(
        instance: Instance
) -> Solution:
    customers = copy(instance.customers)

    depot = customers.pop(0)

    customers = deque(
        sorted(
            customers,
            key=lambda x: (
                x.ready_time,
                x.due_date,
            )
        )
    )

    solution: list[Route] = []
    while customers:
        next_customer: Customer = customers.popleft()
        route_time = 0
        route_capacity = 0
        route: list[Customer] = [next_customer]

        route_capacity += next_customer.demand
        route_time += next_customer.service_time
        while (
                customers
                and route_time + next_customer.service_time <= depot.due_date
                and route_capacity + next_customer.demand <= instance.vehicle_capacity
        ):
            possible_next = [i for i in customers if i.ready_time >= route_time]

            if possible_next:
                next_customer = possible_next[0]
                customers.remove(next_customer)
            else:
                next_customer = customers.popleft()

            route.append(next_customer)
            route_capacity += next_customer.demand
            route_time += next_customer.service_time

        solution.append(Route(route))

    return Solution(solution)
