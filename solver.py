import math
from collections import deque
from copy import copy
from itertools import islice

from instance_parser import Instance, parse_instance, Customer
from dataclasses import dataclass


@dataclass
class Route:
    route: list[Customer]


@dataclass
class Solution:
    routes: list[Route]


def calculate_distance(
        c1: Customer,
        c2: Customer
) -> float:
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


class TooManyVehiclesException(Exception):
    pass


def greedy_heuristic(
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


def main() -> None:
    instance = parse_instance("instances/inst1.TXT")
    solution = greedy_heuristic(instance)

    s = 0
    for i in solution.routes:
        s += len(i.route)

    print(s, len(instance.customers))


if __name__ == "__main__":
    main()
