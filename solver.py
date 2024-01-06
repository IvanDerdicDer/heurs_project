import math

from greedy_solver import Solution, greedy_solver
from instance_parser import Customer, parse_instance
from itertools import pairwise


def calculate_distance(
        c1: Customer,
        c2: Customer
) -> float:
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


def cost_function(
        solution: Solution
) -> float:
    cost = len(solution.routes) * 1000

    cost -= sum((
        calculate_distance(*j)
        for i in solution.routes
        for j in pairwise(i.route)
    ))

    return cost
