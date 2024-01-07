import math

from greedy_solver import Solution, greedy_solver
from instance_parser import parse_instance, Instance


def cost_function(
        solution: Solution,
        instance: Instance
) -> float:
    theoretical_min = math.ceil(sum(i.demand for i in instance.customers) / instance.vehicle_capacity)

    cost = theoretical_min / len(solution.routes)

    max_distance = max(i.distance for i in solution.routes)

    cost -= sum(i.distance / max_distance for i in solution.routes) / len(solution.routes)

    return cost


instance = parse_instance("instances/inst1.TXT")
depot = instance.customers[0]
solution = greedy_solver(instance)

print(cost_function(solution, instance))

assert len(instance.customers) - 1 == sum(len([j for j in i.route if j.customer.number]) for i in solution.routes), f"Not all customers visited"
assert all(i.capacity <= instance.vehicle_capacity for i in solution.routes), f"Some routes are over capacity"
assert all(i.time <= depot.due_date for i in solution.routes), f"Not all routes finish during working hours"
