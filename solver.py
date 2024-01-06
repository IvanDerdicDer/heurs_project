from greedy_solver import Solution, greedy_solver
from instance_parser import parse_instance


def cost_function(
        solution: Solution
) -> float:
    cost = len(solution.routes) * 1000

    cost -= sum(i.distance for i in solution.routes)

    return cost


instance = parse_instance("instances/inst1.TXT")
depot = instance.customers[0]
solution = greedy_solver(instance)
assert len(instance.customers) - 1 == sum(len([j for j in i.route if j.number]) for i in solution.routes), f"Not all customers visited"
assert all(i.time <= depot.due_date for i in solution.routes), f"Not all routes finish during working hours"
