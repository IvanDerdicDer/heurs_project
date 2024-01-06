from greedy_solver import Solution, greedy_solver
from instance_parser import parse_instance


def cost_function(
        solution: Solution
) -> float:
    cost = len(solution.routes) * 1000

    cost -= sum(i.distance for i in solution.routes)

    return cost


instance = parse_instance("instances/inst1.TXT")
solution = greedy_solver(instance)
pass
