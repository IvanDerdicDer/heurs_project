from greedy_solver import Solution


def cost_function(
        solution: Solution
) -> float:
    cost = len(solution.routes) * 1000

    cost -= sum(i.distance for i in solution.routes)

    return cost
