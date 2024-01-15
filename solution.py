from instance_parser import parse_instance
from ant_colony import ant_colony, Solution

def save_to_file(instance_number, timeout_minutes, solution):
    if timeout_minutes == 0:
        filename = f"un-i{instance_number}.txt"
    else:
        filename = f"{timeout_minutes}m-i{instance_number}.txt"

    with open(filename, 'w') as file:
        file.write(solution.pretty_str())

def main() -> None:
    instance_range = range(1, 7)
    timeout_variations = [1, 5, 0]

    for instance_number in instance_range:
        for timeout_minutes in timeout_variations:
            instance = parse_instance(f"instances/inst{instance_number}.TXT")
            solution = ant_colony(instance, timeout_minutes, alpha=2, beta=1, decay_rate=0.5)
            print(solution.pretty_str())
            save_to_file(instance_number, timeout_minutes, solution)

if __name__ == "__main__":
    main()