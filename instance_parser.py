from dataclasses import dataclass
import re


@dataclass(frozen=True, eq=True)
class Customer:
    number: int
    x: int
    y: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int


@dataclass(frozen=True, eq=True)
class Instance:
    vehicle_count: int
    vehicle_capacity: int
    customers: list[Customer]


def parse_instance(path: str) -> Instance:
    with open(path, "r") as f:
        lines = [i.strip() for i in f.readlines()]

    vehicle_count, vehicle_capacity = (int(i) for i in re.split(r' +', lines[2]))

    customers = [
        Customer(*(int(j) for j in re.split(r' +', i)))
        for i in lines[7:]
    ]

    return Instance(
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
        customers=customers
    )
