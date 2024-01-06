from dataclasses import dataclass


@dataclass
class Customer:
    number: int
    x: int
    y: int
    demand: int
    ready_time: int
    due_date: int
    service_time: int


@dataclass
class Instance:
    vehicle_number: int
    vehicle_capacity: int
    customers: list[Customer]