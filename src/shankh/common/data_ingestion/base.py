from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DataRecord:
    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str


class DataIngestor(ABC):
    @abstractmethod
    def fetch(self, ticker: str, from_date: str, to_date: str) -> list[DataRecord]:
        ...

    @abstractmethod
    def validate(self, records: list[DataRecord]) -> bool:
        ...
