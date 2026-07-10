from .base import DataIngestor, DataRecord


class NSEBhavcopyIngestor(DataIngestor):
    def fetch(self, ticker: str, from_date: str, to_date: str) -> list[DataRecord]:
        ...

    def validate(self, records: list[DataRecord]) -> bool:
        ...
