class PostgresStore:
    def __init__(self, connection_string: str) -> None:
        self._conn_str = connection_string

    def insert_market_data(self, records: list[dict]) -> None:
        ...

    def query_features(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        ...
