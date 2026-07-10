from typing import Any, Protocol


class DataSource(Protocol):
    def fetch(self, **kwargs: Any) -> Any: ...
