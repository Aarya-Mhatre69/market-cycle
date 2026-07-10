from typing import Any


class RegimeFuser:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {"macro": 0.4, "breadth": 0.35, "cycle": 0.25}

    def fuse(
        self,
        macro_score: float,
        breadth_score: float,
        cycle_label: str,
    ) -> dict[str, Any]:
        ...

    def calibrate_weights(self, historical_data: list[dict]) -> None:
        ...
