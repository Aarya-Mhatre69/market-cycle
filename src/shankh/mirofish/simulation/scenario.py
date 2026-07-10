from typing import Any


class ScenarioGenerator:
    def expand(self, seed: str) -> list[dict[str, Any]]:
        ...
