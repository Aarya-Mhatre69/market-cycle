from typing import Any


class PersonaPool:
    def __init__(self) -> None:
        self._personas: list[dict[str, Any]] = []

    def add_persona(self, name: str, biases: dict[str, float], priors: dict[str, Any]) -> None:
        ...

    def get_active(self) -> list[dict[str, Any]]:
        ...
