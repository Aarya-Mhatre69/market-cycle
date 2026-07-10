from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentContext:
    run_id: str
    inputs: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResult:
    run_id: str
    outputs: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Explanation:
    assumptions: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)


class BaseAgent(ABC):
    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        ...

    @abstractmethod
    def explain(self) -> Explanation:
        ...

    @abstractmethod
    def validate(self) -> ValidationReport:
        ...
