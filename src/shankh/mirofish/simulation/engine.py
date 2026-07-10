from typing import Any

from shankh.common.agents.base import BaseAgent, AgentContext, AgentResult, Explanation, ValidationReport


class Simulator(BaseAgent):
    def run(self, context: AgentContext) -> AgentResult:
        ...

    def explain(self) -> Explanation:
        return Explanation(
            assumptions=["Swarm dynamics approximate market narratives"],
            inputs=["event_seed", "scenarios", "personas", "rounds"],
            reasoning=["Iterative rounds with persona reactions and consensus tracking"],
        )

    def validate(self) -> ValidationReport:
        ...
