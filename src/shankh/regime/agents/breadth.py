from shankh.common.agents.base import BaseAgent, AgentContext, AgentResult, Explanation, ValidationReport


class MarketBreadthAgent(BaseAgent):
    def run(self, context: AgentContext) -> AgentResult:
        ...

    def explain(self) -> Explanation:
        return Explanation(
            assumptions=["Sector participation indicates conviction"],
            inputs=["sector_indices", "advance_decline", "new_highs", "new_lows"],
            reasoning=["Breadth score composite of advance-decline and participation"],
        )

    def validate(self) -> ValidationReport:
        ...
