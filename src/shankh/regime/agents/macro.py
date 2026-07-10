from shankh.common.agents.base import BaseAgent, AgentContext, AgentResult, Explanation, ValidationReport


class MacroAnalystAgent(BaseAgent):
    def run(self, context: AgentContext) -> AgentResult:
        ...

    def explain(self) -> Explanation:
        return Explanation(
            assumptions=["Macro regimes persist over weeks"],
            inputs=["rates", "crude", "usd_inr", "vix", "fii_dii_flows"],
            reasoning=["Rule-based scoring across macro dimensions"],
        )

    def validate(self) -> ValidationReport:
        ...
