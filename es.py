from dataclasses import field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass
Fact = Tuple[str, str]  # (predicate, value)


@dataclass(frozen=True)
class Atom:
    predicate: str
    value: str

    def __str__(self) -> str:
        return f"{self.predicate}={self.value}"


@dataclass
class Rule:
    rule_id: str
    conditions: List[Atom]
    conclusion: Atom
    priority: int = 0
    description: str = ""

    def __str__(self) -> str:
        conds = " AND ".join(str(c) for c in self.conditions) if self.conditions else "TRUE"
        return f"{self.rule_id}: IF {conds} THEN {self.conclusion}"


@dataclass
class Ontology:
    """
    Ontology is a value schema for predicates.
    Example:
        Market -> Bullish, Bearish, Neutral
        Recommendation -> BUY, HOLD, SELL
    """
    allowed_values: Dict[str, Set[str]]

    def validate_atom(self, atom: Atom) -> Optional[str]:
        if atom.predicate not in self.allowed_values:
            return f"Unknown predicate '{atom.predicate}'"
        if atom.value not in self.allowed_values[atom.predicate]:
            return (
                f"Invalid value '{atom.value}' for predicate '{atom.predicate}'. "
                f"Allowed: {sorted(self.allowed_values[atom.predicate])}"
            )
        return None

    def validate_fact(self, fact: Fact) -> Optional[str]:
        return self.validate_atom(Atom(*fact))

    def pretty_tree(self) -> str:
        lines = ["Ontology"]
        preds = sorted(self.allowed_values)
        for i, pred in enumerate(preds):
            pbranch = "└──" if i == len(preds) - 1 else "├──"
            lines.append(f"{pbranch} {pred}")
            vals = sorted(self.allowed_values[pred])
            for j, val in enumerate(vals):
                vbranch = "└──" if j == len(vals) - 1 else "├──"
                lines.append(f"    {vbranch} {val}")
        return "\n".join(lines)


@dataclass
class KnowledgeBase:
    ontology: Ontology
    facts: Set[Fact] = field(default_factory=set)
    rules: List[Rule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBase":
        ontology = Ontology({k: set(v) for k, v in data["ontology"].items()})
        facts = {(f["predicate"], f["value"]) for f in data.get("facts", [])}
        rules: List[Rule] = []
        for r in data.get("rules", []):
            rules.append(
                Rule(
                    rule_id=r["id"],
                    conditions=[Atom(c["predicate"], c["value"]) for c in r["if"]],
                    conclusion=Atom(r["then"]["predicate"], r["then"]["value"]),
                    priority=int(r.get("priority", 0)),
                    description=r.get("description", ""),
                )
            )
        return cls(ontology=ontology, facts=facts, rules=rules)


@dataclass
class FireEvent:
    rule_id: str
    conclusion: Fact
    supporting_facts: List[Fact]
    step: int


@dataclass
class ProofNode:
    goal: Fact
    rule_id: Optional[str] = None
    children: List["ProofNode"] = field(default_factory=list)
    is_fact: bool = False
    success: bool = False
    reason: str = ""

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        status = "✓" if self.success else "✗"
        if self.is_fact:
            head = f"{pad}{status} FACT {self.goal[0]}={self.goal[1]}"
            if self.reason:
                head += f"  [{self.reason}]"
            return head

        head = f"{pad}{status} GOAL {self.goal[0]}={self.goal[1]}"
        if self.rule_id:
            head += f"  <- {self.rule_id}"
        if self.reason:
            head += f"  [{self.reason}]"

        for child in self.children:
            head += "\n" + child.render(indent + 1)
        return head


class RuleValidationReport:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = ["Validation Report"]
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        else:
            lines.append("Errors: none")

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("Warnings: none")

        lines.append(f"Status: {'PASS' if self.ok() else 'FAIL'}")
        return "\n".join(lines)


class RuleValidator:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def validate(self) -> RuleValidationReport:
        report = RuleValidationReport()

        # Duplicate rule IDs
        seen_ids = set()
        for r in self.kb.rules:
            if r.rule_id in seen_ids:
                report.errors.append(f"Duplicate rule id: {r.rule_id}")
            seen_ids.add(r.rule_id)

        # Facts must be valid ontology values
        for fact in self.kb.facts:
            err = self.kb.ontology.validate_fact(fact)
            if err:
                report.errors.append(f"Invalid fact {fact}: {err}")

        # Conditions and conclusions must be valid ontology values
        for r in self.kb.rules:
            for atom in r.conditions + [r.conclusion]:
                err = self.kb.ontology.validate_atom(atom)
                if err:
                    report.errors.append(f"Rule {r.rule_id}: {err}")

        # Duplicate conditions in same rule
        for r in self.kb.rules:
            seen_cond = set()
            for c in r.conditions:
                key = (c.predicate, c.value)
                if key in seen_cond:
                    report.warnings.append(f"Rule {r.rule_id} repeats condition {c}.")
                seen_cond.add(key)

        # Dependency graph for cycle detection
        dep_graph = defaultdict(set)
        for ra in self.kb.rules:
            for cond in ra.conditions:
                for rb in self.kb.rules:
                    if (rb.conclusion.predicate, rb.conclusion.value) == (cond.predicate, cond.value):
                        dep_graph[rb.rule_id].add(ra.rule_id)

        cycle = self._find_cycle(dep_graph)
        if cycle:
            report.warnings.append("Rule dependency cycle detected: " + " -> ".join(cycle))

        # Reachability from current facts
        reachable_rules = self._reachable_rules()
        for r in self.kb.rules:
            if r.rule_id not in reachable_rules:
                report.warnings.append(
                    f"Rule {r.rule_id} is unreachable from current facts/derived facts."
                )

        # Multiple rules yielding same conclusion
        conclusion_sources = defaultdict(list)
        for r in self.kb.rules:
            conclusion_sources[(r.conclusion.predicate, r.conclusion.value)].append(r.rule_id)
        for concl, rs in conclusion_sources.items():
            if len(rs) > 1:
                report.warnings.append(
                    f"Conclusion {concl[0]}={concl[1]} is produced by multiple rules: {rs}"
                )

        return report

    def _reachable_rules(self) -> Set[str]:
        current = set(self.kb.facts)
        reachable_rules: Set[str] = set()

        changed = True
        while changed:
            changed = False
            for r in self.kb.rules:
                if r.rule_id in reachable_rules:
                    continue
                if all((c.predicate, c.value) in current for c in r.conditions):
                    reachable_rules.add(r.rule_id)
                    new_fact = (r.conclusion.predicate, r.conclusion.value)
                    if new_fact not in current:
                        current.add(new_fact)
                        changed = True
        return reachable_rules

    def _find_cycle(self, graph: Dict[str, Set[str]]) -> List[str]:
        visited: Set[str] = set()
        stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            stack.add(node)
            path.append(node)

            for nxt in graph.get(node, ()):
                if nxt not in visited:
                    res = dfs(nxt)
                    if res:
                        return res
                elif nxt in stack:
                    idx = path.index(nxt)
                    return path[idx:] + [nxt]

            stack.remove(node)
            path.pop()
            return None

        for node in graph:
            if node not in visited:
                res = dfs(node)
                if res:
                    return res
        return []


class ForwardChainer:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.facts: Set[Fact] = set(kb.facts)
        self.trace: List[FireEvent] = []

    def run(self, max_steps: int = 1000) -> Tuple[Set[Fact], List[FireEvent]]:
        # Higher priority first, then more specific rules, then deterministic by rule id
        rules = sorted(self.kb.rules, key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))
        fired_rules: Set[str] = set()
        step = 0

        while step < max_steps:
            fired_this_round = False

            for rule in rules:
                if rule.rule_id in fired_rules:
                    continue

                if all((c.predicate, c.value) in self.facts for c in rule.conditions):
                    concl = (rule.conclusion.predicate, rule.conclusion.value)
                    self.facts.add(concl)
                    self.trace.append(
                        FireEvent(
                            rule_id=rule.rule_id,
                            conclusion=concl,
                            supporting_facts=[(c.predicate, c.value) for c in rule.conditions],
                            step=step,
                        )
                    )
                    fired_rules.add(rule.rule_id)
                    fired_this_round = True
                    step += 1
                    break

            if not fired_this_round:
                break

        return self.facts, self.trace


class BackwardChainer:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.memo: Dict[Tuple[Fact, Tuple[Fact, ...]], ProofNode] = {}

    def prove(
        self,
        goal: Fact,
        facts: Optional[Set[Fact]] = None,
        _stack: Optional[List[Fact]] = None,
    ) -> ProofNode:
        if facts is None:
            facts = set(self.kb.facts)
        if _stack is None:
            _stack = []

        memo_key = (goal, tuple(sorted(facts)))
        if memo_key in self.memo:
            return self.memo[memo_key]

        node = ProofNode(goal=goal)

        if goal in facts:
            node.is_fact = True
            node.success = True
            node.reason = "present in working memory"
            self.memo[memo_key] = node
            return node

        if goal in _stack:
            node.success = False
            node.reason = "cycle detected"
            self.memo[memo_key] = node
            return node

        _stack.append(goal)

        candidate_rules = [
            r for r in self.kb.rules
            if (r.conclusion.predicate, r.conclusion.value) == goal
        ]
        candidate_rules.sort(key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))

        for rule in candidate_rules:
            children = []
            ok = True
            for cond in rule.conditions:
                child = self.prove((cond.predicate, cond.value), facts=facts, _stack=_stack)
                children.append(child)
                if not child.success:
                    ok = False

            if ok:
                node.rule_id = rule.rule_id
                node.children = children
                node.success = True
                node.reason = "all antecedents proven"
                _stack.pop()
                self.memo[memo_key] = node
                return node

        _stack.pop()
        node.success = False
        node.reason = "no supporting rule chain found"
        self.memo[memo_key] = node
        return node


class ExplanationGenerator:
    @staticmethod
    def from_forward_trace(trace: List[FireEvent], goal: Optional[Fact] = None) -> str:
        lines = ["Forward Chaining Explanation"]
        if not trace:
            lines.append("No rule fired.")
            return "\n".join(lines)

        for ev in trace:
            supp = ", ".join(f"{p}={v}" for p, v in ev.supporting_facts) if ev.supporting_facts else "TRUE"
            lines.append(
                f"Step {ev.step + 1}: {ev.rule_id} fired because [{supp}] "
                f"=> {ev.conclusion[0]}={ev.conclusion[1]}"
            )

        if goal:
            lines.append(f"Final queried goal: {goal[0]}={goal[1]}")
        return "\n".join(lines)

    @staticmethod
    def from_proof_tree(tree: ProofNode) -> str:
        return "Backward Chaining Explanation\n" + tree.render()


def demo_kb() -> KnowledgeBase:
    data = {
        "ontology": {
            "Market": {"Bullish", "Bearish", "Neutral"},
            "InterestRate": {"Falling", "Rising", "Stable"},
            "Inflation": {"Low", "High", "Stable"},
            "PE": {"Low", "Fair", "High"},
            "RevenueGrowth": {"Low", "Medium", "High"},
            "Debt": {"Low", "Medium", "High"},
            "Economy": {"Strong", "Weak"},
            "EquitySentiment": {"Positive", "Negative"},
            "BuyCandidate": {"True", "False"},
            "Recommendation": {"BUY", "HOLD", "SELL"},
        },
        "facts": [
            {"predicate": "Market", "value": "Bullish"},
            {"predicate": "InterestRate", "value": "Falling"},
            {"predicate": "Inflation", "value": "Low"},
            {"predicate": "PE", "value": "Low"},
            {"predicate": "RevenueGrowth", "value": "High"},
            {"predicate": "Debt", "value": "Low"},
        ],
        "rules": [
            {
                "id": "R1",
                "if": [{"predicate": "Market", "value": "Bullish"}],
                "then": {"predicate": "Economy", "value": "Strong"},
                "priority": 3,
            },
            {
                "id": "R2",
                "if": [
                    {"predicate": "Economy", "value": "Strong"},
                    {"predicate": "InterestRate", "value": "Falling"},
                ],
                "then": {"predicate": "EquitySentiment", "value": "Positive"},
                "priority": 3,
            },
            {
                "id": "R3",
                "if": [
                    {"predicate": "EquitySentiment", "value": "Positive"},
                    {"predicate": "PE", "value": "Low"},
                    {"predicate": "RevenueGrowth", "value": "High"},
                ],
                "then": {"predicate": "BuyCandidate", "value": "True"},
                "priority": 2,
            },
            {
                "id": "R4",
                "if": [
                    {"predicate": "BuyCandidate", "value": "True"},
                    {"predicate": "Debt", "value": "Low"},
                ],
                "then": {"predicate": "Recommendation", "value": "BUY"},
                "priority": 1,
            },
            {
                "id": "R5",
                "if": [{"predicate": "Inflation", "value": "High"}],
                "then": {"predicate": "Economy", "value": "Weak"},
                "priority": 1,
            },
            {
                "id": "R6",
                "if": [{"predicate": "Economy", "value": "Weak"}],
                "then": {"predicate": "Recommendation", "value": "SELL"},
                "priority": 0,
            },
        ],
    }
    return KnowledgeBase.from_dict(data)


def main() -> None:
    kb = demo_kb()

    print("=" * 80)
    print("ONTOLOGY")
    print("=" * 80)
    print(kb.ontology.pretty_tree())

    print("\n" + "=" * 80)
    print("RULE BASE")
    print("=" * 80)
    for r in kb.rules:
        print(r)

    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    report = RuleValidator(kb).validate()
    print(report.render())

    print("\n" + "=" * 80)
    print("FORWARD CHAINING")
    print("=" * 80)
    fwd = ForwardChainer(kb)
    final_facts, trace = fwd.run()
    for ev in trace:
        print(f"{ev.step + 1}. {ev.rule_id} -> {ev.conclusion[0]}={ev.conclusion[1]}")

    print("\nFinal facts:")
    for fact in sorted(final_facts):
        print(" -", f"{fact[0]}={fact[1]}")

    goal = ("Recommendation", "BUY")

    print("\n" + "=" * 80)
    print("BACKWARD CHAINING")
    print("=" * 80)
    bc = BackwardChainer(kb)
    proof = bc.prove(goal, facts=set(kb.facts))
    print(proof.render())

    print("\n" + "=" * 80)
    print("EXPLANATION GENERATION")
    print("=" * 80)
    print(ExplanationGenerator.from_forward_trace(trace, goal=goal))
    print()
    print(ExplanationGenerator.from_proof_tree(proof))


if __name__ == "__main__":
    main()
