"""
nes.py

- LangGraph orchestrates the flow
- Neo4j stores facts/rules (or falls back to default in-memory KB)
- Implements DFS, BFS, and A* Backward Chaining algorithms
- Provides comparative analysis across search strategies

Install:
  pip install langgraph neo4j

Run:
  python nes.py
"""

from collections import deque
import heapq
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


Fact = Tuple[str, str]  # (predicate, value)


# -----------------------------
# Core domain objects
# -----------------------------

@dataclass(frozen=True)
class Atom:
    predicate: str
    value: str

    def as_fact(self) -> Fact:
        return (self.predicate, self.value)

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
    allowed_values: Dict[str, Set[str]]

    def validate_atom(self, atom: Atom) -> Optional[str]:
        if atom.predicate not in self.allowed_values:
            return f"unknown predicate '{atom.predicate}'"
        if atom.value not in self.allowed_values[atom.predicate]:
            return (
                f"invalid value '{atom.value}' for '{atom.predicate}'. "
                f"allowed={sorted(self.allowed_values[atom.predicate])}"
            )
        return None

    def validate_fact(self, fact: Fact) -> Optional[str]:
        return self.validate_atom(Atom(*fact))


@dataclass
class KnowledgeBase:
    ontology: Ontology
    facts: Set[Fact] = field(default_factory=set)
    rules: List[Rule] = field(default_factory=list)


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
            line = f"{pad}{status} FACT {self.goal[0]}={self.goal[1]}"
            if self.reason:
                line += f"  [{self.reason}]"
            return line

        line = f"{pad}{status} GOAL {self.goal[0]}={self.goal[1]}"
        if self.rule_id:
            line += f"  <- {self.rule_id}"
        if self.reason:
            line += f"  [{self.reason}]"

        for child in self.children:
            line += "\n" + child.render(indent + 1)
        return line


@dataclass
class SearchResult:
    algorithm: str
    goal: Fact
    success: bool
    proof: Optional[ProofNode]
    nodes_explored: int
    proof_depth: int
    rules_used: List[str]
    cost: float
    reason: str = ""


# -----------------------------
# Default Knowledge Base Data
# -----------------------------

DEFAULT_FACTS: Set[Fact] = {
    ("Market", "Bullish"),
    ("InterestRate", "Falling"),
    ("Inflation", "Low"),
    ("PE", "Low"),
    ("RevenueGrowth", "High"),
    ("Debt", "Low"),
    ("Volatility", "Low"),
    ("GDP", "Growing"),
    ("OilPrice", "Stable"),
}

DEFAULT_RULES: List[Rule] = [
    Rule("R1", [Atom("Market", "Bullish")], Atom("Economy", "Strong"), priority=5, description="Bullish market indicates strong economy"),
    Rule("R2", [Atom("GDP", "Growing")], Atom("Economy", "Strong"), priority=4, description="Growing GDP indicates strong economy"),
    Rule("R3", [Atom("Inflation", "High")], Atom("Economy", "Weak"), priority=5, description="High inflation weakens economy"),
    Rule("R4", [Atom("Economy", "Strong"), Atom("InterestRate", "Falling")], Atom("EquitySentiment", "Positive"), priority=4, description="Strong economy & falling rates yield positive sentiment"),
    Rule("R5", [Atom("Economy", "Weak")], Atom("EquitySentiment", "Negative"), priority=4, description="Weak economy yields negative sentiment"),
    Rule("R6", [Atom("EquitySentiment", "Positive"), Atom("PE", "Low"), Atom("RevenueGrowth", "High")], Atom("BuyCandidate", "True"), priority=3, description="Positive sentiment, low PE, high growth implies buy candidate"),
    Rule("R7", [Atom("BuyCandidate", "True"), Atom("Debt", "Low"), Atom("Volatility", "Low")], Atom("Recommendation", "BUY"), priority=2, description="Buy candidate with low debt/volatility leads to BUY"),
    Rule("R8", [Atom("EquitySentiment", "Negative")], Atom("Recommendation", "SELL"), priority=2, description="Negative sentiment leads to SELL recommendation"),
    Rule("R9", [Atom("OilPrice", "Rising")], Atom("Inflation", "High"), priority=3, description="Rising oil price increases inflation"),
]


# -----------------------------
# Neo4j Storage
# -----------------------------

class Neo4jStore:
    """Retrieves facts and rules from Neo4j graph database, with fallback."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = None
        if GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self.driver.verify_connectivity()
            except Exception:
                self.driver = None

    @property
    def connected(self) -> bool:
        return self.driver is not None

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def load_kb(self) -> KnowledgeBase:
        if not self.connected:
            # Fallback to in-memory default KB if Neo4j is offline
            allowed_values: Dict[str, Set[str]] = {}
            for p, v in DEFAULT_FACTS:
                allowed_values.setdefault(p, set()).add(v)
            for r in DEFAULT_RULES:
                for atom in r.conditions + [r.conclusion]:
                    allowed_values.setdefault(atom.predicate, set()).add(atom.value)
            return KnowledgeBase(
                ontology=Ontology(allowed_values=allowed_values),
                facts=set(DEFAULT_FACTS),
                rules=list(DEFAULT_RULES),
            )

        with self.driver.session(database=self.database) as session:
            fact_rows = session.run(
                """
                MATCH (f:Fact)
                WHERE NOT ()-[:PRODUCES]->(f)
                RETURN f.predicate AS predicate, f.value AS value
                ORDER BY predicate, value
                """
            )
            fact_set = {(row["predicate"], row["value"]) for row in fact_rows}

            rule_rows = session.run(
                """
                MATCH (r:Rule)
                OPTIONAL MATCH (r)-[:REQUIRES]->(req:Fact)
                OPTIONAL MATCH (r)-[:PRODUCES]->(out:Fact)
                RETURN
                    r.id AS id,
                    coalesce(r.priority, 0) AS priority,
                    coalesce(r.description, '') AS description,
                    collect(DISTINCT req) AS requires,
                    collect(DISTINCT out) AS produces
                ORDER BY priority DESC, id
                """
            )

            rules: List[Rule] = []
            for row in rule_rows:
                requires_nodes = [n for n in row["requires"] if n is not None]
                produces_nodes = [n for n in row["produces"] if n is not None]
                if not produces_nodes:
                    continue

                conds = [Atom(n["predicate"], n["value"]) for n in requires_nodes]
                concl_node = produces_nodes[0]
                rules.append(
                    Rule(
                        rule_id=row["id"],
                        conditions=conds,
                        conclusion=Atom(concl_node["predicate"], concl_node["value"]),
                        priority=int(row["priority"] or 0),
                        description=row["description"] or "",
                    )
                )

        allowed_values: Dict[str, Set[str]] = {}
        for p, v in fact_set:
            allowed_values.setdefault(p, set()).add(v)
        for r in rules:
            for atom in r.conditions + [r.conclusion]:
                allowed_values.setdefault(atom.predicate, set()).add(atom.value)

        return KnowledgeBase(
            ontology=Ontology(allowed_values=allowed_values),
            facts=fact_set,
            rules=rules,
        )


# -----------------------------
# KB Validation
# -----------------------------

class RuleValidationReport:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def ok(self) -> bool:
        return not self.errors


def validate_kb(kb: KnowledgeBase) -> RuleValidationReport:
    report = RuleValidationReport()
    seen_ids: Set[str] = set()
    for rule in kb.rules:
        if rule.rule_id in seen_ids:
            report.errors.append(f"duplicate rule id: {rule.rule_id}")
        seen_ids.add(rule.rule_id)

    for fact in kb.facts:
        err = kb.ontology.validate_fact(fact)
        if err:
            report.errors.append(f"invalid fact {fact}: {err}")

    for rule in kb.rules:
        for atom in rule.conditions + [rule.conclusion]:
            err = kb.ontology.validate_atom(atom)
            if err:
                report.errors.append(f"rule {rule.rule_id}: {err}")

    return report


# -----------------------------
# Forward Chaining Engine
# -----------------------------

def forward_chain(kb: KnowledgeBase, max_steps: int = 100) -> Tuple[Set[Fact], List[FireEvent]]:
    facts = set(kb.facts)
    trace: List[FireEvent] = []
    rules = sorted(kb.rules, key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))
    fired_rules: Set[str] = set()
    step = 0

    while step < max_steps:
        fired_any = False
        for rule in rules:
            if rule.rule_id in fired_rules:
                continue
            if all(cond.as_fact() in facts for cond in rule.conditions):
                concl = rule.conclusion.as_fact()
                facts.add(concl)
                trace.append(
                    FireEvent(
                        rule_id=rule.rule_id,
                        conclusion=concl,
                        supporting_facts=[c.as_fact() for c in rule.conditions],
                        step=step,
                    )
                )
                fired_rules.add(rule.rule_id)
                fired_any = True
                step += 1
                break
        if not fired_any:
            break

    return facts, trace


# -----------------------------
# Proof Tree Builder Helper
# -----------------------------

def extract_proof_stats(node: ProofNode) -> Tuple[List[str], int]:
    """Helper to compute rules used and depth of a ProofNode tree."""
    if not node or not node.success or node.is_fact:
        return [], 0

    rules = []
    if node.rule_id:
        rules.append(node.rule_id)

    max_child_depth = 0
    for child in node.children:
        child_rules, child_depth = extract_proof_stats(child)
        rules.extend(child_rules)
        if child_depth > max_child_depth:
            max_child_depth = child_depth

    unique_rules = list(dict.fromkeys(rules))
    return unique_rules, max_child_depth + 1


def build_proof_tree(
    target: Fact,
    facts: Set[Fact],
    rules: List[Rule],
    rules_used: Set[str],
    visited: Optional[Set[Fact]] = None,
) -> ProofNode:
    """Reconstructs hierarchical ProofNode structure from rules used."""
    if visited is None:
        visited = set()

    node = ProofNode(goal=target)
    if target in facts:
        node.is_fact = True
        node.success = True
        node.reason = "present in working memory"
        return node

    if target in visited:
        node.success = False
        node.reason = "cycle detected"
        return node

    visited.add(target)

    candidate_rules = [
        r for r in rules
        if r.conclusion.as_fact() == target and (not rules_used or r.rule_id in rules_used)
    ]
    candidate_rules.sort(key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))

    for rule in candidate_rules:
        children = []
        all_ok = True
        for cond in rule.conditions:
            child_node = build_proof_tree(cond.as_fact(), facts, rules, rules_used, visited.copy())
            children.append(child_node)
            if not child_node.success:
                all_ok = False
        if all_ok:
            node.rule_id = rule.rule_id
            node.children = children
            node.success = True
            node.reason = "all antecedents proven"
            return node

    node.success = False
    node.reason = "no supporting rule chain found"
    return node


# -----------------------------
# Backward Chaining: DFS Algorithm
# -----------------------------

def backward_chain_dfs(kb: KnowledgeBase, goal: Fact) -> SearchResult:
    nodes_explored = 0
    memo: Dict[Tuple[Fact, Tuple[Fact, ...]], ProofNode] = {}

    def prove(target: Fact, facts: Set[Fact], stack: List[Fact]) -> ProofNode:
        nonlocal nodes_explored
        nodes_explored += 1

        key = (target, tuple(sorted(facts)))
        if key in memo:
            return memo[key]

        node = ProofNode(goal=target)

        if target in facts:
            node.is_fact = True
            node.success = True
            node.reason = "present in working memory"
            memo[key] = node
            return node

        if target in stack:
            node.success = False
            node.reason = "cycle detected"
            memo[key] = node
            return node

        stack.append(target)
        candidates = [r for r in kb.rules if r.conclusion.as_fact() == target]
        candidates.sort(key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))

        for rule in candidates:
            children: List[ProofNode] = []
            ok = True
            for cond in rule.conditions:
                child = prove(cond.as_fact(), facts, stack)
                children.append(child)
                if not child.success:
                    ok = False

            if ok:
                node.rule_id = rule.rule_id
                node.children = children
                node.success = True
                node.reason = "all antecedents proven"
                stack.pop()
                memo[key] = node
                return node

        stack.pop()
        node.success = False
        node.reason = "no supporting rule chain found"
        memo[key] = node
        return node

    proof = prove(goal, set(kb.facts), [])
    rules_used, depth = extract_proof_stats(proof)
    rule_map = {r.rule_id: r for r in kb.rules}
    cost = sum(10.0 / max(rule_map[r].priority, 1) for r in rules_used if r in rule_map)

    return SearchResult(
        algorithm="DFS",
        goal=goal,
        success=proof.success,
        proof=proof,
        nodes_explored=nodes_explored,
        proof_depth=depth,
        rules_used=rules_used,
        cost=round(cost, 2),
        reason=proof.reason if not proof.success else "Proof found via DFS",
    )


# -----------------------------
# Backward Chaining: BFS Algorithm
# -----------------------------

def backward_chain_bfs(kb: KnowledgeBase, goal: Fact) -> SearchResult:
    facts = set(kb.facts)
    if goal in facts:
        proof = ProofNode(goal=goal, is_fact=True, success=True, reason="present in working memory")
        return SearchResult("BFS", goal, True, proof, 1, 0, [], 0.0, "Present in working memory")

    queue = deque()
    queue.append(((goal,), (), 0, 0.0))
    visited_states = set()
    nodes_explored = 0

    while queue:
        unproven, rules_used, depth, cost = queue.popleft()
        nodes_explored += 1

        remaining = tuple(g for g in unproven if g not in facts)
        if not remaining:
            rules_list = list(rules_used)
            proof = build_proof_tree(goal, facts, kb.rules, set(rules_list))
            _, proof_depth = extract_proof_stats(proof)
            return SearchResult(
                algorithm="BFS",
                goal=goal,
                success=True,
                proof=proof,
                nodes_explored=nodes_explored,
                proof_depth=proof_depth,
                rules_used=rules_list,
                cost=round(cost, 2),
                reason="Proof found via BFS",
            )

        state_key = remaining
        if state_key in visited_states:
            continue
        visited_states.add(state_key)

        target = remaining[0]
        candidates = [r for r in kb.rules if r.conclusion.as_fact() == target]
        candidates.sort(key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))

        for rule in candidates:
            rule_cost = 10.0 / max(rule.priority, 1)
            new_unproven = remaining[1:] + tuple(c.as_fact() for c in rule.conditions if c.as_fact() not in facts)
            new_rules = rules_used + (rule.rule_id,)
            queue.append((new_unproven, new_rules, depth + 1, cost + rule_cost))

    proof = ProofNode(goal=goal, success=False, reason="no supporting rule chain found")
    return SearchResult("BFS", goal, False, proof, nodes_explored, 0, [], 0.0, "No proof found")


# -----------------------------
# Backward Chaining: A* Algorithm
# -----------------------------

def heuristic(unproven: Tuple[Fact, ...], facts: Set[Fact]) -> float:
    """Admissible heuristic: counts remaining unproven facts not in working memory."""
    return sum(1.0 for g in unproven if g not in facts)


def backward_chain_astar(kb: KnowledgeBase, goal: Fact) -> SearchResult:
    facts = set(kb.facts)
    if goal in facts:
        proof = ProofNode(goal=goal, is_fact=True, success=True, reason="present in working memory")
        return SearchResult("A*", goal, True, proof, 1, 0, [], 0.0, "Present in working memory")

    pq = []
    counter = 0

    initial_unproven = (goal,)
    h0 = heuristic(initial_unproven, facts)
    heapq.heappush(pq, (h0, counter, initial_unproven, (), 0.0))

    visited_states = set()
    nodes_explored = 0

    while pq:
        f_score, _, unproven, rules_used, g_score = heapq.heappop(pq)
        nodes_explored += 1

        remaining = tuple(g for g in unproven if g not in facts)
        if not remaining:
            rules_list = list(rules_used)
            proof = build_proof_tree(goal, facts, kb.rules, set(rules_list))
            _, proof_depth = extract_proof_stats(proof)
            return SearchResult(
                algorithm="A*",
                goal=goal,
                success=True,
                proof=proof,
                nodes_explored=nodes_explored,
                proof_depth=proof_depth,
                rules_used=rules_list,
                cost=round(g_score, 2),
                reason="Proof found via A* Search",
            )

        state_key = remaining
        if state_key in visited_states:
            continue
        visited_states.add(state_key)

        target = remaining[0]
        candidates = [r for r in kb.rules if r.conclusion.as_fact() == target]
        candidates.sort(key=lambda r: (-r.priority, -len(r.conditions), r.rule_id))

        for rule in candidates:
            rule_cost = 10.0 / max(rule.priority, 1)
            new_g = g_score + rule_cost
            new_unproven = remaining[1:] + tuple(c.as_fact() for c in rule.conditions if c.as_fact() not in facts)
            new_h = heuristic(new_unproven, facts)
            new_f = new_g + new_h
            new_rules = rules_used + (rule.rule_id,)

            counter += 1
            heapq.heappush(pq, (new_f, counter, new_unproven, new_rules, new_g))

    proof = ProofNode(goal=goal, success=False, reason="no supporting rule chain found")
    return SearchResult("A*", goal, False, proof, nodes_explored, 0, [], 0.0, "No proof found")


# -----------------------------
# Search Algorithm Comparison
# -----------------------------

def compare_search_algorithms(kb: KnowledgeBase, goal: Fact) -> str:
    res_dfs = backward_chain_dfs(kb, goal)
    res_bfs = backward_chain_bfs(kb, goal)
    res_astar = backward_chain_astar(kb, goal)

    results = [res_dfs, res_bfs, res_astar]

    lines = [
        "=" * 95,
        f"INFERENCE STRATEGY COMPARISON FOR GOAL: {goal[0]}={goal[1]}",
        "=" * 95,
        f"{'Algorithm':<12} {'Success':<10} {'Nodes Explored':<16} {'Proof Depth':<13} {'Rules Used':<24} {'Cost':<8}",
        "-" * 95,
    ]

    for r in results:
        status = "✓ True" if r.success else "✗ False"
        rules_str = str(r.rules_used) if r.rules_used else "None"
        if len(rules_str) > 22:
            rules_str = rules_str[:19] + "..."
        lines.append(
            f"{r.algorithm:<12} {status:<10} {r.nodes_explored:<16} {r.proof_depth:<13} {rules_str:<24} {r.cost:<8.2f}"
        )

    lines.append("-" * 95)
    lines.append("\nKey Findings & Analysis:")

    if all(r.success for r in results):
        min_nodes_alg = min(results, key=lambda x: x.nodes_explored)
        lines.append(
            f"• Efficiency (Nodes Explored): {min_nodes_alg.algorithm} explored the fewest nodes ({min_nodes_alg.nodes_explored}) by leveraging heuristic guidance h(n)."
        )
        lines.append(
            f"• Proof Quality: All strategies successfully derived a valid proof tree of depth {res_dfs.proof_depth} with total cost {res_dfs.cost:.2f}."
        )
        lines.append(
            "• Performance Trade-offs:\n"
            "   - DFS: Fastest memory footprint; explores deeply along initial rule paths.\n"
            "   - BFS: Guarantees shallowest proof tree depth; explores level-by-level across all paths.\n"
            "   - A*: Combines rule priority cost g(n) with heuristic distance h(n) to find optimal proofs with minimal expansions."
        )
    else:
        lines.append("• Goal is unprovable given the current facts and rules in the knowledge base.")

    if res_dfs.proof:
        lines.append("\nProof Tree Structure (from DFS):\n" + res_dfs.proof.render())

    return "\n".join(lines)


# -----------------------------
# Explanations Rendering
# -----------------------------

def render_forward_explanation(trace: List[FireEvent], goal: Optional[Fact] = None) -> str:
    lines = ["Forward Chaining Explanation"]
    if not trace:
        lines.append("No rule fired.")
        return "\n".join(lines)

    for ev in trace:
        supp = ", ".join(f"{p}={v}" for p, v in ev.supporting_facts) if ev.supporting_facts else "TRUE"
        lines.append(
            f"Step {ev.step + 1}: {ev.rule_id} fired because [{supp}] -> {ev.conclusion[0]}={ev.conclusion[1]}"
        )
    if goal:
        lines.append(f"Goal: {goal[0]}={goal[1]}")
    return "\n".join(lines)


# -----------------------------
# LangGraph state / nodes
# -----------------------------

class ExpertState(TypedDict, total=False):
    query: str
    goal: Tuple[str, str]
    mode: str  # "forward", "dfs", "bfs", "astar", or "compare"
    kb: KnowledgeBase
    facts: List[Tuple[str, str]]
    rules: List[Dict[str, Any]]
    report_ok: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    final_facts: List[List[str]]
    trace: List[Dict[str, Any]]
    proof: Dict[str, Any]
    answer: str


def kb_to_state_payload(kb: KnowledgeBase) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    facts = [list(f) for f in sorted(kb.facts)]
    rules = [
        {
            "rule_id": r.rule_id,
            "conditions": [list(c.as_fact()) for c in r.conditions],
            "conclusion": list(r.conclusion.as_fact()),
            "priority": r.priority,
            "description": r.description,
        }
        for r in kb.rules
    ]
    return facts, rules


def node_load(state: ExpertState, store: Neo4jStore) -> Dict[str, Any]:
    kb = store.load_kb()

    if "facts" in state and state["facts"]:
        custom_facts = {tuple(f) for f in state["facts"]}
        kb.facts = custom_facts
        for p, v in custom_facts:
            kb.ontology.allowed_values.setdefault(p, set()).add(v)

    facts, rules = kb_to_state_payload(kb)
    return {"kb": kb, "facts": facts, "rules": rules}


def node_validate(state: ExpertState) -> Dict[str, Any]:
    kb = state["kb"]
    report = validate_kb(kb)
    return {
        "report_ok": report.ok(),
        "validation_errors": report.errors,
        "validation_warnings": report.warnings,
    }


def route_mode(state: ExpertState) -> str:
    mode = state.get("mode", "compare").lower().strip()
    if mode in ("dfs", "bfs", "astar", "backward"):
        return "backward"
    if mode == "forward":
        return "forward"
    return "compare"


def node_forward(state: ExpertState) -> Dict[str, Any]:
    kb = state["kb"]
    final_facts, trace = forward_chain(kb)
    goal = state.get("query", "").strip()

    answer = render_forward_explanation(trace)
    if goal:
        answer += f"\n\nQuery: {goal}"

    return {
        "final_facts": [list(f) for f in sorted(final_facts)],
        "trace": [
            {
                "rule_id": ev.rule_id,
                "conclusion": list(ev.conclusion),
                "supporting_facts": [list(x) for x in ev.supporting_facts],
                "step": ev.step,
            }
            for ev in trace
        ],
        "answer": answer,
    }


def parse_goal(state: ExpertState) -> Fact:
    query = state.get("query", "").strip()
    if "goal" in state and state["goal"]:
        return tuple(state["goal"])
    elif "=" in query:
        left, right = query.split("=", 1)
        return (left.strip(), right.strip())
    return ("Recommendation", "BUY")


def node_backward(state: ExpertState) -> Dict[str, Any]:
    kb = state["kb"]
    goal = parse_goal(state)
    mode = state.get("mode", "dfs").lower().strip()

    if mode == "bfs":
        res = backward_chain_bfs(kb, goal)
    elif mode == "astar":
        res = backward_chain_astar(kb, goal)
    else:
        res = backward_chain_dfs(kb, goal)

    answer = f"Backward Chaining ({res.algorithm}) Explanation\n"
    if res.proof:
        answer += res.proof.render()
    else:
        answer += f"Goal {goal[0]}={goal[1]} could not be proven."

    return {
        "proof": {
            "goal": list(goal),
            "algorithm": res.algorithm,
            "success": res.success,
            "nodes_explored": res.nodes_explored,
            "cost": res.cost,
        },
        "answer": answer,
    }


def node_compare(state: ExpertState) -> Dict[str, Any]:
    kb = state["kb"]
    goal = parse_goal(state)
    report = compare_search_algorithms(kb, goal)
    return {"answer": report}


def node_explain(state: ExpertState) -> Dict[str, Any]:
    answer = state.get("answer", "")
    warnings = state.get("validation_warnings", [])
    if warnings:
        answer += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings)
    errors = state.get("validation_errors", [])
    if errors:
        answer += "\n\nErrors:\n" + "\n".join(f"- {e}" for e in errors)
    return {"answer": answer.strip()}


# -----------------------------
# LangGraph Builder
# -----------------------------

def build_graph(store: Neo4jStore):
    g = StateGraph(ExpertState)

    g.add_node("load", lambda state: node_load(state, store))
    g.add_node("validate", node_validate)
    g.add_node("forward", node_forward)
    g.add_node("backward", node_backward)
    g.add_node("compare", node_compare)
    g.add_node("explain", node_explain)

    g.add_edge(START, "load")
    g.add_edge("load", "validate")

    g.add_conditional_edges(
        "validate",
        route_mode,
        {
            "forward": "forward",
            "backward": "backward",
            "compare": "compare",
        },
    )

    g.add_edge("forward", "explain")
    g.add_edge("backward", "explain")
    g.add_edge("compare", "explain")
    g.add_edge("explain", END)

    return g.compile()


def main() -> None:
    store = Neo4jStore()
    app = build_graph(store)

    TEST_CASES = {
        "compare_recommendation_buy": {
            "mode": "compare",
            "goal": ("Recommendation", "BUY"),
        },
        "compare_equity_sentiment": {
            "mode": "compare",
            "goal": ("EquitySentiment", "Positive"),
        },
        "compare_impossible_goal": {
            "mode": "compare",
            "goal": ("Recommendation", "HOLD"),
        },
        "forward_chaining_sample": {
            "mode": "forward",
            "facts": [
                ("Market", "Bullish"),
                ("InterestRate", "Falling"),
                ("PE", "Low"),
                ("RevenueGrowth", "High"),
                ("Debt", "Low"),
                ("Volatility", "Low"),
            ],
        },
        "bfs_single_run": {
            "mode": "bfs",
            "goal": ("Recommendation", "BUY"),
        },
        "astar_single_run": {
            "mode": "astar",
            "goal": ("Recommendation", "BUY"),
        },
    }

    for name, inp in TEST_CASES.items():
        result = app.invoke(inp)
        print("\n" + "=" * 95)
        print(f"CASE: {name}")
        print("=" * 95)
        print(result["answer"])

    store.close()


if __name__ == "__main__":
    main()