"""
seed_neo4j.py

Seeds the Neo4j database with facts and rules for the Expert System graph model.

Run:
  python seed_neo4j.py
"""

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"

FACTS = [
    ("Market", "Bullish"),
    ("InterestRate", "Falling"),
    ("Inflation", "Low"),
    ("PE", "Low"),
    ("RevenueGrowth", "High"),
    ("Debt", "Low"),
    ("Volatility", "Low"),
    ("GDP", "Growing"),
    ("OilPrice", "Stable"),
]

RULES = [
    {
        "id": "R1",
        "priority": 5,
        "description": "Bullish market indicates a strong economy",
        "conditions": [("Market", "Bullish")],
        "conclusion": ("Economy", "Strong"),
    },
    {
        "id": "R2",
        "priority": 4,
        "description": "Growing GDP indicates a strong economy",
        "conditions": [("GDP", "Growing")],
        "conclusion": ("Economy", "Strong"),
    },
    {
        "id": "R3",
        "priority": 5,
        "description": "High inflation weakens the economy",
        "conditions": [("Inflation", "High")],
        "conclusion": ("Economy", "Weak"),
    },
    {
        "id": "R4",
        "priority": 4,
        "description": "Strong economy & falling rates yield positive sentiment",
        "conditions": [
            ("Economy", "Strong"),
            ("InterestRate", "Falling"),
        ],
        "conclusion": ("EquitySentiment", "Positive"),
    },
    {
        "id": "R5",
        "priority": 4,
        "description": "Weak economy yields negative sentiment",
        "conditions": [
            ("Economy", "Weak"),
        ],
        "conclusion": ("EquitySentiment", "Negative"),
    },
    {
        "id": "R6",
        "priority": 3,
        "description": "Positive sentiment, low PE, high growth implies buy candidate",
        "conditions": [
            ("EquitySentiment", "Positive"),
            ("PE", "Low"),
            ("RevenueGrowth", "High"),
        ],
        "conclusion": ("BuyCandidate", "True"),
    },
    {
        "id": "R7",
        "priority": 2,
        "description": "Buy candidate with low debt and volatility leads to BUY recommendation",
        "conditions": [
            ("BuyCandidate", "True"),
            ("Debt", "Low"),
            ("Volatility", "Low"),
        ],
        "conclusion": ("Recommendation", "BUY"),
    },
    {
        "id": "R8",
        "priority": 2,
        "description": "Negative sentiment leads to SELL recommendation",
        "conditions": [
            ("EquitySentiment", "Negative"),
        ],
        "conclusion": ("Recommendation", "SELL"),
    },
    {
        "id": "R9",
        "priority": 3,
        "description": "Rising oil price increases inflation",
        "conditions": [
            ("OilPrice", "Rising"),
        ],
        "conclusion": ("Inflation", "High"),
    },
]


def create_constraints(tx):
    tx.run("""
    CREATE CONSTRAINT fact_unique IF NOT EXISTS
    FOR (f:Fact)
    REQUIRE (f.predicate, f.value) IS UNIQUE
    """)

    tx.run("""
    CREATE CONSTRAINT rule_unique IF NOT EXISTS
    FOR (r:Rule)
    REQUIRE r.id IS UNIQUE
    """)


def clear_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def insert_fact(tx, predicate, value):
    tx.run("""
    MERGE (f:Fact {
        predicate: $p,
        value: $v
    })
    """, p=predicate, v=value)


def insert_rule(tx, rule):
    tx.run("""
    MERGE (r:Rule {id: $id})
    SET
        r.priority = $priority,
        r.description = $description
    """,
    id=rule["id"],
    priority=rule["priority"],
    description=rule.get("description", ""))

    for p, v in rule["conditions"]:
        tx.run("""
        MATCH (r:Rule {id: $id})
        MERGE (f:Fact {
            predicate: $p,
            value: $v
        })
        MERGE (r)-[:REQUIRES]->(f)
        """, id=rule["id"], p=p, v=v)

    cp, cv = rule["conclusion"]
    tx.run("""
    MATCH (r:Rule {id: $id})
    MERGE (f:Fact {
        predicate: $p,
        value: $v
    })
    MERGE (r)-[:PRODUCES]->(f)
    """, id=rule["id"], p=cp, v=cv)


def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            session.execute_write(clear_graph)
            session.execute_write(create_constraints)

            for fact in FACTS:
                session.execute_write(insert_fact, *fact)

            for rule in RULES:
                session.execute_write(insert_rule, rule)

        print("Neo4j database successfully seeded with Facts and Rules.")
    except Exception as e:
        print(f"Failed to seed Neo4j: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()