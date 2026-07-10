WEIGHTS = {
    "governance": 0.25,
    "auditor": 0.15,
    "financial": 0.30,
    "pledge": 0.10,
    "price_volume": 0.10,
    "announcements": 0.10,
}


class RiskScorer:
    def score(self, red_flags: list[dict]) -> dict:
        ...
