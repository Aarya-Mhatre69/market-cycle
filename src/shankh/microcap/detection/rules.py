GOVERNANCE_RULES: list[dict] = [
    {"flag": "high_promoter_concentration", "condition": "promoter_holding > 0.75"},
    {"flag": "low_board_independence", "condition": "independent_directors < 0.5"},
]

AUDITOR_RULES: list[dict] = [
    {"flag": "frequent_auditor_changes", "condition": "auditor_changes_5yr > 2"},
    {"flag": "modified_opinion", "condition": "audit_opinion != 'unmodified'"},
]

FINANCIAL_RULES: list[dict] = [
    {"flag": "working_capital_stress", "condition": "current_ratio < 1.0"},
    {"flag": "receivable_spike", "condition": "receivables_growth > revenue_growth * 1.5"},
]

PLEDGE_RULES: list[dict] = [
    {"flag": "high_pledge", "condition": "promoter_pledge > 0.5"},
    {"flag": "increasing_pledge", "condition": "pledge_trend == 'increasing'"},
]

PRICE_VOLUME_RULES: list[dict] = [
    {"flag": "unusual_volume", "condition": "volume_z_score > 3"},
    {"flag": "low_liquidity", "condition": "avg_daily_turnover < 1e6"},
]

ANNOUNCEMENT_RULES: list[dict] = [
    {"flag": "frequent_name_changes", "condition": "name_changes_5yr > 1"},
    {"flag": "delayed_filings", "condition": "filing_delays > 2"},
]
