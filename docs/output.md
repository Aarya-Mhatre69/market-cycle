(shankh) PS C:\Users\sstha\Desktop\ai_\projects\shankh> uv run pytest -v -s
================================================================================= test session starts ==================================================================================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\sstha\Desktop\ai_\projects\shankh\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\sstha\Desktop\ai_\projects\shankh
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.14.1, langsmith-0.9.8
collecting ... 

collected 3 items                                                                                                                                                                       

tests/unit/test_financial_advisor.py::test_live_financial_advisor_market_snapshot 
--- [LIVE RUN] Market Snapshot Response ---
[{'type': 'text', 'text': "Here's the current market snapshot for today, 2026-07-11:\n\n*   **NIFTY 50:** Trading at 24636.6383, up by 0.2205%.\n*   **India VIX:** At 13.98.\n\nAdditionally, the broader market shows positive sentiment with 1185 advances against 512 declines, and 60 unchanged stocks. The NIFTY MIDCAP 150 is up by 0.7393%, while NIFTY BANK is up by 0.1003%.\n\n*(Data as of 2026-07-11, fetched using `fetch_market_snapshot`)*", 'extras': {'signature': 'CuMCARFNMg8+t8A8OFDtDixvyQ+Iyl+7/ZqcgUBXApwEUEtpoW0Fz6JK47zaHfTI9iBVXlh+QmBmvUXZGFu8vkbh2gPA5AUNqGxYHI1QSwQgNhwRCi+zvhASRhAZBGwDk/QoRWudIW9d4jjGsLO+UHzNKxFa2f0tXjZ6lLIu6rZ+MNqI3aFv1xHE9EXsSR+5chsgI+NfKDVTZSc40oqh3o/tkIomTXrPkfhayRp0o87elkveuOOL7yI8bBjbkM/uRX+Akkj+6CyqVg7F955aqNtGNRTYuNBUdNhg3FoDNcSXAdQ4Zedty3i4rHQN48XpWUn3slJexTywUT9MZzzbW7jEnVwDvHyxaJcnvRnAxrbCe1hkeYP6AFLXdqux71Xbz+mg30EP+SY0CLkiHUz31CKfKYZ3meD4/RzbVSbuDa1gbSkBnR93cV3dFscjRbruYEqB9ospSxj5u6dofm9MQRMEukaoLg=='}}]
FAILED
tests/unit/test_financial_advisor.py::test_live_financial_advisor_macro_delegation 
--- [LIVE RUN] Macro Analyst Delegation Response ---
Here's a deep analysis of the current rate environment and institutional flow trends, as assessed by Shankh's Macro Analyst sub-agent:

**MACRO REGIME ASSESSMENT**
*   **Regime Label**: **risk-on**
*   **Confidence**: **medium**

**Key Signals**

1.  **Interest Rates**: The RBI's monetary policy remains restrictive, with real rates around 1.5%. The yield curve appears normal, and there are no immediate signals for rate cuts.
2.  **FX / INR**: The Indian Rupee (INR) is stable against the US Dollar, showing a marginal 1-month change of -0.14%. External volatility is currently contained.
3.  **Crude Oil**: Brent crude oil prices are around $77/bbl, which is below recent peaks, helping to ease current account deficit (CAD) pressures.
4.  **Inflation**: Consumer Price Index (CPI) is approximately 4.98%, remaining within the RBI's tolerance band, and core inflation trends are showing signs of softening.
5.  **FII/DII Flows**: There is strong domestic institutional investor (DII) buying, with flows of +₹13,243 Crores, which is offsetting mild foreign institutional investor (FII) inflows. The net institutional bias is positive.

**Macro Thesis**

The overall macro backdrop for Indian equities is considered supportive. Inflation is anchored, crude oil prices are benign, and there is abundant liquidity in the market, primarily driven by robust DII participation. The RBI's restrictive stance is data-dependent and not excessively hawkish. Stable INR and strong domestic flows are helping to mitigate external risks. As a result, equity risk appetite is expected to remain elevated.

**Risks to Watch**

*   A sharp rise in crude oil prices, potentially due to geopolitical supply shocks.
*   A reversal in FII flows if US yields spike or if global risk-off sentiment intensifies.
PASSED
tests/unit/test_financial_advisor.py::test_live_financial_advisor_web_search 
--- [LIVE RUN] Tavily Search Response ---
[{'type': 'text', 'text': 'Here\'s a summary of recent news and analyst commentary regarding the Reserve Bank of India\'s (RBI) monetary policy:\n\n**Top Finding:** The Reserve Bank ofIndia is likely to maintain a **patient approach to monetary policy**, with analysts noting that second-quarter inflation is tracking below the RBI\'s own forecasts from its June meeting. This sentiment is further supported by statements from the Indian central bank chief, indicating that it is "premature to talk about rate hikes."\n\n**Supporting Details:**\n*   **Inflation Outlook**: Current inflation trends are below the RBI\'s projections, providing room for the central bank to avoid immediate tightening.\n*   **Rate Hike Stance**: The RBI\'s leadership suggests that discussions around rate hikes are not currently on the table, reinforcing expectations of policy continuity.\n\n**Source:**\n*   Wall Street Journal (WSJ) and Yahoo Finance articles from recent weeks (via `search_web` tool).', 'extras': {'signature': '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'}}]
FAILED

======================================================================================= FAILURES =======================================================================================
_____________________________________________________________________ test_live_financial_advisor_market_snapshot ______________________________________________________________________

    def test_live_financial_advisor_market_snapshot():
        """
        Verifies that the orchestrator compiles correctly, evaluates the question,
        invokes the real market snapshot tool, and parses the returned mock metrics.
        """
        advisor = FinancialAdvisor()
        thread_id = "live-snapshot-test"
    
        question = (
            "Can you check the current market snapshot and tell me how the NIFTY 50 "
            "and India VIX are looking today?"
        )
    
        response = advisor.ask(question, thread_id=thread_id)
    
        print("\n--- [LIVE RUN] Market Snapshot Response ---")
        print(response)
    
>       assert isinstance(response, str)
E       assert False
E        +  where False = isinstance([{'type': 'text', 'text': "Here's the current market snapshot for today, 2026-07-11:\n\n*   **NIFTY 50:** Trading at 24636.6383, up by 0.2205%.\n* **India VIX:** At 13.98.\n\nAdditionally, the broader market shows positive sentiment with 1185 advances against 512 declines, and 60 unchanged stocks. The NIFTY MIDCAP 150 is up by 0.7393%, while NIFTY BANK is up by 0.1003%.\n\n*(Data as of 2026-07-11, fetched using `fetch_market_snapshot`)*", 'extras': {'signature': 'CuMCARFNMg8+t8A8OFDtDixvyQ+Iyl+7/ZqcgUBXApwEUEtpoW0Fz6JK47zaHfTI9iBVXlh+QmBmvUXZGFu8vkbh2gPA5AUNqGxYHI1QSwQgNhwRCi+zvhASRhAZBGwDk/QoRWudIW9d4jjGsLO+UHzNKxFa2f0tXjZ6lLIu6rZ+MNqI3aFv1xHE9EXsSR+5chsgI+NfKDVTZSc40oqh3o/tkIomTXrPkfhayRp0o87elkveuOOL7yI8bBjbkM/uRX+Akkj+6CyqVg7F955aqNtGNRTYuNBUdNhg3FoDNcSXAdQ4Zedty3i4rHQN48XpWUn3slJexTywUT9MZzzbW7jEnVwDvHyxaJcnvRnAxrbCe1hkeYP6AFLXdqux71Xbz+mg30EP+SY0CLkiHUz31CKfKYZ3meD4/RzbVSbuDa1gbSkBnR93cV3dFscjRbruYEqB9ospSxj5u6dofm9MQRMEukaoLg=='}}], str)

tests\unit\test_financial_advisor.py:39: AssertionError
________________________________________________________________________ test_live_financial_advisor_web_search ________________________________________________________________________

    def test_live_financial_advisor_web_search():
        """
        Verifies that the superviser can trigger a live TavilySearch web query,
        retrieve raw internet content, and synthesize it into a readable research summary.
        """
        advisor = FinancialAdvisor()
        thread_id = "live-search-test"
    
        question = (
            "Search the web for any recent news or analyst commentary regarding "
            "the Indian central bank (RBI) monetary policy and summarize the top finding."
        )
    
        response = advisor.ask(question, thread_id=thread_id)
    
        print("\n--- [LIVE RUN] Tavily Search Response ---")
        print(response)
    
>       assert isinstance(response, str)
E       assert False
E        +  where False = isinstance([{'type': 'text', 'text': 'Here\'s a summary of recent news and analyst commentary regarding the Reserve Bank of India\'s (RBI) monetary policy:\n\n**Top Finding:** The Reserve Bank of India is likely to maintain a **patient approach to monetary policy**, with analysts noting that second-quarter inflation is tracking below the RBI\'s own forecasts from its June meeting. This sentiment is further supported by statements from the Indian central bank chief, indicating that it is "premature to talk about rate hikes."\n\n**Supporting Details:**\n*   **Inflation Outlook**: Current inflation trends are below the RBI\'s projections, providing room for the central bank to avoid immediate tightening.\n*  **Rate Hike Stance**: The RBI\'s leadership suggests that discussions around rate hikes are not currently on the table, reinforcing expectations of policy continuity.\n\n**Source:**\n*   Wall Street Journal (WSJ) and Yahoo Finance articles from recent weeks (via `search_web` tool).', 'extras': {'signature': 'CucNARFNMg90AatOMf29ENu8S/8pBCcEi15E9YqUpvWnrpBzLa/4kzjkabNp83Di14VqtzZc7ofyFnL0NGWP9CCLoZcOXGMJkAs3spgInSAhAURAyAXMnxzWeNx46oh1tonb3N/zGd/797ERtceTiZb/2f4QKyGR7JqoomdhNSmSJhvf0p03dWz...5mIJXOl0oUPiOTY53Je7Ahxp5Q/GgDU7tS95jVO6Eit1vmItAYTmHDqqnII/r3pqoU9sPw3SF5Oo/cZiOkSPF4gqX9y6KBK1MzuHs6vopCavz7R8P92QxEWPesRsQcGS6IbnD+IGiDtyKyZt2o9hJOP0u1KQzdUVoXIlk0qyHBCV7bszpRLamUlXemuBXkfHv8HgnuvCuo4o4tHRGewiWQvBC8bp7DEITThDktFFJxl/e4o8K/XNYnWtO3QC4TyxBmPEjPovGunRWIrqk+hQWU4ZBp3XUuxTsRFSnWmXGKtjCGwk2mBxUuNw1PVFBWuaLl/C/3eq6QGYW5fh2lnLBMZ9sAkhCF5n/PasCoaOF/VE0tNeKT6AVNor3YPpFq5uN89HOQtvs/DwhoOOggEY2RKGf/xSg8wlN6Sg6vxK62PMXgBU8qGdYj3AfNMtM/ZcrrijZVjHWHcdoG3EO/vxlejDlA3ybcE8NRnZnI4UbNO0g4aArwgugzQwyqfESsOxc2EeuFesjzgbjQ7H6FUf4/yCT5Qp9wxIAHbKkPMVArRZAkCcGDl9u1yuE5meATubD0i6heLqPurz1DSsnO13OGKZTicqx2UOC/8+A5hD69YsDgsl4sVVDCiQm0qeqiGE9zJVsl4dum+WaPhz4Bvrmb7IACrdQ1ZxvXbYnoWxyLpd2SEPBrHDQ3CxyKHLo5l5WH1iAQXbtE4ZYnUJllyvEjAOcmZ/h31y4iSEFHPFSCnw7C792no9OPFZfzvmdr67nsiV2ou/7+EVdhdDRWctBmrDGTuQaN3Uz+BLjIQddl0R282oTPDVOom63ZYr5Dm8XqlQI5Nvtqzkw7QH7fEWl2/gcnN4msbC4+z7jvIueY3GlLNuYCCpxTeu6ZrPLlVdxvHP9TQPzA7/HcdbjPpJALhYsOUxBhVJmF1BZ502q8mpouVM21xyOnCxGEZR7MEL4eYHTAmApLJsnhesBf6bPbmNAy+w5fumYAMZ+NyQSZQx7Qr0IglMALxtoAbr50np7HNXvQ55k5qkDNksXsfpeBFWDmMwFbqkBuPyNyoHgwbuwGDwM5O9R+ejzzluD4QYfSS31E8wC5YGwuK12Jfs8XoE7JtaLDmsmxz0k+1R12kD6vsGjI2jIwMl4tScwhSFZ8cNaCWrLut1RmOWF12N8lZO58VJyjgX/W9+pukLB2O'}}], str)

tests\unit\test_financial_advisor.py:88: AssertionError
=============================================================================== short test summary info ================================================================================
FAILED tests/unit/test_financial_advisor.py::test_live_financial_advisor_market_snapshot - assert False
FAILED tests/unit/test_financial_advisor.py::test_live_financial_advisor_web_search - assert False
===========================================================