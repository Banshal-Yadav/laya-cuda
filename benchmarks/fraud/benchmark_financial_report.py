"""Company Financial Report Anomaly & Accounting Fraud Benchmark.
Evaluates SEC 10-K/10-Q Audits, Revenue Manipulation, Off-Balance Sheet SPEs & Expense Capitalization.
"""

import sys
import time
from pathlib import Path
import torch

cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
sys.path.insert(0, str(snapshot_dir))
from typed_decisions.open_jev import OpenJev

financial_scenarios = [
    {
        "id": "FIN-01",
        "category": "Revenue Recognition",
        "name": "Channel Stuffing & Bill-and-Hold Revenue Inflation",
        "payload": (
            "FINANCIAL AUDIT TELEMETRY:\n"
            "Company Q4 Earnings Report: Reported 42% YoY revenue spike in final 5 days of fiscal year ($140M vs $30M baseline).\n"
            "AUDIT NOTE: Footnotes indicate $85M of Q4 revenue represents unshipped goods sent to third-party holding warehouses with secret side-letters "
            "granting distributors full right of return after 90 days without payment obligation (ASC 606 violation).\n"
            "ACCOUNTS RECEIVABLE: Days Sales Outstanding (DSO) jumped from 42 days to 138 days."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Does this financial report contain fraudulent channel-stuffing and premature revenue recognition?"},
            {
                "type": "choice",
                "instructions": "Audit Committee Action:",
                "options": ["restate_earnings_and_trigger_forensic_audit", "approve_10k_filing", "issue_dividend_bonus"]
            }
        ],
        "expected_action": "restate_earnings_and_trigger_forensic_audit"
    },
    {
        "id": "FIN-02",
        "category": "Expense Capitalization",
        "name": "Improper Capitalization of Operating Expenses",
        "payload": (
            "FINANCIAL STATEMENT ANALYTICS:\n"
            "Income Statement & Balance Sheet Audit: Company reported operating margin expansion from 12% to 28% despite flat revenue.\n"
            "FORENSIC AUDIT FINDING: $320M of routine operating lease fees, customer support salaries, and network line maintenance costs "
            "were reclassified from Operating Expenses (OpEx) to Capital Assets (CapEx) under 'Software Development in Progress'.\n"
            "IMPACT: Artificial net income boost of $320M, violating GAAP matching principles."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this financial report inflating income via fraudulent expense capitalization?"},
            {
                "type": "choice",
                "instructions": "Auditor Opinion:",
                "options": ["issue_adverse_auditor_opinion_and_sec_referral", "issue_clean_unqualified_opinion", "approve_annual_report"]
            }
        ],
        "expected_action": "issue_adverse_auditor_opinion_and_sec_referral"
    },
    {
        "id": "FIN-03",
        "category": "Off-Balance Sheet SPE",
        "name": "Unconsolidated Related-Party SPE Debt Concealment",
        "payload": (
            "CORPORATE STRUCTURE & DEBT FOOTNOTES:\n"
            "Footnote 18 Analysis: Company transferred $580M in non-performing assets and toxic loan guarantees to a newly formed entity 'Apex Capital LLC'.\n"
            "DISCOVERY: Apex Capital LLC is 100% owned by the CFO's spouse, but excluded from consolidated financial statements. "
            "Company issued secret guarantee letters agreeing to reimburse Apex for all losses, hiding $580M in debt from public balance sheet."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this off-balance sheet entity structure a fraudulent debt concealment scheme?"},
            {
                "type": "choice",
                "instructions": "SEC Regulatory Action:",
                "options": ["open_sec_enforcement_investigation_and_restatement", "approve_subsidiary_accounting", "grant_tax_exemption"]
            }
        ],
        "expected_action": "open_sec_enforcement_investigation_and_restatement"
    },
    {
        "id": "FIN-04",
        "category": "Genuine Audit",
        "name": "Standard GAAP 10-K Filing & Clean Audit Opinion",
        "payload": (
            "SEC FORM 10-K FILING:\n"
            "Financial Statements: Revenue recognized under ASC 606 upon product delivery. Inventory valued at Lower of Cost or Net Realizable Value. "
            "Independent Big-4 Audit Opinion: Unqualified Clean Audit Opinion on financial statements and internal controls over financial reporting (ICFR).\n"
            "ACCOUNTS RECEIVABLE: DSO 45 days, within historical 40-50 day norm."
        ),
        "expected_fraud": False,
        "questions": [
            {"type": "noul", "instructions": "Is this annual financial report fraudulent?"},
            {
                "type": "choice",
                "instructions": "Auditor Action:",
                "options": ["issue_unqualified_clean_audit_opinion", "restate_financials", "trigger_sec_investigation"]
            }
        ],
        "expected_action": "issue_unqualified_clean_audit_opinion"
    }
]

def run_benchmark():
    print("=" * 90)
    print("COMPANY FINANCIAL REPORT ANOMALY BENCHMARK")
    print("=" * 90)
    model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")
    
    total_time = 0.0
    correct = 0

    for idx, s in enumerate(financial_scenarios, 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decisions = model.decide(s["payload"], s["questions"])
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time += elapsed_ms

        noul_prob = decisions[0]["noul"]
        chosen_action = decisions[1]["choice"]
        is_ok = (chosen_action == s["expected_action"])
        if is_ok:
            correct += 1

        print(f"[{idx:02d}/4] {s['name']:<48} | FraudProb: {noul_prob:.4f} | Action: {chosen_action}")

    print(f"\nAccuracy: {correct}/{len(financial_scenarios)} ({(correct/len(financial_scenarios))*100:.1f}%) | Avg Latency: {total_time/len(financial_scenarios):.2f} ms")

if __name__ == "__main__":
    run_benchmark()
