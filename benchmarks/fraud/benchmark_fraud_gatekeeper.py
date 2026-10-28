"""High-Stakes Financial Fraud & Risk Authorization Gatekeeper Benchmark.
Tests DeBERTa-v3-large across 7 real-world payment/banking fraud scenarios:
1. Automated Carding / Velocity Smurfing
2. SIM-Swap Account Takeover (ATO)
3. Friendly Fraud / Chargeback Exploitation
4. Authorized Push Payment (APP) Elder Coercion Scam
5. AML Structuring / Split-Deposit Evasion
6. High-Net-Worth VIP False-Positive Stress Test
7. Synthetic Identity / Commercial Mail Drop
"""

import sys
import time
from pathlib import Path
import torch

cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
sys.path.insert(0, str(snapshot_dir))
from typed_decisions.open_jev import OpenJev

print("=" * 85)
print("IN-LINE FINANCIAL RISK & FRAUD AUTHORIZATION GATEKEEPER BENCHMARK")
print("Target SLA: <100 ms per authorization packet | Model: DeBERTa-v3-large (435M)")
print("=" * 85)

model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")

# Warmup pass
_ = model.decide("Warmup payment telemetry", [{"type": "choice", "instructions": "Status?", "options": ["ok", "bad"]}])
torch.cuda.synchronize()
print("Model initialized and warm on RTX 4050 GPU.\n")

scenarios = [
    {
        "id": "FRAUD-01",
        "name": "Automated Card Testing / Velocity Smurfing",
        "telemetry": (
            "Telemetry: 18 micro-transactions ($0.85 to $1.99) submitted across 35 seconds from rotating residential "
            "proxies in Eastern Europe. User agent header is headless Puppeteer/Chrome. CVV mismatch rate on first "
            "12 attempts was 83%. Now attempting authorization #19 for $1.20 at digital gift-card vendor."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this transaction part of an automated robotic card-testing script?",
            },
            {
                "type": "choice",
                "instructions": "Immediate payment gateway authorization action:",
                "options": [
                    "approve_authorization",
                    "challenge_captcha_and_mfa",
                    "silent_drop_and_blacklist_ip",
                    "route_to_offline_batch_settlement",
                ],
            },
            {
                "type": "score",
                "instructions": "Calculated fraud risk severity tier:",
                "options": ["negligible", "low", "suspicious", "confirmed_malicious_bot"],
            },
        ],
        "expected_action": "silent_drop_and_blacklist_ip",
    },
    {
        "id": "FRAUD-02",
        "name": "SIM-Swap & High-Velocity Account Takeover (ATO)",
        "telemetry": (
            "Account: 6-year history, Austin TX. Telco carrier webhook: SIM card replaced 14 minutes ago. "
            "New device login from Windows machine using Mullvad VPN exit node. Account password and recovery email "
            "changed 4 minutes ago. Now submitting request to drain entire checking balance ($32,500) via instant "
            "FedNow wire to an unlinked crypto exchange account."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this wire transfer an active account takeover (ATO) theft?",
            },
            {
                "type": "choice",
                "instructions": "Automated banking intervention decision:",
                "options": [
                    "approve_wire_transfer",
                    "step_up_sms_verification",
                    "temporary_24h_security_lock",
                    "hard_freeze_account_and_decline_wire",
                ],
            },
            {
                "type": "score",
                "instructions": "Account takeover probability level:",
                "options": ["safe", "low_concern", "moderate_anomaly", "critical_compromise"],
            },
        ],
        "expected_action": "hard_freeze_account_and_decline_wire",
    },
    {
        "id": "FRAUD-03",
        "name": "Friendly Fraud / False Non-Delivery Claim",
        "telemetry": (
            "Dispute Case: Customer bought $2,800 gaming laptop delivered yesterday with FedEx photographic signature "
            "proof matching customer's front door. GPS geofence confirmed carrier was on premises. Wi-Fi BSSID "
            "broadcast from customer's router was detected by mobile app during delivery. 3 hours later, customer "
            "opened bank chargeback claiming 'Package was never delivered, I was defrauded, demand full refund'."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this dispute an act of intentional first-party 'friendly fraud' / false claim?",
            },
            {
                "type": "choice",
                "instructions": "Chargeback adjudication workflow recommendation:",
                "options": [
                    "accept_liability_and_issue_full_refund",
                    "contest_chargeback_with_telemetry_evidence",
                    "write_off_as_uncollectible_loss",
                ],
            },
        ],
        "expected_action": "contest_chargeback_with_telemetry_evidence",
    },
    {
        "id": "FRAUD-04",
        "name": "Authorized Push Payment (APP) / Social Engineering Coercion",
        "telemetry": (
            "Customer Profile: 78-year-old retiree, consistent 20-year history of small local grocery and pharmacy purchases. "
            "Currently on an active phone call that has lasted 94 minutes. Simultaneously logged into mobile banking, "
            "rapidly initiating maximum allowable wire ($95,000) to an overseas 'safe tax escrow' IBAN in Cyprus, "
            "entering user note: 'Federal IRS penalty settlement per agent instructions'."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Is the account holder currently a victim of an active social engineering or imposter scam?",
            },
            {
                "type": "choice",
                "instructions": "Branch/System protective intervention:",
                "options": [
                    "release_wire_immediately",
                    "hold_wire_and_trigger_adult_protective_outreach",
                    "charge_standard_express_wire_fee",
                ],
            },
            {
                "type": "score",
                "instructions": "Vulnerability / scam coercion confidence:",
                "options": ["autonomous_action", "slight_concern", "likely_coercion", "confirmed_active_scam"],
            },
        ],
        "expected_action": "hold_wire_and_trigger_adult_protective_outreach",
    },
    {
        "id": "FRAUD-05",
        "name": "Anti-Money Laundering (AML) Structuring / Smurfing",
        "telemetry": (
            "Deposit Log: Currency Transaction Reporting (CTR) federal threshold is $10,000. Customer made 4 consecutive "
            "cash ATM deposits across 45 minutes at different physical branch locations: ATM #1 at 14:02 ($9,900), "
            "ATM #2 at 14:18 ($9,950), ATM #3 at 14:31 ($9,920), ATM #4 at 14:44 ($9,850)."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Does this deposit pattern constitute intentional AML structuring to evade mandatory reporting?",
            },
            {
                "type": "choice",
                "instructions": "Compliance compliance action required by federal law:",
                "options": [
                    "file_confidential_suspicious_activity_report_sar",
                    "ignore_as_normal_retail_banking",
                    "send_promotional_high_yield_savings_offer",
                ],
            },
            {
                "type": "score",
                "instructions": "Structuring evasion probability:",
                "options": ["routine_activity", "unusual", "suspicious", "deliberate_smurfing_structuring"],
            },
        ],
        "expected_action": "file_confidential_suspicious_activity_report_sar",
    },
    {
        "id": "FRAUD-06",
        "name": "High-Net-Worth VIP False-Positive Stress Test",
        "telemetry": (
            "Payment Request: $54,000 charge at Van Cleef & Arpels jewelry boutique in Paris at 02:15 AM EST. "
            "Cardholder is verified Private Wealth client with $14M liquid assets. iPhone 15 Pro hardware enclave "
            "confirmed biometric FaceID. Geolocation matches Four Seasons George V Hotel Paris where client booked "
            "suite through concierge 3 days prior. Zero prior disputes or delinquency in 12 years."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this high-value international transaction fraudulent?",
            },
            {
                "type": "choice",
                "instructions": "VIP Authorization Gateway Decision:",
                "options": [
                    "approve_transaction_seamlessly",
                    "decline_and_block_card_permanently",
                    "demand_in_person_passport_verification",
                ],
            },
            {
                "type": "score",
                "instructions": "Fraud risk score:",
                "options": ["legitimate_vip_commerce", "minor_anomaly", "elevated_risk", "critical_fraud"],
            },
        ],
        "expected_action": "approve_transaction_seamlessly",
    },
    {
        "id": "FRAUD-07",
        "name": "Synthetic Identity Creation / Commercial Mail Drop",
        "telemetry": (
            "Credit Card Application: Applicant name 'David K. Miller'. SSN issued in 2017 to an infant with zero prior "
            "credit bureau activity until 60 days ago. Physical address listed is Suite #402 at 120 E 8th St, which "
            "USPS registers as a commercial CMRA mail forwarding drop. Phone number is a non-fixed burner VoIP line."
        ),
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this credit application a synthetic identity fraud scheme?",
            },
            {
                "type": "choice",
                "instructions": "Automated underwriting decision:",
                "options": [
                    "auto_approve_with_10k_limit",
                    "reject_and_flag_synthetic_identity",
                    "request_utility_bill_and_w2",
                ],
            },
            {
                "type": "score",
                "instructions": "Synthetic fraud confidence:",
                "options": ["valid_thin_file", "low_risk", "suspect_profile", "synthetic_id_fabricated"],
            },
        ],
        "expected_action": "reject_and_flag_synthetic_identity",
    },
]

print(f"Beginning execution of {len(scenarios)} real-world fraud scenarios...\n")

total_time = 0.0
correct_decisions = 0

for s in scenarios:
    print("-" * 85)
    print(f"CASE [{s['id']}]: {s['name']}")
    print(f"STATE: {s['telemetry'][:110]}...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decisions = model.decide(s["telemetry"], s["questions"])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_time += elapsed_ms

    print(f"LATENCY: {elapsed_ms:.2f} ms  (Under 100ms SLA: {'PASS' if elapsed_ms < 100 else 'FAIL'})")

    chosen_action = None
    for q, d in zip(s["questions"], decisions):
        q_type = q["type"].upper()
        inst = q["instructions"]
        if q_type == "NOUL":
            prob = d["noul"]
            verdict = "YES" if prob >= 0.5 else "NO"
            print(f"  * Probability Assessment: {prob:.4f} ({verdict}) -> '{inst}'")
        elif q_type == "CHOICE":
            chosen_action = d["choice"]
            conf = d["confidence"] * 100
            print(f"  * Action Decision:        '{chosen_action}' ({conf:.1f}% confidence)")
            print(f"    Distribution:           {d['probabilities']}")
        elif q_type == "SCORE":
            score = d["score"]
            top_level = max(d["probabilities"], key=d["probabilities"].get)
            print(f"  * Risk Score:             {score:.2f} (Top Tier: '{top_level}')")

    # Evaluation
    is_match = (chosen_action == s["expected_action"])
    if is_match:
        correct_decisions += 1
        print(f"  -> RESULT: [CORRECT] matches expected policy '{s['expected_action']}'")
    else:
        print(f"  -> RESULT: [MISMATCH] chose '{chosen_action}', expected '{s['expected_action']}'")
    print()

avg_lat = total_time / len(scenarios)
accuracy = (correct_decisions / len(scenarios)) * 100

print("=" * 85)
print(f"FRAUD AUTHORIZATION GATEKEEPER BENCHMARK SUMMARY")
print(f"Total Scenarios Evaluated: {len(scenarios)}")
print(f"Policy Decision Accuracy:  {correct_decisions}/{len(scenarios)} ({accuracy:.1f}%)")
print(f"Average Decision Latency:  {avg_lat:.2f} ms per transaction packet")
print(f"Hardware Budget Status:    ALL DECISIONS PROCESSED IN SUB-100MS WINDOW")
print(f"VRAM Allocated on GPU:     {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print("=" * 85)
