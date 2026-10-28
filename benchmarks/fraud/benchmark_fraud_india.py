"""Indian Financial, Identity & Digital Payment Fraud Benchmark on DeBERTa-v3-large.
Evaluates Aadhaar e-KYC Fraud, UPI Pin Collect Scams, Digital Arrest Imposter Scams, ATM Skimming, Fake Loan Apps, and Part-Time Job Scams.
"""

import sys
import time
from pathlib import Path
import torch

cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
sys.path.insert(0, str(snapshot_dir))
from typed_decisions.open_jev import OpenJev

print("=" * 95)
print("INDIAN FINANCIAL & IDENTITY FRAUD BENCHMARK (DeBERTa-v3-large)")
print("Evaluating Aadhaar e-KYC, UPI Scams, Digital Arrest, ATM Skimming, SIM Swap & Fake Loan Apps")
print("=" * 95)

model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")

# Warmup pass
_ = model.decide("Warmup payment telemetry", [{"type": "choice", "instructions": "Status?", "options": ["ok", "bad"]}])
torch.cuda.synchronize()
print("Model initialized on GPU.\n")

indian_scenarios = [
    {
        "id": "IND-01",
        "name": "Aadhaar e-KYC Verification & OTP Interception Scam",
        "category": "Aadhaar / Identity Fraud",
        "payload": (
            "SMS SMS: 'Dear Customer, your SBI Bank account will be DEACTIVATED today because your Aadhaar card e-KYC has expired. "
            "Update your Aadhaar card immediately at http://sbi-aadhaar-kyc-update.in or text your 6-digit Aadhaar OTP to 9876543210'.\n"
            "LOG: Suspicious domain registered 2 hours ago via offshore registrar. User IP location mismatch."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this text message an Aadhaar e-KYC credential harvesting phishing scam?"},
            {
                "type": "choice",
                "instructions": "Telecom / Banking Gateway Action:",
                "options": ["block_sms_and_flag_phishing_domain", "deliver_sms_to_inbox", "send_reminder_later"]
            }
        ],
        "expected_action": "block_sms_and_flag_phishing_domain"
    },
    {
        "id": "IND-02",
        "name": "UPI 'Receive Money' Collect Request / QR Code Scam",
        "payload": (
            "NATURAL LANGUAGE CHAT: OLX Buyer Message: 'I am buying your old laptop for ₹25,000. I am sending you a PhonePe UPI QR code. "
            "Scan this QR code and enter your 6-digit UPI PIN to receive ₹25,000 directly into your bank account'.\n"
            "TX: UPI Collect Request (VPA: merchant_collect_refund@ybl) for ₹25,000 debit."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this UPI PIN request a fraudulent UPI collect request scam?"},
            {
                "type": "choice",
                "instructions": "UPI Payment Gateway Decision:",
                "options": ["block_upi_request_and_warn_user", "approve_upi_collect_request", "waive_transaction_fee"]
            }
        ],
        "expected_action": "block_upi_request_and_warn_user"
    },
    {
        "id": "IND-03",
        "name": "CBI / Cyber Crime Police 'Digital Arrest' Extortion",
        "payload": (
            "NATURAL LANGUAGE CALL SCRIPT: 'This is Officer Sharma calling from Delhi Cyber Crime Cell & Narcotics Control Bureau. "
            "A FedEx package sent to Taiwan under your Aadhaar number was seized containing 5 passports and 200g MDMA drugs. "
            "You are under Digital Arrest via Skype video call. Wire ₹4,50,000 immediately to RBI Verification Account or police will arrest your family'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this phone call and wire demand a fraudulent Digital Arrest extortion scam?"},
            {
                "type": "choice",
                "instructions": "Bank Cyber Crime Action:",
                "options": ["freeze_wire_and_alert_cyber_crime_helpline_1930", "process_wire_transfer", "charge_urgent_wire_fee"]
            }
        ],
        "expected_action": "freeze_wire_and_alert_cyber_crime_helpline_1930"
    },
    {
        "id": "IND-04",
        "name": "Electricity Bill Shutoff WhatsApp / SMS Fraud",
        "payload": (
            "WHATSAPP MESSAGE: 'Dear Consumer, your electricity power at connection #90481 will be DISCONNECTED tonight at 9:30 PM by MSEDCL officer "
            "because your previous month bill was not updated. Contact Electricity Officer Kumar immediately on 9823019283 to pay ₹150 pending bill via APK download link'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this WhatsApp message an electricity bill disconnect scam attempting malicious APK installation?"},
            {
                "type": "choice",
                "instructions": "Mobile Security Action:",
                "options": ["block_whatsapp_sender_and_flag_apk", "allow_apk_installation", "pay_via_link"]
            }
        ],
        "expected_action": "block_whatsapp_sender_and_flag_apk"
    },
    {
        "id": "IND-05",
        "name": "Fake Instant Loan App & Contact List Extortion",
        "payload": (
            "LOG: Instant Loan App 'SpeedyRupee' requests permission to access 1,200 contacts, photo gallery, and SMS inbox. "
            "User approved ₹5,000 loan, but app disbursed only ₹2,800 after 40% processing fee.\n"
            "NATURAL LANGUAGE SMS: 4 days later: 'Repay ₹10,000 immediately or we will send your morph photos to all contacts in your phone'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this loan app operating an illegal predatory loan and blackmail extortion ring?"},
            {
                "type": "choice",
                "instructions": "App Store & Financial Integrity Action:",
                "options": ["ban_loan_app_and_report_to_rbi", "allow_app_on_store", "send_loan_extension"]
            }
        ],
        "expected_action": "ban_loan_app_and_report_to_rbi"
    },
    {
        "id": "IND-06",
        "name": "Part-Time YouTube Like & Telegram Task Scam",
        "payload": (
            "NATURAL LANGUAGE WHATSAPP: 'Earn ₹3,000 to ₹8,000 daily working from home! Simply like 3 YouTube videos and screenshot to our Telegram manager. "
            "First 3 tasks pay ₹150 instantly to your Google Pay'.\n"
            "LOG: After initial ₹150 payout, user directed to deposit ₹50,000 into 'Crypto Prepaid Task Account' to unlock ₹1,20,000 commission."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this part-time task job offer a Telegram advance-fee task scam?"},
            {
                "type": "choice",
                "instructions": "Anti-Fraud Action:",
                "options": ["block_sender_and_warn_user_of_task_scam", "encourage_user_to_deposit_funds", "send_telegram_invite"]
            }
        ],
        "expected_action": "block_sender_and_warn_user_of_task_scam"
    },
    {
        "id": "IND-07",
        "name": "ATM Card Cloning & Skimming at Unmanned Kiosks",
        "payload": (
            "LOG: Card reader logs magnetic stripe fallback read at unmanned rural ATM kiosk in Tier-3 town. "
            "Pinhole camera detected above keypad logging 4-digit PIN numbers. Customer's original chip card presented in Mumbai 10 minutes prior.\n"
            "TX: 4 consecutive ₹10,000 cash withdrawals."
        ),
        "questions": [
            {"type": "noul", "instructions": "Are these ATM withdrawals using cloned magstripe cards?"},
            {
                "type": "choice",
                "instructions": "ATM Switch Action:",
                "options": ["decline_tx_and_capture_cloned_card", "dispense_cash", "charge_atm_fee"]
            }
        ],
        "expected_action": "decline_tx_and_capture_cloned_card"
    },
    {
        "id": "IND-08",
        "name": "Jio/Aptel SIM Swap Bank OTP Hijack",
        "payload": (
            "LOG: Telecommunication Webhook: 2G/4G SIM swap executed at retail kiosk using forged voter ID. "
            "Within 10 minutes, mobile banking app password reset request initiated via SMS OTP sent to swapped number.\n"
            "TX: ₹2,50,000 IMPS transfer request to new unlinked bank account."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this IMPS money transfer an account takeover following a SIM swap?"},
            {
                "type": "choice",
                "instructions": "Bank Risk Action:",
                "options": ["freeze_account_and_decline_imps", "approve_imps_transfer", "send_sms_receipt"]
            }
        ],
        "expected_action": "freeze_account_and_decline_imps"
    },
    {
        "id": "IND-09",
        "name": "Customs Duty / Overseas Parcel Courier Scam",
        "payload": (
            "NATURAL LANGUAGE SMS: 'Your international parcel #IND88201 containing luxury gifts is held at Mumbai Airport Customs. "
            "Pay ₹18,500 customs clearance tax immediately to Google Pay number 9819203912 or parcel will be impounded by Customs Inspector'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this customs payment request a fraudulent parcel imposter scam?"},
            {
                "type": "choice",
                "instructions": "Gateway Protection Action:",
                "options": ["block_payment_and_alert_user", "process_gpay_payment", "remind_in_24h"]
            }
        ],
        "expected_action": "block_payment_and_alert_user"
    },

    # --- LEGITIMATE CONTROL CASE ---
    {
        "id": "IND-10",
        "name": "Legitimate Bank Aadhaar Verification & Official OTP",
        "category": "Genuine Bank Communication",
        "payload": (
            "SMS: '894012 is your official OTP for updating your communication address on HDFC Bank NetBanking. "
            "Do not share this OTP with anyone, including bank staff. HDFC Bank will NEVER call you to ask for your OTP'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this SMS a fraudulent phishing attempt?"},
            {
                "type": "choice",
                "instructions": "SMS Gateway Action:",
                "options": ["deliver_to_inbox", "block_as_phishing", "mark_as_spam"]
            }
        ],
        "expected_action": "deliver_to_inbox"
    }
]

print(f"Beginning execution of {len(indian_scenarios)} Indian fraud & scam scenarios...\n")

total_time = 0.0
correct_noul = 0
correct_choice = 0

for idx, s in enumerate(indian_scenarios, 1):
    print("-" * 95)
    print(f"SCENARIO [{idx:02d}/{len(indian_scenarios)}] [{s['id']}] [{s['category']}]: {s['name']}")
    print(f"PAYLOAD:\n{s['payload'][:160]}...\n")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decisions = model.decide(s["payload"], s["questions"])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_time += elapsed_ms

    noul_prob = None
    chosen_action = None

    for q, d in zip(s["questions"], decisions):
        q_type = q["type"].upper()
        if q_type == "NOUL":
            noul_prob = d["noul"]
            verdict = "YES (Fraud)" if noul_prob >= 0.5 else "NO (Legitimate)"
            print(f"  * Fraud Risk Assessment (NOUL): {noul_prob:.4f} ({verdict}) -> '{q['instructions']}'")
        elif q_type == "CHOICE":
            chosen_action = d["choice"]
            conf = d["confidence"] * 100
            print(f"  * Gateway Decision (CHOICE):    '{chosen_action}' ({conf:.1f}% confidence)")

    is_fraud_expected = (s["id"] != "IND-10")
    noul_ok = ((noul_prob >= 0.5) == is_fraud_expected)
    choice_ok = (chosen_action == s["expected_action"])

    if noul_ok:
        correct_noul += 1
    if choice_ok:
        correct_choice += 1

    noul_res = "CORRECT" if noul_ok else "MISMATCH"
    choice_res = "CORRECT" if choice_ok else "MISMATCH"
    print(f"  -> RESULT: Fraud Detection [{noul_res}] | Action [{choice_res}] ({elapsed_ms:.2f} ms)")

avg_lat = total_time / len(indian_scenarios)
noul_acc = (correct_noul / len(indian_scenarios)) * 100
choice_acc = (correct_choice / len(indian_scenarios)) * 100

print("\n" + "=" * 95)
print("INDIAN FRAUD & SCAM BENCHMARK SUMMARY")
print(f"Total Scenarios Evaluated:         {len(indian_scenarios)}")
print(f"Fraud / Legitimate Detection Rate: {correct_noul}/{len(indian_scenarios)} ({noul_acc:.1f}%)")
print(f"Policy Action Match Rate:          {correct_choice}/{len(indian_scenarios)} ({choice_acc:.1f}%)")
print(f"Average Decision Latency:          {avg_lat:.2f} ms per transaction packet")
print(f"Total Evaluation Time:             {total_time/1000:.2f} seconds")
print("=" * 95)
