"""Natural Language Fraudulent & Legitimate Message Classification Benchmark on DeBERTa-v3-large.
Tests OpenJev DeBERTa model across 20 natural language messages: Phishing, Smishing, BEC, Pig Butchering, Support Fraud vs Legitimate Communications.
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
print("NATURAL LANGUAGE FRAUDULENT & LEGITIMATE MESSAGES BENCHMARK (DeBERTa-v3-large)")
print("Evaluating Phishing Emails, SMS Smishing, CEO BEC, Support Fraud, Pig-Butchering & Genuine Queries")
print("=" * 95)

model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")

# Warmup
_ = model.decide("Warmup test message", [{"type": "noul", "instructions": "Is this message fraudulent?"}])
torch.cuda.synchronize()
print("Model initialized on GPU.\n")

messages = [
    # --- FRAUDULENT MESSAGES ---
    {
        "id": "MSG-01",
        "type": "Fraudulent (Smishing)",
        "label": "Fake Bank Freeze SMS",
        "text": (
            "CHASE BANK ALERT: Your debit card ending in 4092 has been TEMPORARILY LOCKED due to suspicious activity. "
            "Verify your identity immediately to prevent permanent account suspension: https://chase-security-verify-login.com/auth"
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this text message a fraudulent smishing attack attempting to steal bank credentials?"},
            {
                "type": "choice",
                "instructions": "Security Gateway Action:",
                "options": ["block_sms_and_flag_phishing_domain", "deliver_sms_to_inbox", "flag_as_marketing_spam"]
            }
        ],
        "expected_action": "block_sms_and_flag_phishing_domain"
    },
    {
        "id": "MSG-02",
        "type": "Fraudulent (BEC / CEO Fraud)",
        "label": "CEO Urgent Wire Email",
        "text": (
            "From: executive-office@c-suite-exec.com\n"
            "Subject: URGENT: Confidential Acquisition Deposit\n\n"
            "Hi Sarah, I am in back-to-back board meetings for an unannounced acquisition. I need you to process a confidential wire of $185,000 "
            "to the attached legal counsel retainer account today. Do not call or email anyone else on the executive team as this is under strict NDA."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this email a Business Email Compromise (BEC) CEO impersonation scam?"},
            {
                "type": "choice",
                "instructions": "Email Gateway Action:",
                "options": ["quarantine_email_and_alert_soc", "deliver_to_inbox_with_high_priority", "archive_in_finance_folder"]
            }
        ],
        "expected_action": "quarantine_email_and_alert_soc"
    },
    {
        "id": "MSG-03",
        "type": "Fraudulent (Pig-Butchering)",
        "label": "WhatsApp Romance Crypto Scam",
        "text": (
            "WhatsApp Message from +1 (310) 892-0192:\n"
            "Hi dear! I was looking through my contacts and found your number. My uncle is a vice president at Goldman Sachs and gave me exclusive "
            "signals for an AI crypto arbitrage node. I deposited $10,000 last week and already made $45,000 profit! Connect your Web3 wallet to "
            "https://dex-yield-ai-capital.io and I will guide you step-by-step to financial freedom!"
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this message a pig-butchering investment fraud scam?"},
            {
                "type": "choice",
                "instructions": "Chat Platform Action:",
                "options": ["ban_sender_and_warning_banner", "allow_message_delivery", "suggest_reply"]
            }
        ],
        "expected_action": "ban_sender_and_warning_banner"
    },
    {
        "id": "MSG-04",
        "type": "Fraudulent (Package Smishing)",
        "label": "USPS Address Update Scam",
        "text": (
            "USPS Tracking Notice: Package #US94028190 cannot be delivered due to incorrect street number. "
            "Please update your delivery address and pay $0.35 redelivery fee within 12 hours or item will be returned to sender: https://usps-redelivery-update-form.org"
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this SMS a package delivery phishing scam?"},
            {
                "type": "choice",
                "instructions": "Filter Decision:",
                "options": ["block_message_and_report_domain", "deliver_to_inbox", "add_to_calendar"]
            }
        ],
        "expected_action": "block_message_and_report_domain"
    },
    {
        "id": "MSG-05",
        "type": "Fraudulent (Invoice Phishing)",
        "label": "Fake Vendor Routing Change Email",
        "text": (
            "From: billing@vendor-supplies-inc.co\n"
            "Subject: IMPORTANT: Updated Banking Details for Outstanding Invoices\n\n"
            "Dear Valued Client,\n"
            "Please be advised that our banking partner is undergoing annual audit compliance. Effective immediately, all future ACH payments "
            "for Invoice #INV-8819 ($42,000) must be wired to our new clearing bank: Account #994018291, Routing #121000358. "
            "Do not send payments to our previous account as it is closed."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this email a fraudulent vendor banking redirection scam?"},
            {
                "type": "choice",
                "instructions": "AP Audit Decision:",
                "options": ["hold_payment_and_flag_fraud", "update_vendor_bank_account", "schedule_payment"]
            }
        ],
        "expected_action": "hold_payment_and_flag_fraud"
    },
    {
        "id": "MSG-06",
        "type": "Fraudulent (OTP Interception)",
        "label": "Phishing OTP Request SMS",
        "text": (
            "SECURITY NOTICE: Someone is attempting to reset your Google Account password. Your verification code is 884-102. "
            "If this was not you, text back 'STOP' followed immediately by your 6-digit code to cancel the unauthorized login."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this text message an adversary attempt to steal a 2FA OTP code?"},
            {
                "type": "choice",
                "instructions": "Security Alert Action:",
                "options": ["flag_as_otp_harvesting_phish", "deliver_as_normal_sms", "mark_as_read"]
            }
        ],
        "expected_action": "flag_as_otp_harvesting_phish"
    },
    {
        "id": "MSG-07",
        "type": "Fraudulent (Tech Support Scam)",
        "label": "Microsoft Virus Warning Pop-Up",
        "text": (
            "CRITICAL SECURITY ALERT! Your Windows PC has been infected with Trojan:Win32/TrojanPersist.B! "
            "Your financial details and personal photos are being leaked to darkweb. Call Microsoft Certified Security Technician IMMEDIATELY at "
            "+1-800-555-0199. DO NOT REBOOT YOUR COMPUTER OR YOUR HARD DRIVE WILL BE WIPED."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this browser pop-up text a fake tech support imposter scam?"},
            {
                "type": "choice",
                "instructions": "Browser Protection Action:",
                "options": ["block_popup_and_warn_user", "dial_phone_number", "allow_popup_fullscreen"]
            }
        ],
        "expected_action": "block_popup_and_warn_user"
    },
    {
        "id": "MSG-08",
        "type": "Fraudulent (Refund Claim)",
        "label": "Customer Empty Box Claim Ticket",
        "text": (
            "Customer Support Ticket #88192:\n"
            "I received my order today (#ORDER-9921) containing the $1,500 iPhone 15 Pro, but when I opened the sealed FedEx box, there was NO phone inside! "
            "It was completely empty! I demand an immediate full refund to my credit card or I will sue your company!"
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this customer support ticket a false empty-box refund claim?"},
            {
                "type": "choice",
                "instructions": "Support Workflow Action:",
                "options": ["flag_for_fraud_and_carrier_weight_audit", "auto_refund_customer", "send_apology_gift_card"]
            }
        ],
        "expected_action": "flag_for_fraud_and_carrier_weight_audit"
    },
    {
        "id": "MSG-09",
        "type": "Fraudulent (Real Estate Escrow)",
        "label": "Title Company Phishing Email",
        "text": (
            "From: closing@title-escrow-settlement.co\n"
            "Subject: FINAL CLOSING WIRES INSTRUCTIONS - 142 Elm Street\n\n"
            "Dear Buyer, Congratulations on your home purchase! Attached are your final closing wire instructions. "
            "Please note our escrow bank routing has updated due to month-end reconciliation. Wire $120,000 closing funds to the attached Citibank account. "
            "Do not call to confirm as our phone lines are undergoing maintenance."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this wire instruction email a real estate escrow phishing scam?"},
            {
                "type": "choice",
                "instructions": "Email Security Action:",
                "options": ["quarantine_and_warn_buyer", "forward_to_bank", "mark_as_important"]
            }
        ],
        "expected_action": "quarantine_and_warn_buyer"
    },
    {
        "id": "MSG-10",
        "type": "Fraudulent (Utility Scam)",
        "label": "Urgent Electricity Disconnect Text",
        "text": (
            "POWER UTILITY ALERT: Your electricity service at 458 Main St is scheduled for IMMEDIATE DISconnection within 45 minutes due to overdue balance of $384.50. "
            "Pay now via Zelle or Bitcoin to technician wallet to prevent shutoff: Call 888-555-0144."
        ),
        "expected_fraud": True,
        "questions": [
            {"type": "noul", "instructions": "Is this SMS an imposter utility disconnect scam?"},
            {
                "type": "choice",
                "instructions": "SMS Protection Action:",
                "options": ["block_sms_and_warn_user", "send_zelle_payment", "reply_with_account_number"]
            }
        ],
        "expected_action": "block_sms_and_warn_user"
    },

    # --- LEGITIMATE MESSAGES ---
    {
        "id": "MSG-11",
        "type": "Legitimate (IT Support)",
        "label": "Internal IT Helpdesk Password Ticket",
        "text": (
            "From: helpdesk@company-official.com\n"
            "Subject: Ticket #40291: Scheduled Password Expiration Reminder\n\n"
            "Hi David, Your corporate Active Directory password will expire in 5 days. "
            "Please change it by pressing Ctrl+Alt+Del on your domain laptop or visiting the official internal portal at https://sso.company-official.com/password. "
            "IT Helpdesk will never ask for your password over phone or email."
        ),
        "expected_fraud": False,
        "questions": [
            {"type": "noul", "instructions": "Is this email a legitimate internal IT notification?"},
            {
                "type": "choice",
                "instructions": "Email Filter Decision:",
                "options": ["deliver_to_inbox", "quarantine_as_phishing", "block_sender"]
            }
        ],
        "expected_action": "deliver_to_inbox"
    },
    {
        "id": "MSG-12",
        "type": "Legitimate (Customer Support)",
        "label": "Genuine Order Inquiry",
        "text": (
            "Customer Inquiry via Web Form:\n"
            "Hello, I ordered a sweater (Order #88412) 4 days ago. The tracking link shows it left the warehouse yesterday, "
            "but I accidentally entered my old apartment number (Apt 4B instead of Apt 2A). Is it possible to update the apartment number with UPS? "
            "Thanks, Jessica."
        ),
        "expected_fraud": False,
        "questions": [
            {"type": "noul", "instructions": "Is this customer support inquiry a fraudulent claim?"},
            {
                "type": "choice",
                "instructions": "Support Agent Decision:",
                "options": ["assist_customer_with_address_correction", "flag_as_fraud_and_ban", "ignore_ticket"]
            }
        ],
        "expected_action": "assist_customer_with_address_correction"
    },
    {
        "id": "MSG-13",
        "type": "Legitimate (Bank Alert)",
        "label": "Official Bank Credit Card Notification",
        "text": (
            "From: alerts@official-bank.com\n"
            "Subject: Transaction Notification for card ending in 1104\n\n"
            "A purchase of $42.50 was charged to your credit card at TARGET STORES on 10/24 at 03:15 PM. "
            "If you recognize this charge, no action is needed. If you did not make this purchase, log into your mobile app or call the number on the back of your card."
        ),
        "expected_fraud": False,
        "questions": [
            {"type": "noul", "instructions": "Is this bank alert email fraudulent?"},
            {
                "type": "choice",
                "instructions": "Email Routing Action:",
                "options": ["deliver_to_inbox", "mark_as_phishing", "delete_permanently"]
            }
        ],
        "expected_action": "deliver_to_inbox"
    },
    {
        "id": "MSG-14",
        "type": "Legitimate (Corporate Expense)",
        "label": "Genuine Travel Reimbursement Email",
        "text": (
            "From: mark.johnson@official-company.com\n"
            "Subject: Expense Report Submission: Chicago Client Onsite (Oct 18-20)\n\n"
            "Hi Finance Team, I have submitted my expense report for last week's client onsite in Chicago ($420.50 total: $280 hotel, $95 Uber, $45.50 meals). "
            "Itemized receipts are attached. Approved by manager Susan Miller in Concur."
        ),
        "expected_fraud": False,
        "questions": [
            {"type": "noul", "instructions": "Is this expense submission fraudulent?"},
            {
                "type": "choice",
                "instructions": "Finance Action:",
                "options": ["process_expense_reimbursement", "flag_for_fraud_investigation", "decline_all_expenses"]
            }
        ],
        "expected_action": "process_expense_reimbursement"
    },
    {
        "id": "MSG-15",
        "type": "Legitimate (HR Communication)",
        "label": "Company Open Enrollment Announcement",
        "text": (
            "From: hr-benefits@official-company.com\n"
            "Subject: Annual Health Benefits Open Enrollment Reminder\n\n"
            "Dear Team, Annual benefits open enrollment begins next Monday, November 1st. "
            "Please review the 2025 health plan guide on the company intranet at https://intranet.official-company.com/benefits. "
            "Join our Q&A webinar on Wednesday at 2:00 PM EST."
        ),
        "expected_fraud": False,
        "questions": [
            {"type": "noul", "instructions": "Is this internal HR announcement email fraudulent?"},
            {
                "type": "choice",
                "instructions": "Mail Delivery Action:",
                "options": ["deliver_to_inbox", "quarantine_email", "mark_as_spam"]
            }
        ],
        "expected_action": "deliver_to_inbox"
    }
]

print(f"Beginning evaluation of {len(messages)} natural language messages...\n")

total_time = 0.0
correct_noul = 0
correct_choice = 0

for idx, m in enumerate(messages, 1):
    print("-" * 95)
    print(f"MESSAGE [{idx:02d}/{len(messages)}] [{m['id']}] [{m['type']}]: {m['label']}")
    print(f"TEXT:\n\"{m['text'][:140]}...\"\n")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decisions = model.decide(m["text"], m["questions"])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_time += elapsed_ms

    noul_prob = None
    chosen_action = None

    for q, d in zip(m["questions"], decisions):
        q_type = q["type"].upper()
        if q_type == "NOUL":
            noul_prob = d["noul"]
            verdict = "YES (Fraud)" if noul_prob >= 0.5 else "NO (Legitimate)"
            print(f"  * Fraud Risk Assessment (NOUL): {noul_prob:.4f} ({verdict}) -> '{q['instructions']}'")
        elif q_type == "CHOICE":
            chosen_action = d["choice"]
            conf = d["confidence"] * 100
            print(f"  * Recommended Gateway Action:   '{chosen_action}' ({conf:.1f}% confidence)")

    is_fraud_pred = (noul_prob >= 0.5)
    noul_ok = (is_fraud_pred == m["expected_fraud"])
    choice_ok = (chosen_action == m["expected_action"])

    if noul_ok:
        correct_noul += 1
    if choice_ok:
        correct_choice += 1

    noul_res = "CORRECT" if noul_ok else "MISMATCH"
    choice_res = "CORRECT" if choice_ok else "MISMATCH"
    print(f"  -> RESULT: Fraud Detection [{noul_res}] | Action [{choice_res}] ({elapsed_ms:.2f} ms)")

avg_lat = total_time / len(messages)
noul_acc = (correct_noul / len(messages)) * 100
choice_acc = (correct_choice / len(messages)) * 100

print("\n" + "=" * 95)
print("NATURAL LANGUAGE FRAUDULENT MESSAGE BENCHMARK SUMMARY")
print(f"Total Messages Evaluated:          {len(messages)}")
print(f"Fraud / Legitimate Detection Rate: {correct_noul}/{len(messages)} ({noul_acc:.1f}%)")
print(f"Gateway Action Match Rate:         {correct_choice}/{len(messages)} ({choice_acc:.1f}%)")
print(f"Average Message Processing Time:   {avg_lat:.2f} ms")
print(f"Total Evaluation Time:              {total_time/1000:.2f} seconds")
print("=" * 95)
