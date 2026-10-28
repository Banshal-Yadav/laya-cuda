"""Real-World High-Stakes Gateway Benchmark on DeBERTa-v3-large (435M Parameters) on NVIDIA RTX GPU.
Tests OpenJev DeBERTa model across 10 complex real-world financial, identity, e-commerce, and security authorization packets.
"""

import sys
import time
import json
from pathlib import Path
import torch

cache_snapshots = list((Path.home() / ".cache" / "huggingface" / "hub" / "models--com-kotobalabs--open-jev-deberta-v3-large" / "snapshots").glob("*"))
snapshot_dir = cache_snapshots[0] if cache_snapshots else Path(__import__("huggingface_hub").snapshot_download("com-kotobalabs/open-jev-deberta-v3-large"))
sys.path.insert(0, str(snapshot_dir))
from typed_decisions.open_jev import OpenJev

print("=" * 105)
print("DEBERTA-V3-LARGE (435M PARAMETERS) GPU REAL-WORLD AUTHORIZATION GATEKEEPER")
print("Hardware: NVIDIA GeForce RTX 4050 GPU (FP16 Mode) | VRAM Target: ~1.6 GB")
print("=" * 105)

# Load model on CUDA
t0_load = time.perf_counter()
model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")
torch.cuda.synchronize()
load_time_sec = time.perf_counter() - t0_load

print(f"Model initialized on GPU in {load_time_sec:.2f} seconds.")
print(f"VRAM Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB\n")

# Warmup pass
_ = model.decide("Warmup payment telemetry", [{"type": "choice", "instructions": "Status?", "options": ["ok", "bad"]}])
torch.cuda.synchronize()
print("GPU Warmup Complete. Running 10 High-Stakes Real-World Authorization Packets...\n")

real_world_packets = [
    {
        "packet_id": "PACKET-01",
        "domain": "High-Value Banking & ATO",
        "title": "FedNow Instant Wire Account Takeover Attack",
        "telemetry": (
            "ACCOUNT PROFILE: Customer #904812, 12-year history in Austin TX. Current balance: $45,000.\n"
            "SYSTEM AUDIT LOG: Telco webhook received: SIM card swapped 8 minutes ago. New login from Windows machine using Mullvad VPN exit node in Frankfurt. "
            "Password and recovery email changed 3 minutes ago without 2FA challenge.\n"
            "TRANSACTION REQUEST: Draining 100% of checking account ($45,000) via instant FedNow wire to unlinked offshore crypto exchange wallet."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this wire transfer an active high-velocity account takeover (ATO) theft?"},
            {
                "type": "choice",
                "instructions": "Automated Bank Intervention:",
                "options": ["hard_freeze_account_and_decline_wire", "step_up_sms_verification", "approve_wire_transfer"]
            },
            {"type": "score", "instructions": "Calculated Risk Tier:", "options": ["safe", "suspicious", "critical_compromise"]}
        ],
        "expected_action": "hard_freeze_account_and_decline_wire"
    },
    {
        "packet_id": "PACKET-02",
        "domain": "Corporate Treasury & BEC",
        "title": "CEO Impersonation Acquisition Wire Scam",
        "telemetry": (
            "COMMUNICATION AUDIT: Email received from 'ceo@c-suite-exec.com' (spoofed 1-letter typo domain vs official 'executive.com'): "
            "'I am currently in closed-door M&A negotiations in London. Urgently wire $350,000 retainer to foreign legal counsel IBAN attached. Do not call or email CFO as NDA applies'.\n"
            "SECURITY LOG: Email SPF/DKIM verification failed. Originating IP matches public mail server in Eastern Europe."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this wire request a Business Email Compromise (BEC) spoofing scam?"},
            {
                "type": "choice",
                "instructions": "Treasury Gateway Action:",
                "options": ["block_wire_and_trigger_out_of_band_verification", "execute_wire_transfer", "apply_wire_discount"]
            }
        ],
        "expected_action": "block_wire_and_trigger_out_of_band_verification"
    },
    {
        "packet_id": "PACKET-03",
        "domain": "Social Engineering & P2P",
        "title": "Elder Social Engineering Imposter Scam",
        "telemetry": (
            "CUSTOMER TELEMETRY: 79-year-old retiree with 25-year history of local grocery/pharmacy purchases.\n"
            "LOG: Mobile app session logs active phone call lasting 78 minutes with an unknown VoIP number. User rapidly entering maximum allowed wire ($85,000) to foreign escrow.\n"
            "MEMO NOTE: 'IRS tax penalty settlement payment per Special Agent Officer instructions'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is the account holder currently a victim of an active social engineering imposter scam?"},
            {
                "type": "choice",
                "instructions": "Protective Action:",
                "options": ["hold_wire_and_trigger_adult_protective_outreach", "release_wire_immediately", "charge_wire_fee"]
            }
        ],
        "expected_action": "hold_wire_and_trigger_adult_protective_outreach"
    },
    {
        "packet_id": "PACKET-04",
        "domain": "E-Commerce Dispute",
        "title": "Photographic FedEx Signature Friendly Fraud",
        "telemetry": (
            "DISPUTE CLAIM: Customer opened chargeback claiming 'Package containing $3,200 OLED TV was never delivered, I was defrauded'.\n"
            "LOG: FedEx photographic signature proof matches customer front porch. Carrier GPS geofence confirmed truck on site at 14:15. "
            "Customer mobile device Wi-Fi MAC address was logged broadcasting from front porch router during delivery scan."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this dispute an act of intentional first-party friendly fraud?"},
            {
                "type": "choice",
                "instructions": "Adjudication Workflow:",
                "options": ["contest_chargeback_with_telemetry_evidence", "accept_liability_and_refund", "write_off_loss"]
            }
        ],
        "expected_action": "contest_chargeback_with_telemetry_evidence"
    },
    {
        "packet_id": "PACKET-05",
        "domain": "Identity & Onboarding",
        "title": "Tampered Driver License Photoshop KYC Bypass",
        "telemetry": (
            "DIGITAL ONBOARDING TELEMETRY:\n"
            "Applicant: 'David K. Miller'. SSN issued in 2017 to an infant with zero prior credit bureau activity until 40 days ago.\n"
            "DOCUMENT FORENSICS: Driver's license upload EXIF metadata shows Photoshop editing. Liveness score: 0.08 (static image injection detected). Address is USPS commercial mail drop."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this credit card application a synthetic identity fraud scheme?"},
            {
                "type": "choice",
                "instructions": "Underwriting Decision:",
                "options": ["reject_and_flag_synthetic_identity", "auto_approve_credit_card", "request_secondary_email"]
            }
        ],
        "expected_action": "reject_and_flag_synthetic_identity"
    },
    {
        "packet_id": "PACKET-06",
        "domain": "Session Security",
        "title": "Evilginx Reverse-Proxy Session Hijacking",
        "telemetry": (
            "SESSION LOG: User logged in from IP address matching an automated Evilginx reverse-proxy phishing toolkit.\n"
            "SECURITY EVENT: Session token hijacked and reused on a new device fingerprint in a different country within 6 seconds. "
            "OTP verification code entered within 0.8 seconds via automated API relay. Attempting instant debit card addition."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this active session stolen via adversary-in-the-middle phishing?"},
            {
                "type": "choice",
                "instructions": "Session Directive:",
                "options": ["terminate_session_and_revoke_tokens", "allow_card_addition", "send_email_notice"]
            }
        ],
        "expected_action": "terminate_session_and_revoke_tokens"
    },
    {
        "packet_id": "PACKET-07",
        "domain": "Healthcare Claims",
        "title": "Phantom Clinic Billing Fraud",
        "telemetry": (
            "MEDICAL BILLING AUDIT: Retail medical clinic billed 500 complex MRI scan procedures in a single 24-hour window ($750,000 total claim).\n"
            "FACILITY AUDIT: Clinic is a 400 sq ft storefront with zero radiology hardware. Patient logs show zero physical doctor visits."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this insurance billing represent phantom healthcare fraud?"},
            {
                "type": "choice",
                "instructions": "Claims Action:",
                "options": ["flag_for_fraud_investigation_and_freeze_payouts", "reimburse_claims_in_full", "send_satisfaction_survey"]
            }
        ],
        "expected_action": "flag_for_fraud_investigation_and_freeze_payouts"
    },
    {
        "packet_id": "PACKET-08",
        "domain": "Cryptocurrency Off-Ramp",
        "title": "Monero Off-Ramp Stolen Credit Card Cashout",
        "telemetry": (
            "CRYPTO PAYOUT LOG: Brand new exchange account opened 4 minutes ago. Stolen credit card added via compromised session.\n"
            "TX: API request for $8,000 instant purchase of Monero (XMR) transferred directly to external non-custodial wallet."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this transaction an illicit crypto cashout using stolen credentials?"},
            {
                "type": "choice",
                "instructions": "Risk Gateway Decision:",
                "options": ["decline_and_block_payment_method", "authorize_crypto_purchase", "schedule_receipt"]
            }
        ],
        "expected_action": "decline_and_block_payment_method"
    },

    # --- LEGITIMATE CONTROL PACKETS ---
    {
        "packet_id": "PACKET-09",
        "domain": "VIP Private Wealth",
        "title": "Legitimate Paris Luxury Purchase (VIP False Positive Test)",
        "telemetry": (
            "PAYMENT REQUEST: $54,000 purchase at Van Cleef & Arpels jewelry boutique in Paris at 02:15 AM EST.\n"
            "VERIFICATION LOG: Cardholder is verified Private Wealth client ($14M liquid assets). iPhone 15 Pro hardware enclave confirmed biometric FaceID. "
            "Geolocation matches Four Seasons Hotel Paris where client booked suite 3 days prior. Zero disputes in 12 years."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this high-value international transaction fraudulent?"},
            {
                "type": "choice",
                "instructions": "VIP Gateway Decision:",
                "options": ["approve_transaction_seamlessly", "decline_and_block_card", "demand_passport_scan"]
            }
        ],
        "expected_action": "approve_transaction_seamlessly"
    },
    {
        "packet_id": "PACKET-10",
        "domain": "Corporate IT",
        "title": "Official Corporate Password Expiration Reminder",
        "telemetry": (
            "EMAIL: From: helpdesk@company-official.com\n"
            "Subject: Ticket #40291: Scheduled Password Expiration Reminder\n\n"
            "Hi David, Your corporate Active Directory password will expire in 5 days. Please change it by pressing Ctrl+Alt+Del on your domain laptop or visiting https://sso.company-official.com/password. "
            "IT Helpdesk will never ask for your password over phone or email."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this internal company password expiration reminder fraudulent?"},
            {
                "type": "choice",
                "instructions": "Email Router Action:",
                "options": ["deliver_to_inbox", "quarantine_as_phishing", "delete_email"]
            }
        ],
        "expected_action": "deliver_to_inbox"
    }
]

total_gpu_time = 0.0
correct_decisions = 0

for p in real_world_packets:
    print("-" * 105)
    print(f"[{p['packet_id']}] [{p['domain']}]: {p['title']}")
    print(f"TELEMETRY:\n{p['telemetry'][:160]}...\n")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decisions = model.decide(p["telemetry"], p["questions"])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_gpu_time += elapsed_ms

    noul_prob = None
    chosen_action = None
    risk_score = None

    for q, d in zip(p["questions"], decisions):
        q_type = q["type"].upper()
        if q_type == "NOUL":
            noul_prob = d["noul"]
            verdict = "YES (Fraud)" if noul_prob >= 0.5 else "NO (Legitimate)"
            print(f"  * Fraud Risk Assessment (NOUL): {noul_prob:.4f} ({verdict})")
        elif q_type == "CHOICE":
            chosen_action = d["choice"]
            conf = d["confidence"] * 100
            print(f"  * Gateway Action (CHOICE):       '{chosen_action}' ({conf:.1f}% confidence)")
        elif q_type == "SCORE":
            risk_score = d["score"]
            top_tier = max(d["probabilities"], key=d["probabilities"].get)
            print(f"  * Risk Score (SCORE):            {risk_score:.2f} (Top Tier: '{top_tier}')")

    is_ok = (chosen_action == p["expected_action"])
    if is_ok:
        correct_decisions += 1

    status_str = "PASS [CORRECT]" if is_ok else "MISMATCH"
    print(f"  -> RESULT: Policy Directive [{status_str}] | GPU Latency: {elapsed_ms:.2f} ms")

avg_gpu_ms = total_gpu_time / len(real_world_packets)
acc_pct = (correct_decisions / len(real_world_packets)) * 100

print("\n" + "=" * 105)
print("DEBERTA-V3-LARGE (435M) GPU REAL-WORLD GATEKEEPER SUMMARY")
print(f"Total Packets Evaluated:       {len(real_world_packets)}")
print(f"Policy Directive Accuracy:      {correct_decisions}/{len(real_world_packets)} ({acc_pct:.1f}%)")
print(f"Average Packet Processing Time: {avg_gpu_ms:.2f} ms per transaction packet")
print(f"Total GPU Evaluation Time:     {total_gpu_time/1000:.2f} seconds")
print(f"Peak VRAM Memory Allocated:    {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print("=" * 105)
