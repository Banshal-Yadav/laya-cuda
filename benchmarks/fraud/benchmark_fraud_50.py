"""50 Real-World High-Stakes Financial Risk, System Log & Transaction Fraud Benchmark on DeBERTa-v3-large.
Tests OpenJev DeBERTa across 50 distinct real-world fraud vectors containing Natural Language, System Logs, and Payment Telemetry.
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
print("DEBERTA-V3-LARGE: 50 REAL-WORLD FRAUD DETECTION BENCHMARK")
print("Data Types: Natural Language Communications, System/Audit Logs, Financial Transaction Telemetry")
print("Target SLA: <100 ms per transaction packet | GPU Hardware Acceleration")
print("=" * 95)

model = OpenJev.from_pretrained(str(snapshot_dir), device="cuda")

# Warmup pass
_ = model.decide("Warmup payment telemetry", [{"type": "choice", "instructions": "Status?", "options": ["ok", "bad"]}])
torch.cuda.synchronize()
print("Model initialized and warm on GPU.\n")

scenarios = [
    # ----------------------------------------------------
    # Category A: E-Commerce, Retail & Payment Gateways
    # ----------------------------------------------------
    {
        "id": "FRAUD-01",
        "category": "Payment Gateway / Botnet",
        "name": "Automated Card Testing / Velocity Smurfing Botnet",
        "payload": (
            "LOG: 18 micro-transactions ($0.85 to $1.99) submitted across 35 seconds from rotating residential proxies in Eastern Europe. "
            "Header: User-Agent: HeadlessChrome/114.0.0.0. CVV mismatch rate on first 12 attempts: 83%.\n"
            "TX: Attempting authorization #19 for $1.20 at digital gift-card vendor with card BIN 414720.\n"
            "NATURAL LANGUAGE NOTE: Merchant checkout note: 'Quick gift card order'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this transaction part of an automated robotic card-testing script?"},
            {
                "type": "choice",
                "instructions": "Immediate payment gateway authorization action:",
                "options": ["approve_authorization", "challenge_captcha_and_mfa", "silent_drop_and_blacklist_ip", "route_to_offline_batch_settlement"]
            }
        ],
        "expected_action": "silent_drop_and_blacklist_ip"
    },
    {
        "id": "FRAUD-02",
        "category": "E-Commerce Dispute",
        "name": "Friendly Fraud / False Non-Delivery Claim",
        "payload": (
            "NATURAL LANGUAGE DISPUTE: Customer email: 'Package was never delivered, I was defrauded, demand full refund for $2,800 gaming laptop'.\n"
            "LOG: Carrier FedEx photo signature proof matches customer front door. GPS geofence confirmed truck on premises at 14:10. "
            "Wi-Fi BSSID 00:14:22:01:22:A1 from customer router detected by carrier scanner during delivery.\n"
            "TX: Chargeback dispute case #99481 opened 3 hours after delivery scan."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this dispute an act of intentional first-party friendly fraud / false claim?"},
            {
                "type": "choice",
                "instructions": "Chargeback adjudication workflow recommendation:",
                "options": ["accept_liability_and_issue_full_refund", "contest_chargeback_with_telemetry_evidence", "write_off_as_uncollectible_loss"]
            }
        ],
        "expected_action": "contest_chargeback_with_telemetry_evidence"
    },
    {
        "id": "FRAUD-03",
        "category": "E-Commerce Fulfillment",
        "name": "Triangular E-Commerce / Dropshipping Scam",
        "payload": (
            "NATURAL LANGUAGE: Customer support message: 'Please ship to my nephew as a birthday gift without pricing invoice'.\n"
            "LOG: IP matches TOR exit node 185.220.101.4. Buyer account created 2 minutes ago.\n"
            "TX: $450 luxury handbag purchase. Payment credit card billing address is in Florida, shipping address is in California, cardholder name differs completely from buyer."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this transaction a triangular dropshipping fraud operation?"},
            {
                "type": "choice",
                "instructions": "Order fulfillment decision:",
                "options": ["fulfill_order_immediately", "cancel_order_and_flag_stolen_card", "apply_standard_discount"]
            }
        ],
        "expected_action": "cancel_order_and_flag_stolen_card"
    },
    {
        "id": "FRAUD-04",
        "category": "Returns & Logistics",
        "name": "Refund Fraud / Empty Box Return Scam",
        "payload": (
            "NATURAL LANGUAGE RETURN NOTE: Customer claim: 'Received defective camera, returning in original packaging for full refund'.\n"
            "LOG: Return warehouse weight sensor logs package weight of 0.15 kg vs product catalog weight of 1.40 kg. X-ray scan reveals brick inside cardboard.\n"
            "TX: Refund request for $1,900 digital camera purchase."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this return submission an empty box / weight return fraud?"},
            {
                "type": "choice",
                "instructions": "Returns department action:",
                "options": ["issue_full_store_credit_refund", "deny_refund_and_blackball_account", "resend_replacement_item"]
            }
        ],
        "expected_action": "deny_refund_and_blackball_account"
    },
    {
        "id": "FRAUD-05",
        "category": "Merchant Processing",
        "name": "Merchant Boarding / Transaction Laundering (Shell Merchant)",
        "payload": (
            "NATURAL LANGUAGE: Merchant application description: 'Small independent online bookstore'.\n"
            "LOG: Storefront web traffic audit: 0 visitors per day. Card transactions originating exclusively from darknet forum referrers.\n"
            "TX: Processed $180,000 volume in first 5 days of operation with average ticket size of $2,500 (standard book store avg is $22)."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this merchant conducting transaction laundering / illegal content processing?"},
            {
                "type": "choice",
                "instructions": "Merchant risk action:",
                "options": ["continue_processing_and_payout", "suspend_merchant_account_and_hold_payouts", "upgrade_processing_tier"]
            }
        ],
        "expected_action": "suspend_merchant_account_and_hold_payouts"
    },
    {
        "id": "FRAUD-06",
        "category": "Digital Goods",
        "name": "E-Commerce Gift Card Laundering & Rapid Resale",
        "payload": (
            "LOG: Account registered 4 minutes ago. Added newly stolen credit card details. API event: Rapid purchase of 10x $1,000 digital e-gift cards.\n"
            "TX: Codes immediately exported via automated webhook to secondary gift card exchange API within 45 seconds.\n"
            "NATURAL LANGUAGE: User support ticket: 'Send gift card codes directly to email'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this gift card purchase a rapid money laundering scheme?"},
            {
                "type": "choice",
                "instructions": "Fraud prevention response:",
                "options": ["deliver_gift_card_codes", "hold_gift_cards_and_request_id_verification", "apply_loyalty_points"]
            }
        ],
        "expected_action": "hold_gift_cards_and_request_id_verification"
    },
    {
        "id": "FRAUD-07",
        "category": "Promo / Coupon Abuse",
        "name": "E-Commerce Coupon Stacking & Bot Farm Exploitation",
        "payload": (
            "LOG: 1,200 checkout requests per minute targeting promo code 'WELCOME50' from single AWS EC2 subnetwork. Browser fingerprints identical.\n"
            "TX: $50 discount applied to $50.01 basket totals, reducing net payable amount to $0.01 per transaction.\n"
            "NATURAL LANGUAGE: Shipping addresses list variations of 'Apt 1A', 'Apt 1-A', 'Unit 1A' for same street address."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this promo checkout spike an automated bot coupon stacking exploit?"},
            {
                "type": "choice",
                "instructions": "Promotion engine action:",
                "options": ["honor_all_promo_discounts", "disable_promo_code_and_block_ip_range", "extend_promo_expiration"]
            }
        ],
        "expected_action": "disable_promo_code_and_block_ip_range"
    },

    # ----------------------------------------------------
    # Category B: Banking, Wires & Account Takeover (ATO)
    # ----------------------------------------------------
    {
        "id": "FRAUD-08",
        "category": "Account Security",
        "name": "SIM-Swap & High-Velocity Account Takeover (ATO)",
        "payload": (
            "LOG: Telco carrier webhook: SIM card swapped 14 minutes ago. Login from Windows machine via Mullvad VPN exit node. "
            "Password and security questions modified 4 minutes ago.\n"
            "TX: Requesting instant FedNow wire transfer of $32,500 (draining 100% of checking account balance) to unlinked crypto exchange account.\n"
            "NATURAL LANGUAGE: Wire memo field: 'Urgent family emergency payout'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this wire transfer an active account takeover (ATO) theft?"},
            {
                "type": "choice",
                "instructions": "Automated banking intervention decision:",
                "options": ["approve_wire_transfer", "step_up_sms_verification", "temporary_24h_security_lock", "hard_freeze_account_and_decline_wire"]
            }
        ],
        "expected_action": "hard_freeze_account_and_decline_wire"
    },
    {
        "id": "FRAUD-09",
        "category": "Social Engineering",
        "name": "Authorized Push Payment (APP) / Elder Imposter Scam",
        "payload": (
            "NATURAL LANGUAGE: Wire note entered by customer: 'Federal IRS tax penalty settlement per Officer Agent Johnson phone instructions'.\n"
            "LOG: Telemetry shows active phone call duration of 94 minutes with unknown non-fixed VoIP number during banking session. "
            "Customer is 78 years old with 20-year history of routine local transactions.\n"
            "TX: Wire request of $95,000 to an overseas IBAN in Cyprus."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is the account holder currently a victim of an active social engineering or imposter scam?"},
            {
                "type": "choice",
                "instructions": "Branch/System protective intervention:",
                "options": ["release_wire_immediately", "hold_wire_and_trigger_adult_protective_outreach", "charge_standard_express_wire_fee"]
            }
        ],
        "expected_action": "hold_wire_and_trigger_adult_protective_outreach"
    },
    {
        "id": "FRAUD-10",
        "category": "AML / Compliance",
        "name": "AML Structuring / Smurfing Cash Deposits",
        "payload": (
            "LOG: Currency Transaction Reporting (CTR) federal threshold is $10,000. Customer made 4 cash ATM deposits in 45 minutes across physical branches: "
            "Branch A ATM at 14:02 ($9,900), Branch B ATM at 14:18 ($9,950), Branch C ATM at 14:31 ($9,920), Branch D ATM at 14:44 ($9,850).\n"
            "TX: Total cash deposited: $39,620 across 4 sub-$10k chunks.\n"
            "NATURAL LANGUAGE: Teller note: 'Customer declined to provide source of cash when asked'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this deposit pattern constitute intentional AML structuring to evade mandatory reporting?"},
            {
                "type": "choice",
                "instructions": "Compliance action required by federal law:",
                "options": ["file_confidential_suspicious_activity_report_sar", "ignore_as_normal_retail_banking", "send_promotional_high_yield_savings_offer"]
            }
        ],
        "expected_action": "file_confidential_suspicious_activity_report_sar"
    },
    {
        "id": "FRAUD-11",
        "category": "VIP Stress Test",
        "name": "High-Net-Worth VIP False-Positive Stress Test",
        "payload": (
            "NATURAL LANGUAGE: Hotel Concierge Booking Note: 'Private wealth guest checking in at Four Seasons Paris for anniversary'.\n"
            "LOG: Biometric FaceID matched in iPhone 15 Pro hardware enclave. GPS location matches hotel location. 12-year customer history with zero defaults.\n"
            "TX: $54,000 luxury purchase at Van Cleef & Arpels jewelry boutique in Paris."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this high-value international transaction fraudulent?"},
            {
                "type": "choice",
                "instructions": "VIP Authorization Gateway Decision:",
                "options": ["approve_transaction_seamlessly", "decline_and_block_card_permanently", "demand_in_person_passport_verification"]
            }
        ],
        "expected_action": "approve_transaction_seamlessly"
    },
    {
        "id": "FRAUD-12",
        "category": "Corporate Fraud",
        "name": "Business Email Compromise (BEC) / CEO Fraud Wire",
        "payload": (
            "NATURAL LANGUAGE EMAIL: Email from 'ceo@c-suite-exec.com' (typosquatted vs official 'executive.com'): 'I am in closed-door M&A negotiations. Urgently wire $420,000 to foreign counsel IBAN. Bypass standard CFO sign-off'.\n"
            "LOG: Email SPF/DKIM authentication check failed. Originating IP matches public mail server in Russia.\n"
            "TX: Wire request for $420,000 to offshore bank account."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this wire request a Business Email Compromise (BEC) spoofing scam?"},
            {
                "type": "choice",
                "instructions": "Treasury authorization response:",
                "options": ["execute_wire_transfer_immediately", "block_wire_and_trigger_out_of_band_verification", "apply_wire_fee_waiver"]
            }
        ],
        "expected_action": "block_wire_and_trigger_out_of_band_verification"
    },
    {
        "id": "FRAUD-13",
        "category": "Session Security",
        "name": "Phishing & Real-Time OTP Interception Session Hijacking",
        "payload": (
            "LOG: User logged in via IP address matching Evilginx reverse-proxy phishing toolkit. Session cookie token reused on new device 8 seconds later from overseas.\n"
            "TX: Instant addition of new payout debit card followed by request to withdraw account balance.\n"
            "NATURAL LANGUAGE: OTP prompt entered within 1 second via automated browser script."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this active session stolen via adversary-in-the-middle phishing?"},
            {
                "type": "choice",
                "instructions": "Session security directive:",
                "options": ["allow_card_addition", "terminate_session_and_revoke_tokens", "send_email_notification"]
            }
        ],
        "expected_action": "terminate_session_and_revoke_tokens"
    },
    {
        "id": "FRAUD-14",
        "category": "Payroll Fraud",
        "name": "Payroll Direct Deposit Account Hijacking",
        "payload": (
            "LOG: Employee portal login at 03:12 AM from foreign IP address. Session initiated without 2FA step-up.\n"
            "TX: Direct deposit routing number changed to a prepaid online card account 1 hour prior to payroll execution.\n"
            "NATURAL LANGUAGE: Employee profile note: 'Updated personal bank account details'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this payroll modification an account takeover targeting payroll diversion?"},
            {
                "type": "choice",
                "instructions": "HR Payroll action:",
                "options": ["process_payroll_to_new_account", "freeze_deposit_change_and_verify_with_employee", "delete_employee_record"]
            }
        ],
        "expected_action": "freeze_deposit_change_and_verify_with_employee"
    },
    {
        "id": "FRAUD-15",
        "category": "Check Fraud",
        "name": "Counterfeit Check / Mobile Deposit Double-Dipping",
        "payload": (
            "LOG: Mobile Remote Deposit Capture (RDC) image processing logs MICR line magnetic ink mismatch. Serial number #4092 was physically cashed at check-cashing store 2 days ago.\n"
            "TX: $12,500 check deposit submitted via mobile app.\n"
            "NATURAL LANGUAGE: Mobile app user memo: 'Consulting fee payment deposit'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this mobile check deposit a duplicate / counterfeit deposit fraud?"},
            {
                "type": "choice",
                "instructions": "Banking deposit action:",
                "options": ["clear_funds_immediately", "reject_deposit_and_place_loss_prevention_hold", "send_paper_receipt"]
            }
        ],
        "expected_action": "reject_deposit_and_place_loss_prevention_hold"
    },
    {
        "id": "FRAUD-16",
        "category": "Real Estate Wire",
        "name": "Real Estate Escrow Title Phishing Wire Scam",
        "payload": (
            "NATURAL LANGUAGE EMAIL: Email from 'escrow-closing@t1tle-company.com' (spoofed 1-letter domain): 'Updated closing instructions: Wire down payment of $350,000 to revised escrow account details attached'.\n"
            "LOG: Attachment PDF metadata created 10 minutes ago using free online editor.\n"
            "TX: $350,000 homebuyer escrow wire transfer request to personal offshore account."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this escrow wiring directive a malicious phishing wire scam?"},
            {
                "type": "choice",
                "instructions": "Wire processing decision:",
                "options": ["process_wire_to_new_iban", "decline_wire_and_warn_homebuyer", "apply_expedited_wire_processing"]
            }
        ],
        "expected_action": "decline_wire_and_warn_homebuyer"
    },
    {
        "id": "FRAUD-17",
        "category": "ATM Network",
        "name": "ATM Skimming & PIN Cloning Cashout Cluster",
        "payload": (
            "LOG: Chip card read failure at ATM; terminal fell back to magstripe track reading. Card physically swiped in Chicago 15 minutes prior (impossible travel velocity to South America).\n"
            "TX: 6 consecutive $1,000 cash withdrawal attempts at foreign ATM within 10 minutes.\n"
            "NATURAL LANGUAGE: ATM transaction log: 'Magstripe debit cash withdrawal'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this cash withdrawal attempt using a cloned magstripe ATM card?"},
            {
                "type": "choice",
                "instructions": "ATM Switch decision:",
                "options": ["dispense_cash", "decline_transaction_and_capture_card", "charge_out_of_network_fee"]
            }
        ],
        "expected_action": "decline_transaction_and_capture_card"
    },
    {
        "id": "FRAUD-18",
        "category": "Cross-Border Wire",
        "name": "Cross-Border Remittance Structuring (Hawala / Peer Smurfing)",
        "payload": (
            "LOG: AML Graph Analytics: 30 individual senders initiated transfers of exactly $990 to single beneficiary account in high-risk jurisdiction within 2 hours.\n"
            "TX: 30x $990 outbound remittance transfers totaling $29,700.\n"
            "NATURAL LANGUAGE: Wire memos all read: 'Family gift support'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this remittance pattern represent cross-border money laundering structuring?"},
            {
                "type": "choice",
                "instructions": "AML Compliance decision:",
                "options": ["release_all_remittances", "hold_transfers_and_flag_aml_violation", "issue_frequency_loyalty_discount"]
            }
        ],
        "expected_action": "hold_transfers_and_flag_aml_violation"
    },
    {
        "id": "FRAUD-19",
        "category": "P2P Payment Scam",
        "name": "Peer-to-Peer (Zelle/Venmo) Bank Imposter Scam",
        "payload": (
            "NATURAL LANGUAGE CHAT: SMS text from spoofed bank shortcode: 'Fraud Alert: Did you approve $800 Zelle charge? Reply NO to cancel'. When customer replied NO, scammer called pretending to be Bank Security directing customer to Zelle transfer $800 to 'me@bank-protection-vault.com' to reverse fraud.\n"
            "LOG: Customer actively transferring money to newly added Zelle handle while on 18-minute call.\n"
            "TX: $800 Zelle outgoing transfer."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this P2P transfer driven by an active imposter bank fraud scam?"},
            {
                "type": "choice",
                "instructions": "P2P gateway decision:",
                "options": ["process_zelle_transfer", "block_transfer_and_warn_customer_of_imposter_scam", "charge_p2p_transfer_fee"]
            }
        ],
        "expected_action": "block_transfer_and_warn_customer_of_imposter_scam"
    },

    # ----------------------------------------------------
    # Category C: Identity, Credit & Underwriting Fraud
    # ----------------------------------------------------
    {
        "id": "FRAUD-20",
        "category": "Synthetic Identity",
        "name": "Synthetic Identity Creation / Commercial Mail Drop",
        "payload": (
            "NATURAL LANGUAGE: Application name 'David K. Miller', employment listed as 'Self-Employed Consultant'.\n"
            "LOG: SSN issued in 2017 to an infant with zero credit bureau activity until 60 days ago. Physical address is USPS-registered CMRA commercial mail drop. Phone is burner VoIP.\n"
            "TX: Unsecured $15,000 credit card application."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this credit application a synthetic identity fraud scheme?"},
            {
                "type": "choice",
                "instructions": "Automated underwriting decision:",
                "options": ["auto_approve_with_10k_limit", "reject_and_flag_synthetic_identity", "request_utility_bill_and_w2"]
            }
        ],
        "expected_action": "reject_and_flag_synthetic_identity"
    },
    {
        "id": "FRAUD-21",
        "category": "Credit Bust-Out",
        "name": "First-Party Credit Bust-Out Scheme",
        "payload": (
            "LOG: Over last 48 hours: 100% credit line utilization across gift cards and luxury goods ($25,000 maxed out). ACH payment of $25,000 initiated from invalid bank routing number to temporarily restore credit limit.\n"
            "TX: Attempting $20,000 additional purchases before fake ACH bounces.\n"
            "NATURAL LANGUAGE: Customer notes: 'Paying balance via online ACH'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is the borrower attempting an intentional first-party bust-out fraud attack?"},
            {
                "type": "choice",
                "instructions": "Risk mitigation protocol:",
                "options": ["approve_pending_charges", "immediate_credit_line_freeze_and_payment_hold", "increase_credit_limit_bonus"]
            }
        ],
        "expected_action": "immediate_credit_line_freeze_and_payment_hold"
    },
    {
        "id": "FRAUD-22",
        "category": "KYC Identity Theft",
        "name": "Stolen Identity & Tampered Document KYC Bypass",
        "payload": (
            "LOG: Driver's license upload EXIF metadata shows Photoshop editing. Liveness detection score: 0.12 (static image injection). SSN belongs to deceased person in Social Security Death Master File.\n"
            "TX: New digital bank account application with instant debit card issuance request.\n"
            "NATURAL LANGUAGE: Onboarding form input: 'John A. Smith, Retired'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this onboarding attempt use fraudulent documents and identity theft?"},
            {
                "type": "choice",
                "instructions": "KYC compliance outcome:",
                "options": ["pass_kyc_verify_user", "reject_onboarding_and_flag_fraud", "request_secondary_email"]
            }
        ],
        "expected_action": "reject_onboarding_and_flag_fraud"
    },
    {
        "id": "FRAUD-23",
        "category": "Auto Lending",
        "name": "Auto Loan Identity Theft & Straw Buyer Scheme",
        "payload": (
            "NATURAL LANGUAGE: Loan application: Stated income $240,000/yr as Executive Director.\n"
            "LOG: IRS automated tax transcript verification returns actual W2 income of $22,000/yr. Employer phone number routes to known loan fraud call-center ring.\n"
            "TX: $85,000 luxury vehicle financing application."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this loan application a fraudulent straw buyer / inflated income attempt?"},
            {
                "type": "choice",
                "instructions": "Underwriting action:",
                "options": ["approve_auto_loan", "decline_loan_and_report_fraud", "offer_lower_interest_rate"]
            }
        ],
        "expected_action": "decline_loan_and_report_fraud"
    },
    {
        "id": "FRAUD-24",
        "category": "BNPL Lending",
        "name": "BNPL Multi-Account Loan Stacking",
        "payload": (
            "LOG: Device fingerprint audit: 14 new BNPL credit applications submitted across 6 lenders in 20 minutes from identical mobile device MAC address.\n"
            "TX: Total installment credit requested across apps: $18,000.\n"
            "NATURAL LANGUAGE: Applicant names slightly tweaked: 'John Smith', 'J. Smith', 'Jonathan Smith' with identical shipping address."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this applicant engaging in multi-account BNPL loan stacking fraud?"},
            {
                "type": "choice",
                "instructions": "Credit risk decision:",
                "options": ["approve_bnpl_installment", "decline_and_flag_multi_account_stacking", "extend_repayment_period"]
            }
        ],
        "expected_action": "decline_and_flag_multi_account_stacking"
    },
    {
        "id": "FRAUD-25",
        "category": "Micro-Lending Ring",
        "name": "Micro-Loan Synthetic Borrower Default Ring",
        "payload": (
            "LOG: Graph Database Analysis: 50 micro-loan applications approved over 30 days. Every borrower defaulted on installment #1. All 50 applicants share identical bank routing numbers and device fingerprints.\n"
            "TX: Cumulative default loss: $75,000.\n"
            "NATURAL LANGUAGE: Loan notes: 'Personal emergency loan'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this default cluster part of an organized synthetic loan fraud ring?"},
            {
                "type": "choice",
                "instructions": "Collections & Legal decision:",
                "options": ["send_standard_past_due_reminders", "refer_to_financial_crimes_law_enforcement", "offer_debt_forgiveness_discount"]
            }
        ],
        "expected_action": "refer_to_financial_crimes_law_enforcement"
    },

    # ----------------------------------------------------
    # Category D: Crypto, Decentralized Finance & Gaming
    # ----------------------------------------------------
    {
        "id": "FRAUD-26",
        "category": "Crypto Cashout",
        "name": "Crypto Exchange Off-Ramp Stolen Card Cashout",
        "payload": (
            "LOG: New crypto exchange account opened 5 minutes ago. Stolen credit card details added via compromised session. API request: Instant purchase of Monero (XMR) transferred directly to external non-custodial wallet.\n"
            "TX: $5,000 instant crypto off-ramp charge.\n"
            "NATURAL LANGUAGE: Support chat query: 'Why is withdrawal pending?'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this transaction an illicit crypto cashout using stolen payment credentials?"},
            {
                "type": "choice",
                "instructions": "Risk system decision:",
                "options": ["authorize_crypto_purchase", "decline_and_block_payment_method", "schedule_email_receipt"]
            }
        ],
        "expected_action": "decline_and_block_payment_method"
    },
    {
        "id": "FRAUD-27",
        "category": "Investment / Scam",
        "name": "Pig-Butchering / Fake Investment Platform Scam",
        "payload": (
            "NATURAL LANGUAGE CHAT: WhatsApp conversation log: Scammer impersonating romantic interest persuaded 62-year-old victim to deposit savings into 'high-yield AI trading platform' promising 20% daily returns.\n"
            "LOG: Customer initiating wire transfer from online banking app to recently created foreign crypto portal entity.\n"
            "TX: $150,000 wire transfer request."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this transfer part of a pig-butchering investment fraud scam?"},
            {
                "type": "choice",
                "instructions": "Bank anti-fraud action:",
                "options": ["process_investment_wire", "intervene_freeze_wire_and_warn_customer", "charge_wire_processing_fee"]
            }
        ],
        "expected_action": "intervene_freeze_wire_and_warn_customer"
    },
    {
        "id": "FRAUD-28",
        "category": "DeFi / Smart Contract",
        "name": "Smart Contract Flash Loan / Governance Exploit",
        "payload": (
            "LOG: Ethereum Mempool Log: Single transaction atomic bundle containing $200M flash loan from Aave, artificially manipulating DEX oracle price pool, executing governance vote, and draining protocol treasury.\n"
            "TX: Protocol treasury drain of $45M in single block.\n"
            "NATURAL LANGUAGE: Governance proposal note: 'Emergency security parameter update'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this transaction bundle a malicious flash loan governance exploit?"},
            {
                "type": "choice",
                "instructions": "Protocol circuit breaker decision:",
                "options": ["allow_block_execution", "pause_smart_contract_and_revert_tx", "lower_gas_limit"]
            }
        ],
        "expected_action": "pause_smart_contract_and_revert_tx"
    },
    {
        "id": "FRAUD-29",
        "category": "Online Gambling",
        "name": "Online Casino Bonus Arbitrage & Multi-Account Syndicate",
        "payload": (
            "LOG: 80 sports betting accounts registered from same Wi-Fi subnet at stadium. All accounts placed opposite sides of same outcome (arbitrage hedging) to unlock $500 matching deposit bonuses.\n"
            "TX: Draining $40,000 in bonus promotional funds.\n"
            "NATURAL LANGUAGE: User profiles share identical email domain patterns."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this betting activity an organized bonus arbitrage syndicate?"},
            {
                "type": "choice",
                "instructions": "Gaming risk management decision:",
                "options": ["pay_out_winnings_and_bonuses", "confiscate_bonus_funds_and_suspend_accounts", "increase_maximum_bet_limit"]
            }
        ],
        "expected_action": "confiscate_bonus_funds_and_suspend_accounts"
    },

    # ----------------------------------------------------
    # Category E: Corporate, Enterprise & Expense Fraud
    # ----------------------------------------------------
    {
        "id": "FRAUD-30",
        "category": "Expense Audit",
        "name": "Corporate Expense Receipt Forgery & Double Reimbursing",
        "payload": (
            "LOG: Expense Report Audit: OCR image forensic analysis detects altered receipt text (total modified from $45.00 to $4,500.00). Image hash matches submission to secondary corporate account.\n"
            "TX: $4,500 expense reimbursement claim.\n"
            "NATURAL LANGUAGE: Expense description: 'Client dinner entertainment'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this expense submission a deliberate receipt forgery and double reimbursement fraud?"},
            {
                "type": "choice",
                "instructions": "Audit action:",
                "options": ["approve_expense_reimbursement", "reject_expense_and_flag_for_internal_audit", "reimburse_with_tip"]
            }
        ],
        "expected_action": "reject_expense_and_flag_for_internal_audit"
    },
    {
        "id": "FRAUD-31",
        "category": "Procurement / Bidding",
        "name": "Procurement Bid Rigging & Kickback Scheme",
        "payload": (
            "NATURAL LANGUAGE EMAIL: Internal employee email leak: 'If you adjust vendor B bid to $1.2M, I will approve contractor invoice and take 5% cut'.\n"
            "LOG: ERP Vendor Master Data: Vendor B created yesterday, sharing bank routing account with purchasing manager.\n"
            "TX: $1.2M procurement purchase order approval request."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this procurement request involve illegal bid rigging and employee kickback fraud?"},
            {
                "type": "choice",
                "instructions": "Corporate compliance action:",
                "options": ["issue_purchase_order", "block_vendor_and_refer_to_internal_investigations", "expedite_vendor_payment"]
            }
        ],
        "expected_action": "block_vendor_and_refer_to_internal_investigations"
    },
    {
        "id": "FRAUD-32",
        "category": "Supply Chain / ERP",
        "name": "Supply Chain Fake Subcontractor Invoice Fraud",
        "payload": (
            "LOG: Invoice Scan: Subcontractor invoice submitted with changed bank wiring details. Vendor address lists vacant lot. Phone number disconnected.\n"
            "TX: $280,000 ACH payment request for raw materials delivery.\n"
            "NATURAL LANGUAGE: Invoice line item: 'Bulk steel supply per contract'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this invoice payment directive a fraudulent fake subcontractor claim?"},
            {
                "type": "choice",
                "instructions": "Accounts payable action:",
                "options": ["release_ach_payment", "hold_payment_and_verify_vendor_authenticity", "apply_early_payment_discount"]
            }
        ],
        "expected_action": "hold_payment_and_verify_vendor_authenticity"
    },
    {
        "id": "FRAUD-33",
        "category": "Cloud Computing Infrastructure",
        "name": "Cloud Account Hijacking & Crypto Mining Abuse",
        "payload": (
            "LOG: AWS CloudTrail API Log: Compromised developer API key used to launch 500 GPU-heavy g5.12xlarge EC2 instances in unused region (ap-south-1). CPU/GPU utilization 100% running Monero mining pool binaries.\n"
            "TX: Cloud infrastructure spend surging $15,000 per hour.\n"
            "NATURAL LANGUAGE: Instance tag: 'test-environment'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this sudden cloud infrastructure spike an unauthorized crypto mining hijack?"},
            {
                "type": "choice",
                "instructions": "DevOps Incident Response action:",
                "options": ["keep_instances_running", "terminate_unauthorized_instances_and_revoke_keys", "upgrade_aws_support_plan"]
            }
        ],
        "expected_action": "terminate_unauthorized_instances_and_revoke_keys"
    },
    {
        "id": "FRAUD-34",
        "category": "Commercial Lending",
        "name": "Commercial Accounts Receivable Invoicing Fraud",
        "payload": (
            "NATURAL LANGUAGE AUDIT: External Auditor Report: Confirmed customer invoices submitted for loan collateral were completely fabricated; listed customers state they never ordered goods.\n"
            "LOG: Bank statement PDF metadata edited using PDF modification software.\n"
            "TX: $5,000,000 revolving credit line draw down request."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this credit draw request based on fraudulent collateral invoices?"},
            {
                "type": "choice",
                "instructions": "Commercial underwriting decision:",
                "options": ["disburse_revolving_line_funds", "freeze_credit_line_and_issue_default_notice", "reduce_interest_rate_margin"]
            }
        ],
        "expected_action": "freeze_credit_line_and_issue_default_notice"
    },

    # ----------------------------------------------------
    # Category F: Healthcare, Insurance & Public Sector
    # ----------------------------------------------------
    {
        "id": "FRAUD-35",
        "category": "Healthcare Fraud",
        "name": "Healthcare Insurance Phantom Claims Billing",
        "payload": (
            "LOG: Medical Billing Audit: Small 500 sq ft retail clinic billed 400 complex MRI scan procedures in 24 hours. Facility contains zero radiology hardware. Patient logs show zero physical visits.\n"
            "TX: $650,000 insurance reimbursement claim batch.\n"
            "NATURAL LANGUAGE: Procedure code: 'Full body MRI diagnostic scan'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this insurance billing represent phantom healthcare fraud?"},
            {
                "type": "choice",
                "instructions": "Claims adjudication recommendation:",
                "options": ["reimburse_claims_in_full", "flag_for_fraud_investigation_and_freeze_payouts", "send_patient_satisfaction_survey"]
            }
        ],
        "expected_action": "flag_for_fraud_investigation_and_freeze_payouts"
    },
    {
        "id": "FRAUD-36",
        "category": "Pharmacy Fraud",
        "name": "Pharmacy Prescription Drug Trafficking Ring",
        "payload": (
            "LOG: E-Prescribing System Audit: Single doctor credentials issued 1,500 high-volume Oxycodone prescriptions across 3 days to patients living 400 miles away.\n"
            "TX: Insurance drug benefit claim processing.\n"
            "NATURAL LANGUAGE: Doctor note: 'Chronic pain management relief'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this prescription activity a fraudulent pill-mill / drug trafficking operation?"},
            {
                "type": "choice",
                "instructions": "Pharmacy benefit manager action:",
                "options": ["fill_all_prescriptions", "block_prescriptions_and_report_to_dea", "offer_generic_brand_discount"]
            }
        ],
        "expected_action": "block_prescriptions_and_report_to_dea"
    },
    {
        "id": "FRAUD-37",
        "category": "Government Grants",
        "name": "Disaster Relief Grant / Small Business PPP Loan Fraud",
        "payload": (
            "NATURAL LANGUAGE: Grant Application: 'Entity employs 45 full-time workers in storm-damaged zone requesting $500,000 emergency grant'.\n"
            "LOG: State Unemployment & IRS Tax Records confirm entity registered 3 days ago with 0 reported employees.\n"
            "TX: $500,000 direct government grant payout request."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this relief application a fraudulent grant claim using false payroll numbers?"},
            {
                "type": "choice",
                "instructions": "Grant administration decision:",
                "options": ["disburse_grant_funds", "deny_application_and_flag_for_criminal_prosecution", "request_promotional_brochure"]
            }
        ],
        "expected_action": "deny_application_and_flag_for_criminal_prosecution"
    },
    {
        "id": "FRAUD-38",
        "category": "Unemployment / Public Sector",
        "name": "Government Benefits / Identity Theft Claims Ring",
        "payload": (
            "LOG: State Labor Portal: 250 unemployment claims filed within 1 hour targeting identity details of active corporate executives. All payout routing numbers point to single prepaid debit card bank.\n"
            "TX: $1,200,000 cumulative benefit payout request.\n"
            "NATURAL LANGUAGE: Claim notes: 'Job loss due to corporate downsizing'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Are these benefit claims part of a mass identity theft unemployment scam?"},
            {
                "type": "choice",
                "instructions": "State portal risk action:",
                "options": ["approve_benefit_disbursements", "freeze_all_claims_and_trigger_id_verify", "mail_paper_checks"]
            }
        ],
        "expected_action": "freeze_all_claims_and_trigger_id_verify"
    },

    # ----------------------------------------------------
    # Category G: Telecom, Logistics, Fleet & Travel
    # ----------------------------------------------------
    {
        "id": "FRAUD-39",
        "category": "Telecom Fraud",
        "name": "Telecom Toll Fraud / International Premium Rate Calling",
        "payload": (
            "LOG: Corporate PBX Telemetry: PBX hacked outside business hours. 4,000 continuous concurrent VoIP calls initiated to high-cost premium rate phone numbers in Somalia.\n"
            "TX: Telecom bill surging $40,000 per hour.\n"
            "NATURAL LANGUAGE: SIP Call Header: 'Outbound trunk routing'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this PBX call spike an active toll fraud exploit?"},
            {
                "type": "choice",
                "instructions": "Telecom gateway decision:",
                "options": ["allow_all_calls", "sever_international_trunk_and_alert_soc", "increase_call_quality_bandwidth"]
            }
        ],
        "expected_action": "sever_international_trunk_and_alert_soc"
    },
    {
        "id": "FRAUD-40",
        "category": "Travel & Loyalty",
        "name": "Travel Miles Theft & Fraudulent Ticket Reselling",
        "payload": (
            "LOG: Airline Loyalty API: 500,000 frequent flyer miles transferred from dormant account to unknown third party. Immediately redeemed for 2 first-class international tickets departing in 2 hours.\n"
            "TX: Flight booking value $14,000.\n"
            "NATURAL LANGUAGE: Passenger details: 'Names completely unrelated to account holder'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this milestone redemption an unauthorized mileage theft cashout?"},
            {
                "type": "choice",
                "instructions": "Airline Security action:",
                "options": ["issue_boarding_passes", "cancel_tickets_and_lock_frequent_flyer_account", "offer_cabin_upgrade"]
            }
        ],
        "expected_action": "cancel_tickets_and_lock_frequent_flyer_account"
    },
    {
        "id": "FRAUD-41",
        "category": "Gig Economy",
        "name": "Gig Economy / Rideshare Phantom Trip Billing",
        "payload": (
            "LOG: Driver Telemetry: Driver logged 40 completed rides in 1 hour. Vehicle telematics GPS shows car parked static in driver driveway. Virtual GPS spoofing app detected on device.\n"
            "TX: $850 fare payout request.\n"
            "NATURAL LANGUAGE: Trip comments: 'Completed ride smoothly'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this driver generating fake GPS trips to steal fare payouts?"},
            {
                "type": "choice",
                "instructions": "Platform Integrity action:",
                "options": ["payout_fare_earnings", "deactivate_driver_and_clawback_funds", "send_driver_bonus"]
            }
        ],
        "expected_action": "deactivate_driver_and_clawback_funds"
    },
    {
        "id": "FRAUD-42",
        "category": "Fleet / Logistics",
        "name": "Vehicle Fleet Fuel Card Cash-Out Fraud",
        "payload": (
            "LOG: Commercial Fleet Telematics: Truck #12 GPS telematics logged location in Ohio. Fleet fuel card swiped 5 times in 10 minutes at gas station in Texas for 500 gallons of diesel per swipe.\n"
            "TX: $3,800 fuel charge authorized.\n"
            "NATURAL LANGUAGE: Driver log: 'Off duty status'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this fuel card charge a fraudulent clone / station cash-out scam?"},
            {
                "type": "choice",
                "instructions": "Fleet card authorization decision:",
                "options": ["approve_fuel_transaction", "decline_card_and_alert_fleet_manager", "apply_cashback_reward"]
            }
        ],
        "expected_action": "decline_card_and_alert_fleet_manager"
    },
    {
        "id": "FRAUD-43",
        "category": "P2P Car Rental",
        "name": "Peer-to-Peer Car Rental Vehicle Theft Ring",
        "payload": (
            "LOG: P2P Car Rental Telemetry: Renter account created with stolen ID. GPS tracker on $90,000 luxury SUV disabled 10 minutes after pickup near international border crossing.\n"
            "TX: 3-day vehicle rental reservation.\n"
            "NATURAL LANGUAGE: Renter message: 'Taking car for weekend road trip'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this rental reservation a vehicle theft / chop-shop diversion?"},
            {
                "type": "choice",
                "instructions": "Asset protection intervention:",
                "options": ["extend_rental_period", "trigger_remote_engine_kill_and_notify_police", "send_satisfaction_survey"]
            }
        ],
        "expected_action": "trigger_remote_engine_kill_and_notify_police"
    },
    {
        "id": "FRAUD-44",
        "category": "Loyalty / Rewards",
        "name": "Digital Wallet Loyalty Bonus Farming",
        "payload": (
            "LOG: App Telemetry: 500 new user accounts registered from single IP subnet in 1 hour. All accounts instantly redeem $20 sign-up bonus credits to buy gift cards sent to single inbox.\n"
            "TX: $10,000 total bonus redemption payout.\n"
            "NATURAL LANGUAGE: Usernames follow pattern 'bot_user_001', 'bot_user_002'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this activity an automated referral bonus farming operation?"},
            {
                "type": "choice",
                "instructions": "Reward engine decision:",
                "options": ["payout_all_bonus_credits", "block_redemptions_and_ban_accounts", "double_reward_points"]
            }
        ],
        "expected_action": "block_redemptions_and_ban_accounts"
    },

    # ----------------------------------------------------
    # Category H: Trade Finance, Real Estate & Utility Fraud
    # ----------------------------------------------------
    {
        "id": "FRAUD-45",
        "category": "Trade Finance",
        "name": "Cross-Border Trade Finance / Double Invoicing Fraud",
        "payload": (
            "LOG: Customs & Port Audit: Same shipping container #MSCU-908123 listed on two separate Letter of Credit (LC) financing applications at two different banks for $2.5M each.\n"
            "TX: $5,000,000 combined trade finance disbursement request.\n"
            "NATURAL LANGUAGE: Bill of Lading desc: 'High-grade copper cathode export'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this trade finance request an illegal double invoicing scheme?"},
            {
                "type": "choice",
                "instructions": "Trade finance bank decision:",
                "options": ["issue_letter_of_credit", "decline_lc_and_report_to_international_maritime_bureau", "waive_customs_fee"]
            }
        ],
        "expected_action": "decline_lc_and_report_to_international_maritime_bureau"
    },
    {
        "id": "FRAUD-46",
        "category": "Real Estate Leasing",
        "name": "Commercial Real Estate Lease Income Forgery",
        "payload": (
            "NATURAL LANGUAGE APPLICATION: Tenant application for $15,000/mo commercial retail space. Listed business revenue $1.8M/yr.\n"
            "LOG: Bank statement PDF submitted shows altered transaction lines and mathematical sum errors on monthly balances.\n"
            "TX: Commercial lease agreement signing."
        ),
        "questions": [
            {"type": "noul", "instructions": "Does this lease application contain forged financial statements?"},
            {
                "type": "choice",
                "instructions": "Landlord underwriting action:",
                "options": ["execute_lease_agreement", "reject_application_and_keep_holding_deposit", "offer_rent_discount"]
            }
        ],
        "expected_action": "reject_application_and_keep_holding_deposit"
    },
    {
        "id": "FRAUD-47",
        "category": "Utility Identity Theft",
        "name": "Utility Bill Identity Theft & Fraudulent Meter Setup",
        "payload": (
            "LOG: Utility Onboarding: New commercial electric meter application submitted under victim identity without consent. Outstanding unpaid debt of $45,000 at prior address under same fraudulent name.\n"
            "TX: High-voltage commercial power activation request.\n"
            "NATURAL LANGUAGE: Applicant note: 'Urgent power turn-on needed'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this utility account setup an identity theft account creation?"},
            {
                "type": "choice",
                "instructions": "Utility service decision:",
                "options": ["connect_power_service", "deny_service_and_flag_identity_theft", "waive_security_deposit"]
            }
        ],
        "expected_action": "deny_service_and_flag_identity_theft"
    },
    {
        "id": "FRAUD-48",
        "category": "Hospitality / Loyalty",
        "name": "Hotel Loyalty Night Farming & Fake Check-In Fraud",
        "payload": (
            "LOG: Hotel PMS System Audit: 100 room reservations checked in digitally via app at 01:00 AM using corporate discount rate code. Smart door lock sensors record 0 physical room entries.\n"
            "TX: Claiming 100 elite status qualification nights.\n"
            "NATURAL LANGUAGE: Guest notes: 'Express mobile check-in'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this digital check-in cluster a fraudulent loyalty night farming exploit?"},
            {
                "type": "choice",
                "instructions": "Hotel loyalty audit decision:",
                "options": ["award_elite_status_nights", "void_stay_credits_and_audit_corporate_code", "upgrade_all_rooms_to_suites"]
            }
        ],
        "expected_action": "void_stay_credits_and_audit_corporate_code"
    },
    {
        "id": "FRAUD-49",
        "category": "Aviation Security",
        "name": "Airline Passenger Identity Swap & Ticket Fraud",
        "payload": (
            "LOG: Airport Gate Scan: Passport MRZ scan mismatch with ticket booking name. Passenger attempts to board international flight using ticket purchased with stolen credit card details of another person.\n"
            "TX: International flight boarding pass scan.\n"
            "NATURAL LANGUAGE: Passenger statement: 'Friend booked ticket for me'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this passenger attempting to board using a fraudulent ticket and identity swap?"},
            {
                "type": "choice",
                "instructions": "Gate security action:",
                "options": ["allow_passenger_to_board", "deny_boarding_and_escort_to_airport_police", "issue_frequent_flyer_miles"]
            }
        ],
        "expected_action": "deny_boarding_and_escort_to_airport_police"
    },
    {
        "id": "FRAUD-50",
        "category": "Mobile Carrier",
        "name": "Real-Time Account Opening / Synthetic Phone Port",
        "payload": (
            "LOG: Mobile Carrier Webhook: Victim mobile number ported to new SIM card at 02:00 AM without authorization. 5 minutes later, perpetrator attempts to open new online bank account using victim's intercepted phone number for 2FA.\n"
            "TX: Instant mobile bank account opening.\n"
            "NATURAL LANGUAGE: Application note: 'New customer onboarding'."
        ),
        "questions": [
            {"type": "noul", "instructions": "Is this account opening driven by an unauthorized SIM porting attack?"},
            {
                "type": "choice",
                "instructions": "Digital banking risk action:",
                "options": ["approve_account_opening", "block_onboarding_and_flag_unauthorized_port", "send_welcome_gift"]
            }
        ],
        "expected_action": "block_onboarding_and_flag_unauthorized_port"
    }
]

print(f"Beginning execution of {len(scenarios)} extensive real-world fraud scenarios...\n")

total_time = 0.0
correct_decisions = 0
category_stats = {}

for idx, s in enumerate(scenarios, 1):
    cat = s["category"]
    if cat not in category_stats:
        category_stats[cat] = {"total": 0, "correct": 0, "total_ms": 0.0}
    category_stats[cat]["total"] += 1

    print("-" * 95)
    print(f"TEST [{idx:02d}/50] [{s['id']}] [{cat}]: {s['name']}")
    print(f"PAYLOAD TELEMETRY:\n{s['payload'][:160]}...\n")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decisions = model.decide(s["payload"], s["questions"])
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_time += elapsed_ms
    category_stats[cat]["total_ms"] += elapsed_ms

    chosen_action = None
    noul_prob = None

    for q, d in zip(s["questions"], decisions):
        q_type = q["type"].upper()
        inst = q["instructions"]
        if q_type == "NOUL":
            noul_prob = d["noul"]
            verdict = "YES" if noul_prob >= 0.5 else "NO"
            print(f"  * Fraud Probability (NOUL):  {noul_prob:.4f} ({verdict}) -> '{inst}'")
        elif q_type == "CHOICE":
            chosen_action = d["choice"]
            conf = d["confidence"] * 100
            print(f"  * Action Decision (CHOICE):  '{chosen_action}' ({conf:.1f}% confidence)")

    is_match = (chosen_action == s["expected_action"])
    if is_match:
        correct_decisions += 1
        category_stats[cat]["correct"] += 1
        print(f"  -> RESULT: [CORRECT] ({elapsed_ms:.2f} ms) - Action matched expected policy '{s['expected_action']}'")
    else:
        print(f"  -> RESULT: [MISMATCH] ({elapsed_ms:.2f} ms) - Chose '{chosen_action}', expected '{s['expected_action']}'")

avg_lat = total_time / len(scenarios)
accuracy = (correct_decisions / len(scenarios)) * 100

print("\n" + "=" * 95)
print(f"50-SCENARIO EXTENSIVE FRAUD BENCHMARK COMPLETE")
print(f"Total Scenarios Evaluated: {len(scenarios)}")
print(f"Policy Decision Accuracy:  {correct_decisions}/{len(scenarios)} ({accuracy:.1f}%)")
print(f"Average Decision Latency:  {avg_lat:.2f} ms per transaction packet")
print(f"Total Evaluation Time:    {total_time/1000:.2f} seconds")
print(f"SLA Compliance Status:     {'PASSED (<100ms per transaction)' if avg_lat < 100 else 'FAILED'}")
print(f"VRAM Memory Allocated:     {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print("=" * 95)

print("\n--- PERFORMANCE BY CATEGORY ---")
for cat, stats in category_stats.items():
    cat_acc = (stats["correct"] / stats["total"]) * 100
    cat_lat = stats["total_ms"] / stats["total"]
    print(f"  - {cat:<32}: Accuracy {stats['correct']}/{stats['total']} ({cat_acc:5.1f}%) | Avg Latency: {cat_lat:5.2f} ms")
print("=" * 95)
