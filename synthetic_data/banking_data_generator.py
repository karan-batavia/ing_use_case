#!/usr/bin/env python3
"""
Banking Synthetic Data Generator for ML Training

Generates realistic banking/financial prompts with sensitive data across all banking contexts:
- Personal banking, corporate banking, investment banking
- Regulatory compliance, risk management, fraud detection
- Digital banking, payments, loans, insurance
- Internal operations, IT security, customer service
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json
import pandas as pd
from faker import Faker
import re
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

fake = Faker(
    ["en_US", "nl_NL", "en_GB", "fr_FR", "de_DE"]
)  # Multi-locale for European banking


class BankingEntityGenerator:
    """Generator for all banking-related sensitive entities"""

    @staticmethod
    def generate_iban():
        """Generate realistic European IBANs"""
        country_codes = [
            "BE",
            "NL",
            "DE",
            "FR",
            "GB",
            "ES",
            "IT",
            "LU",
            "AT",
            "CH",
            "IE",
            "PT",
        ]
        country = random.choice(country_codes)
        check_digits = random.randint(10, 99)

        if country == "BE":
            bank = random.randint(100, 999)
            account = random.randint(1000000, 9999999)
            check = random.randint(10, 99)
            return f"BE{check_digits} {bank:03d} {account:07d} {check:02d}"
        elif country == "NL":
            banks = ["ABNA", "RABO", "INGB", "TRIO", "KNAB"]
            account = random.randint(1000000000, 9999999999)
            return f"NL{check_digits} {random.choice(banks)} {account:010d}"
        elif country == "DE":
            bank = random.randint(10000000, 99999999)
            account = random.randint(1000000000, 9999999999)
            return f"DE{check_digits} {bank:08d} {account:010d}"
        else:
            # Generic European format
            bank = random.randint(1000, 9999)
            account = random.randint(100000000000, 999999999999)
            return f"{country}{check_digits} {bank:04d} {account:012d}"

    @staticmethod
    def generate_bic_swift():
        """Generate BIC/SWIFT codes"""
        bank_codes = [
            "INGB",
            "ABNA",
            "RABO",
            "DEUT",
            "BNPP",
            "CITI",
            "HSBC",
            "BARC",
            "CRLY",
            "UBSW",
        ]
        country_codes = ["NL", "BE", "DE", "FR", "GB", "US", "CH", "LU"]
        location_codes = ["2A", "2X", "33", "XX", "AA", "44", "BB"]
        branch_codes = ["", "001", "002", "XXX", "100", "200"]

        return f"{random.choice(bank_codes)}{random.choice(country_codes)}{random.choice(location_codes)}{random.choice(branch_codes)}"

    @staticmethod
    def generate_credit_card():
        """Generate realistic credit card numbers (Luhn-valid test numbers)"""
        # Using well-known test card prefixes
        card_types = {
            "Visa": ["4111", "4532", "4000"],
            "Mastercard": ["5555", "5105", "2223"],
            "Amex": ["3714", "3787", "3400"],
            "Discover": ["6011", "6222"],
        }

        card_type = random.choice(list(card_types.keys()))
        prefix = random.choice(card_types[card_type])

        if card_type == "Amex":
            # Amex is 15 digits
            suffix = "".join([str(random.randint(0, 9)) for _ in range(11)])
            return f"{prefix} {suffix[:6]} {suffix[6:11]}"
        else:
            # Others are 16 digits
            suffix = "".join([str(random.randint(0, 9)) for _ in range(12)])
            return f"{prefix} {suffix[:4]} {suffix[4:8]} {suffix[8:12]}"

    @staticmethod
    def generate_account_number():
        """Generate various bank account number formats"""
        formats = [
            f"{random.randint(100000000, 999999999)}",  # 9 digits
            f"{random.randint(1000000000, 9999999999)}",  # 10 digits
            f"{random.randint(10000000, 99999999)}-{random.randint(100, 999)}",  # With check digit
            f"ACC{random.randint(100000, 999999)}",  # Prefixed
            f"{random.randint(1000, 9999)}-{random.randint(100000, 999999)}",  # Branch-account
        ]
        return random.choice(formats)

    @staticmethod
    def generate_amount():
        """Generate financial amounts across all scales"""
        # Different amount ranges for different contexts
        amount_types = [
            (1, 100),  # Small transactions
            (100, 1000),  # Medium transactions
            (1000, 50000),  # Large personal transactions
            (50000, 1000000),  # Corporate transactions
            (1000000, 100000000),  # Large corporate/institutional
            (0.01, 10),  # Micro-payments
        ]

        min_amt, max_amt = random.choice(amount_types)
        amount = random.uniform(min_amt, max_amt)

        # Various currency formats
        currencies = [
            ("€", True),  # Euro with symbol before
            ("$", True),  # Dollar with symbol before
            ("£", True),  # Pound with symbol before
            ("CHF ", False),  # Swiss Franc with code after
            ("SEK ", False),  # Swedish Krona with code after
            ("DKK ", False),  # Danish Krone with code after
            ("NOK ", False),  # Norwegian Krone with code after
        ]

        currency, symbol_before = random.choice(currencies)

        if symbol_before:
            return f"{currency}{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency.strip()}"

    @staticmethod
    def generate_tax_id():
        """Generate tax/social security numbers"""
        formats = [
            f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",  # US SSN
            f"{random.randint(100000000, 999999999)}",  # 9-digit format
            f"NL{random.randint(100000000, 999999999)}B{random.randint(10, 99)}",  # Dutch VAT
            f"BE{random.randint(1000000000, 9999999999)}",  # Belgian VAT
            f"DE{random.randint(100000000, 999999999)}",  # German VAT
            f"FR{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100000000, 999999999)}",  # French VAT
        ]
        return random.choice(formats)

    @staticmethod
    def generate_customer_id():
        """Generate customer identification numbers"""
        prefixes = [
            "CUST",
            "CLI",
            "CL",
            "CUSTOMER",
            "ID",
            "KLT",
            "REL",
            "PARTY",
            "PID",
            "CID",
            "PERS",
            "CORP",
            "ENTITY",
            "ACCOUNT_HOLDER",
            "CLIENT",
        ]

        formats = [
            f"{random.choice(prefixes)}-{random.randint(1, 999999):06d}",
            f"{random.choice(prefixes)}{random.randint(100000, 9999999)}",
            f"{random.choice(prefixes)}_{random.randint(1000, 99999)}",
            f"{random.randint(10000000, 99999999)}",  # Pure numeric
        ]
        return random.choice(formats)


class BankingDataGenerator:
    """Generates synthetic data covering all aspects of banking operations"""

    def __init__(self):
        self.entity_generator = BankingEntityGenerator()
        self.entity_patterns = self._define_entities()
        self.document_templates = self._define_templates()

    def _define_entities(self):
        """Define all possible banking-related sensitive entities"""

        return {
            # CRITICAL FINANCIAL IDENTIFIERS
            "IBAN": {
                "generator": self.entity_generator.generate_iban,
                "sensitivity": "CRITICAL",
            },
            "ACCOUNT_NUMBER": {
                "generator": self.entity_generator.generate_account_number,
                "sensitivity": "CRITICAL",
            },
            "CREDIT_CARD": {
                "generator": self.entity_generator.generate_credit_card,
                "sensitivity": "CRITICAL",
            },
            "DEBIT_CARD": {
                "generator": self.entity_generator.generate_credit_card,
                "sensitivity": "CRITICAL",
            },
            "BIC_SWIFT": {
                "generator": self.entity_generator.generate_bic_swift,
                "sensitivity": "CRITICAL",
            },
            "ROUTING_NUMBER": {
                "generator": lambda: f"{random.randint(100000000, 999999999)}",
                "sensitivity": "CRITICAL",
            },
            "SORT_CODE": {
                "generator": lambda: f"{random.randint(10, 99)}-{random.randint(10, 99)}-{random.randint(10, 99)}",
                "sensitivity": "CRITICAL",
            },
            # TRANSACTION & PAYMENT DATA
            "AMOUNT": {
                "generator": self.entity_generator.generate_amount,
                "sensitivity": "HIGH",
            },
            "TRANSACTION_ID": {
                "generator": lambda: f"TXN{random.randint(100000000000, 999999999999)}",
                "sensitivity": "HIGH",
            },
            "PAYMENT_REF": {
                "generator": lambda: f"PAY-{random.randint(100000, 999999)}-{random.randint(1000, 9999)}",
                "sensitivity": "HIGH",
            },
            "WIRE_REF": {
                "generator": lambda: f"WIRE{random.randint(10000000, 99999999)}",
                "sensitivity": "HIGH",
            },
            "SEPA_REF": {
                "generator": lambda: f"SEPA{random.randint(100000000, 999999999)}",
                "sensitivity": "HIGH",
            },
            "MANDATE_ID": {
                "generator": lambda: f"MNDT{random.randint(100000, 999999)}",
                "sensitivity": "HIGH",
            },
            # LOAN & CREDIT DATA
            "LOAN_ID": {
                "generator": lambda: f"LOAN{random.randint(100000000, 999999999)}",
                "sensitivity": "HIGH",
            },
            "MORTGAGE_ID": {
                "generator": lambda: f"MTG{random.randint(10000000, 99999999)}",
                "sensitivity": "HIGH",
            },
            "CREDIT_LIMIT": {
                "generator": self.entity_generator.generate_amount,
                "sensitivity": "HIGH",
            },
            "INTEREST_RATE": {
                "generator": lambda: f"{random.uniform(0.1, 15.0):.2f}%",
                "sensitivity": "MEDIUM",
            },
            "CREDIT_SCORE": {
                "generator": lambda: str(random.randint(300, 850)),
                "sensitivity": "HIGH",
            },
            "LOAN_BALANCE": {
                "generator": self.entity_generator.generate_amount,
                "sensitivity": "HIGH",
            },
            # PERSONAL IDENTIFIABLE INFORMATION
            "PERSON": {"generator": lambda: fake.name(), "sensitivity": "HIGH"},
            "CUSTOMER_ID": {
                "generator": self.entity_generator.generate_customer_id,
                "sensitivity": "HIGH",
            },
            "TAX_ID": {
                "generator": self.entity_generator.generate_tax_id,
                "sensitivity": "CRITICAL",
            },
            "SSN": {
                "generator": lambda: f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
                "sensitivity": "CRITICAL",
            },
            "PASSPORT": {
                "generator": lambda: f"P{random.randint(10000000, 99999999)}",
                "sensitivity": "HIGH",
            },
            "DRIVING_LICENSE": {
                "generator": lambda: f"DL{random.randint(10000000, 99999999)}",
                "sensitivity": "HIGH",
            },
            "NATIONAL_ID": {
                "generator": lambda: f"ID{random.randint(100000000, 999999999)}",
                "sensitivity": "HIGH",
            },
            # CONTACT INFORMATION
            "EMAIL": {"generator": lambda: fake.email(), "sensitivity": "MEDIUM"},
            "PHONE_NUMBER": {
                "generator": lambda: fake.phone_number(),
                "sensitivity": "MEDIUM",
            },
            "ADDRESS": {
                "generator": lambda: fake.address().replace("\n", ", "),
                "sensitivity": "MEDIUM",
            },
            "POSTAL_CODE": {
                "generator": lambda: fake.postcode(),
                "sensitivity": "MEDIUM",
            },
            # EMPLOYMENT & INCOME DATA
            "EMPLOYER": {"generator": lambda: fake.company(), "sensitivity": "MEDIUM"},
            "SALARY": {
                "generator": lambda: f"€{random.randint(25000, 200000):,} annually",
                "sensitivity": "HIGH",
            },
            "INCOME": {
                "generator": lambda: f"€{random.randint(1000, 20000):,}/month",
                "sensitivity": "HIGH",
            },
            "OCCUPATION": {"generator": lambda: fake.job(), "sensitivity": "MEDIUM"},
            # INSURANCE DATA
            "POLICY_NUMBER": {
                "generator": lambda: f"POL{random.randint(100000000, 999999999)}",
                "sensitivity": "HIGH",
            },
            "CLAIM_ID": {
                "generator": lambda: f"CLM{random.randint(1000000, 9999999)}",
                "sensitivity": "HIGH",
            },
            "PREMIUM": {
                "generator": self.entity_generator.generate_amount,
                "sensitivity": "MEDIUM",
            },
            # INVESTMENT & PORTFOLIO DATA
            "PORTFOLIO_ID": {
                "generator": lambda: f"PF{random.randint(100000, 999999)}",
                "sensitivity": "HIGH",
            },
            "INVESTMENT_AMOUNT": {
                "generator": self.entity_generator.generate_amount,
                "sensitivity": "HIGH",
            },
            "ISIN": {
                "generator": lambda: f"{random.choice(['US', 'NL', 'DE', 'FR'])}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' * 2)[:10]}",
                "sensitivity": "MEDIUM",
            },
            "TICKER": {
                "generator": lambda: "".join(
                    random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(3, 5))
                ),
                "sensitivity": "LOW",
            },
            # BUSINESS/CORPORATE DATA
            "COMPANY_REG": {
                "generator": lambda: f"KVK{random.randint(10000000, 99999999)}",
                "sensitivity": "MEDIUM",
            },
            "VAT_NUMBER": {
                "generator": self.entity_generator.generate_tax_id,
                "sensitivity": "HIGH",
            },
            "DUNS_NUMBER": {
                "generator": lambda: f"{random.randint(100000000, 999999999)}",
                "sensitivity": "MEDIUM",
            },
            "LEI_CODE": {
                "generator": lambda: "".join(
                    random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=20)
                ),
                "sensitivity": "MEDIUM",
            },
            # TECHNICAL/SYSTEM DATA
            "IP_ADDRESS": {"generator": lambda: fake.ipv4(), "sensitivity": "MEDIUM"},
            "MAC_ADDRESS": {
                "generator": lambda: fake.mac_address(),
                "sensitivity": "MEDIUM",
            },
            "SESSION_ID": {"generator": lambda: fake.uuid4(), "sensitivity": "LOW"},
            "API_KEY": {
                "generator": lambda: fake.uuid4().replace("-", "")[:32],
                "sensitivity": "HIGH",
            },
            "TOKEN": {
                "generator": lambda: fake.uuid4().replace("-", ""),
                "sensitivity": "HIGH",
            },
            # REGULATORY & COMPLIANCE
            "FATCA_ID": {
                "generator": lambda: f"FATCA{random.randint(100000, 999999)}",
                "sensitivity": "HIGH",
            },
            "CRS_ID": {
                "generator": lambda: f"CRS{random.randint(1000000, 9999999)}",
                "sensitivity": "HIGH",
            },
            "AML_CASE": {
                "generator": lambda: f"AML{random.randint(100000, 999999)}",
                "sensitivity": "HIGH",
            },
            "SAR_ID": {
                "generator": lambda: f"SAR{random.randint(1000000, 9999999)}",
                "sensitivity": "HIGH",
            },
            # TEMPORAL DATA
            "DATE": {
                "generator": lambda: fake.date_between(
                    start_date="-5y", end_date="today"
                ).strftime("%Y-%m-%d"),
                "sensitivity": "LOW",
            },
            "DATE_OF_BIRTH": {
                "generator": lambda: fake.date_of_birth(
                    minimum_age=18, maximum_age=80
                ).strftime("%Y-%m-%d"),
                "sensitivity": "HIGH",
            },
            "EXPIRY_DATE": {
                "generator": lambda: fake.date_between(
                    start_date="today", end_date="+5y"
                ).strftime("%m/%y"),
                "sensitivity": "MEDIUM",
            },
            # REFERENCE DATA
            "DOCUMENT_REF": {
                "generator": lambda: f"DOC{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))}",
                "sensitivity": "LOW",
            },
            "URL": {"generator": lambda: fake.url(), "sensitivity": "LOW"},
            "FILE_PATH": {
                "generator": lambda: f"/secure/{fake.file_path()}",
                "sensitivity": "MEDIUM",
            },
        }

    def _define_templates(self):
        """Define templates covering all banking business areas"""

        return {
            # RETAIL BANKING
            "account_opening": [
                "Open new savings account for {PERSON}, address {ADDRESS}, phone {PHONE_NUMBER}, initial deposit {AMOUNT}",
                "Create checking account {ACCOUNT_NUMBER} for customer {CUSTOMER_ID} ({PERSON}), SSN {SSN}",
                "New account application: {PERSON}, DOB {DATE_OF_BIRTH}, employer {EMPLOYER}, salary {SALARY}",
                "Customer {CUSTOMER_ID} opening joint account with {PERSON}, both residing at {ADDRESS}",
            ],
            "loan_processing": [
                "Mortgage application {MORTGAGE_ID}: {PERSON}, property value {AMOUNT}, loan amount {LOAN_BALANCE}",
                "Personal loan {LOAN_ID} approved for {CUSTOMER_ID}, amount {AMOUNT}, rate {INTEREST_RATE}",
                "Credit limit increase for {PERSON}: current limit {CREDIT_LIMIT}, credit score {CREDIT_SCORE}",
                "Auto loan disbursement {LOAN_ID}: {AMOUNT} to dealer account {ACCOUNT_NUMBER}",
            ],
            "payments_transfers": [
                "Wire transfer {WIRE_REF}: {AMOUNT} from {IBAN} to {IBAN}, beneficiary {PERSON}",
                "SEPA payment {SEPA_REF}: {CUSTOMER_ID} sending {AMOUNT} to IBAN {IBAN}",
                "International transfer {TRANSACTION_ID}: {AMOUNT} via SWIFT {BIC_SWIFT}",
                "Standing order setup: {AMOUNT} monthly from {ACCOUNT_NUMBER} to {IBAN}",
            ],
            # CORPORATE BANKING
            "corporate_accounts": [
                "Business account opening: {EMPLOYER}, VAT {VAT_NUMBER}, authorized signers {PERSON} and {PERSON}",
                "Corporate credit facility {LOAN_ID}: {EMPLOYER}, limit {CREDIT_LIMIT}, rate {INTEREST_RATE}",
                "Treasury management setup for {EMPLOYER}, contact {PERSON} ({EMAIL}), phone {PHONE_NUMBER}",
                "Multi-currency account {ACCOUNT_NUMBER}: {EMPLOYER}, trading in EUR, USD, GBP",
            ],
            "trade_finance": [
                "Letter of credit {DOCUMENT_REF}: {EMPLOYER} importing goods worth {AMOUNT}",
                "Export financing {LOAN_ID}: {EMPLOYER} shipping to {ADDRESS}, amount {AMOUNT}",
                "Bank guarantee {DOCUMENT_REF}: {EMPLOYER}, beneficiary {EMPLOYER}, amount {AMOUNT}",
                "Documentary collection {TRANSACTION_ID}: {AMOUNT} payment against documents",
            ],
            # INVESTMENT BANKING
            "securities_trading": [
                "Portfolio {PORTFOLIO_ID}: client {PERSON} purchased {AMOUNT} in {TICKER} (ISIN: {ISIN})",
                "Bond trading: {CUSTOMER_ID} selling {AMOUNT} government bonds, settlement {DATE}",
                "Equity transaction {TRANSACTION_ID}: {AMOUNT} shares of {TICKER} for account {PORTFOLIO_ID}",
                "Derivatives trade: {CUSTOMER_ID} options contract {AMOUNT}, expiry {EXPIRY_DATE}",
            ],
            "wealth_management": [
                "Private banking client {PERSON}: portfolio value {INVESTMENT_AMOUNT}, risk profile moderate",
                "Trust account {ACCOUNT_NUMBER}: beneficiary {PERSON}, trustee {PERSON}, assets {AMOUNT}",
                "Estate planning: {PERSON} transferring {AMOUNT} to account {ACCOUNT_NUMBER}",
                "Investment advisory: {CUSTOMER_ID} allocating {AMOUNT} across asset classes",
            ],
            # RISK & COMPLIANCE
            "fraud_detection": [
                "Suspicious transaction {TRANSACTION_ID}: {AMOUNT} from IP {IP_ADDRESS}, customer {CUSTOMER_ID}",
                "Card fraud alert: {CREDIT_CARD} used at unusual location, amount {AMOUNT}",
                "AML case {AML_CASE}: {PERSON} multiple cash deposits totaling {AMOUNT}",
                "Identity verification failed: {CUSTOMER_ID} using document {PASSPORT}",
            ],
            "regulatory_reporting": [
                "FATCA reporting {FATCA_ID}: US person {PERSON}, account balance {AMOUNT}",
                "CRS report {CRS_ID}: {PERSON} tax resident in multiple jurisdictions",
                "SAR filing {SAR_ID}: suspicious activity by {CUSTOMER_ID}, amount {AMOUNT}",
                "Large transaction report: {PERSON} cash deposit {AMOUNT} on {DATE}",
            ],
            "credit_risk": [
                "Credit review {CUSTOMER_ID}: current exposure {AMOUNT}, credit score {CREDIT_SCORE}",
                "Default notice: {PERSON} loan {LOAN_ID} overdue, balance {LOAN_BALANCE}",
                "Collateral valuation: property at {ADDRESS} securing loan {MORTGAGE_ID}",
                "Risk rating downgrade: {EMPLOYER} from A+ to BBB, exposure {AMOUNT}",
            ],
            # DIGITAL BANKING
            "mobile_banking": [
                "Mobile login: {CUSTOMER_ID} from device {MAC_ADDRESS}, IP {IP_ADDRESS}",
                "App transaction: {PERSON} transferred {AMOUNT} using token {TOKEN}",
                "Digital onboarding: {PERSON} opened account via app, verification {DOCUMENT_REF}",
                "Biometric authentication: {CUSTOMER_ID} enrolled fingerprint for account {ACCOUNT_NUMBER}",
            ],
            "api_banking": [
                "API transaction {TRANSACTION_ID}: third-party app accessed account {ACCOUNT_NUMBER}",
                "Open banking consent: {CUSTOMER_ID} authorized data sharing via API key {API_KEY}",
                "Payment initiation: external service transferred {AMOUNT} from {IBAN}",
                "Account aggregation: {PERSON} connected external account {ACCOUNT_NUMBER}",
            ],
            # INSURANCE
            "life_insurance": [
                "Life policy {POLICY_NUMBER}: {PERSON}, beneficiary {PERSON}, coverage {AMOUNT}",
                "Premium payment: policy {POLICY_NUMBER}, amount {PREMIUM}, from account {ACCOUNT_NUMBER}",
                "Claim processing {CLAIM_ID}: {PERSON} deceased, payout {AMOUNT} to {PERSON}",
                "Policy loan: {CUSTOMER_ID} borrowing {AMOUNT} against policy {POLICY_NUMBER}",
            ],
            "property_insurance": [
                "Home insurance {POLICY_NUMBER}: {PERSON}, property {ADDRESS}, coverage {AMOUNT}",
                "Claim settlement {CLAIM_ID}: storm damage at {ADDRESS}, payout {AMOUNT}",
                "Auto insurance: vehicle insured for {AMOUNT}, policy {POLICY_NUMBER}",
                "Commercial property: {EMPLOYER} insuring facility at {ADDRESS} for {AMOUNT}",
            ],
            # OPERATIONS & IT
            "customer_service": [
                "Service request: {CUSTOMER_ID} ({PERSON}) requesting card replacement, phone {PHONE_NUMBER}",
                "Account inquiry: {PERSON} checking balance via phone, verified with SSN {SSN}",
                "Dispute resolution: {CUSTOMER_ID} challenging transaction {TRANSACTION_ID} amount {AMOUNT}",
                "Password reset: {CUSTOMER_ID} requested new credentials, verified via {EMAIL}",
            ],
            "system_operations": [
                "System maintenance: payment system offline, affecting IBAN {IBAN} transfers",
                "Data backup: customer records including {CUSTOMER_ID} archived to {FILE_PATH}",
                "Security incident: unauthorized access attempt from IP {IP_ADDRESS}",
                "Database update: customer {PERSON} address changed to {ADDRESS}",
            ],
            # REGULATORY SCENARIOS
            "audit_scenarios": [
                "Internal audit: reviewing loan {LOAN_ID} documentation for {PERSON}",
                "Regulatory examination: providing transaction data for customer {CUSTOMER_ID}",
                "Compliance review: {PERSON} PEP status verification, account {ACCOUNT_NUMBER}",
                "GDPR request: {PERSON} requesting data deletion, customer ID {CUSTOMER_ID}",
            ],
        }

    def generate_sample(self) -> Dict[str, Any]:
        """Generate a single banking training sample"""

        # Choose random category and template
        category = random.choice(list(self.document_templates.keys()))
        template = random.choice(self.document_templates[category])

        # Find entity placeholders
        placeholders = re.findall(r"\{([^}]+)\}", template)

        original_text = template
        redacted_text = template
        entity_mappings = []

        # Replace each placeholder with generated data
        for placeholder in placeholders:
            if placeholder in self.entity_patterns:
                entity_info = self.entity_patterns[placeholder]
                original_value = entity_info["generator"]()
                redacted_placeholder = f"<{placeholder}>"

                # Replace in texts
                original_text = original_text.replace(
                    f"{{{placeholder}}}", original_value, 1
                )
                redacted_text = redacted_text.replace(
                    f"{{{placeholder}}}", redacted_placeholder, 1
                )

                # Store mapping
                entity_mappings.append(
                    {
                        "entity_type": placeholder,
                        "original": original_value,
                        "placeholder": redacted_placeholder,
                        "sensitivity": entity_info["sensitivity"],
                        "confidence": round(random.uniform(0.85, 0.99), 3),
                    }
                )

        # Calculate overall sensitivity
        sensitivity_levels = [mapping["sensitivity"] for mapping in entity_mappings]
        if "CRITICAL" in sensitivity_levels:
            overall_sensitivity = "CRITICAL"
        elif "HIGH" in sensitivity_levels:
            overall_sensitivity = "HIGH"
        elif "MEDIUM" in sensitivity_levels:
            overall_sensitivity = "MEDIUM"
        else:
            overall_sensitivity = "LOW"

        return {
            "original_text": original_text,
            "redacted_text": redacted_text,
            "entity_mappings": entity_mappings,
            "business_category": category,
            "sensitivity_level": overall_sensitivity,
            "entity_count": len(entity_mappings),
            "timestamp": datetime.now() - timedelta(days=random.randint(0, 365)),
        }

    def generate_dataset(self, num_samples: int = 10000) -> List[Dict[str, Any]]:
        """Generate banking dataset"""
        print(f"🏦 Generating {num_samples} banking samples...")

        dataset = []
        for i in range(num_samples):
            sample = self.generate_sample()
            dataset.append(sample)

            if (i + 1) % 1000 == 0:
                print(f"   ✅ Generated {i + 1} samples...")

        return dataset

    def analyze_dataset(self, dataset: List[Dict[str, Any]]):
        """Analyze the banking dataset"""
        print(f"\n📊 Banking Dataset Analysis:")
        print(f"   Total samples: {len(dataset)}")

        # Business category distribution
        categories = {}
        for sample in dataset:
            cat = sample["business_category"]
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n   Business Categories:")
        for cat, count in sorted(categories.items()):
            print(f"     {cat}: {count} ({count/len(dataset)*100:.1f}%)")

        # Sensitivity distribution
        sensitivity_dist = {}
        for sample in dataset:
            level = sample["sensitivity_level"]
            sensitivity_dist[level] = sensitivity_dist.get(level, 0) + 1

        print(f"\n   Sensitivity Distribution:")
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = sensitivity_dist.get(level, 0)
            print(f"     {level}: {count} ({count/len(dataset)*100:.1f}%)")

        # Entity type analysis
        entity_types = {}
        for sample in dataset:
            for entity in sample["entity_mappings"]:
                etype = entity["entity_type"]
                entity_types[etype] = entity_types.get(etype, 0) + 1

        print(f"\n   Top 15 Entity Types:")
        sorted_entities = sorted(
            entity_types.items(), key=lambda x: x[1], reverse=True
        )[:15]
        for etype, count in sorted_entities:
            print(f"     {etype}: {count}")


def main():
    """Generate banking training data"""
    print("🏦 Banking Synthetic Data Generator")
    print("=" * 60)
    print("Covering: Retail, Corporate, Investment Banking, Insurance, Compliance")

    generator = BankingDataGenerator()

    try:
        num_samples = int(
            input("Number of samples to generate (default: 10000): ") or "10000"
        )
    except ValueError:
        num_samples = 10000

    # Generate dataset
    dataset = generator.generate_dataset(num_samples)

    # Analyze
    generator.analyze_dataset(dataset)

    # Export
    print(f"\n💾 Export Options:")
    print(f"1. CSV format")
    print(f"2. JSON format")
    print(f"3. Both formats")

    choice = input("Choose export format (1-3): ").strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if choice in ["1", "3"]:
        # Flatten for CSV
        flattened = []
        for sample in dataset:
            base_row = {
                "original_text": sample["original_text"],
                "redacted_text": sample["redacted_text"],
                "business_category": sample["business_category"],
                "sensitivity_level": sample["sensitivity_level"],
                "entity_count": sample["entity_count"],
                "timestamp": sample["timestamp"],
            }

            # Add entity details
            for i, entity in enumerate(sample["entity_mappings"]):
                base_row.update(
                    {
                        f"entity_{i}_type": entity["entity_type"],
                        f"entity_{i}_sensitivity": entity["sensitivity"],
                        f"entity_{i}_confidence": entity["confidence"],
                    }
                )

            flattened.append(base_row)

        df = pd.DataFrame(flattened)
        csv_filename = f"banking_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"📊 CSV exported: {csv_filename}")

    if choice in ["2", "3"]:
        # JSON export
        json_dataset = []
        for sample in dataset:
            json_sample = sample.copy()
            json_sample["timestamp"] = sample["timestamp"].isoformat()
            json_dataset.append(json_sample)

        json_filename = f"banking_data_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(json_dataset, f, indent=2)
        print(f"📄 JSON exported: {json_filename}")

    print(f"\n✨ Generation complete!")
    print(f"Dataset covers {len(generator.document_templates)} business areas")
    print(f"with {len(generator.entity_patterns)} entity types")


if __name__ == "__main__":
    main()
