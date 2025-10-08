#!/usr/bin/env python3
"""
Synthetic Data Analysis and Generation for ING Prompt Scrubber ML Training

This script analyzes existing prompt patterns and generates synthetic training data
for the sensitivity classification machine learning model.
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

fake = Faker(["en_US", "nl_NL", "en_GB", 'fr_FR'])  # US, Dutch, UK, French locales for banking context


class EntityPattern:
    """Represents a type of sensitive entity that needs redaction"""

    def __init__(
        self,
        entity_type: str,
        examples: List[str],
        generator_func,
        placeholder_pattern: str,
    ):
        self.entity_type = entity_type
        self.examples = examples
        self.generator_func = generator_func
        self.placeholder_pattern = placeholder_pattern

    def generate_value(self) -> str:
        """Generate a realistic value for this entity type"""
        return self.generator_func()

    def get_placeholder(self, index: int = 0) -> str:
        """Get placeholder for this entity type"""
        if index == 0:
            return f"<{self.entity_type}>"
        return f"<{self.entity_type}_{index+1}>"


class SyntheticDataGenerator:
    """Generates synthetic training data for prompt classification"""

    def __init__(self):
        self.entity_patterns = self._define_entity_patterns()
        self.document_templates = self._define_document_templates()

    def _define_entity_patterns(self) -> Dict[str, EntityPattern]:
        """Define all entity types we want to detect and redact"""

        def generate_iban():
            """Generate realistic European IBAN"""
            country_codes = ["BE", "NL", "DE", "FR", "GB", "ES", "IT"]
            country = random.choice(country_codes)
            if country == "BE":
                return f"BE{random.randint(10,99)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
            elif country == "NL":
                return f"NL{random.randint(10,99)} ABNA {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(10,99)}"
            else:
                return f"{country}{random.randint(10,99)} {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

        def generate_amount():
            """Generate realistic financial amounts"""
            currency = random.choice(["€", "$", "£"])
            amount = random.uniform(100, 999999)
            return f"{currency}{amount:,.2f}"

        def generate_customer_id():
            """Generate customer ID patterns"""
            prefix = random.choice(["CUST", "CLI", "CL", "CUSTOMER"])
            number = random.randint(1, 9999)
            return f"{prefix}-{number:04d}"

        def generate_document_id():
            """Generate document reference IDs"""
            letters = "".join(
                random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8)
            )
            return f"DOC{letters}"

        def generate_phone():
            """Generate phone numbers"""
            return fake.phone_number()

        def generate_email():
            """Generate ING-style email addresses"""
            domains = ["ing.com", "ing.be", "ing.nl", "ing-group.com"]
            first_name = fake.first_name().lower()
            last_name = fake.last_name().lower()
            return f"{first_name}.{last_name}@{random.choice(domains)}"

        def generate_corp_key():
            """Generate corporate key IDs"""
            return f"CK{random.randint(100000, 999999)}"

        def generate_app_id():
            """Generate application IDs"""
            return f"IaaSApp_{random.randint(1, 50)}"

        def generate_transfer_id():
            """Generate transfer/transaction IDs"""
            prefix = random.choice(["TXN", "TRF", "PAY", "WIRE"])
            return f"{prefix}{random.randint(100000000, 999999999)}"

        return {
            "PERSON": EntityPattern(
                "PERSON",
                [
                    "Emily Davis",
                    "Laura Smith",
                    "John Smith",
                    "Michael Clark",
                    "Sarah Johnson",
                ],
                lambda: fake.name(),
                "<PERSON>",
            ),
            "AMOUNT": EntityPattern(
                "AMOUNT",
                ["€162,507.76", "€263,429.48", "€499,667.34"],
                generate_amount,
                "<AMOUNT>",
            ),
            "IBAN": EntityPattern(
                "IBAN",
                ["BE12 3456 7890 1234", "NL91 ABNA 0123 4567 89"],
                generate_iban,
                "<IBAN>",
            ),
            "CUSTOMER_ID": EntityPattern(
                "CUSTOMER_ID",
                ["CUST-0001", "CUST-0002", "CUST-0023"],
                generate_customer_id,
                "<CUSTOMER_ID>",
            ),
            "PHONE_NUMBER": EntityPattern(
                "PHONE_NUMBER",
                ["555-0123", "555-0456", "+32 2 123 45 67"],
                generate_phone,
                "<PHONE>",
            ),
            "EMAIL": EntityPattern(
                "EMAIL", ["user@ing.com", "analyst@ing.be"], generate_email, "<EMAIL>"
            ),
            "YEAR": EntityPattern(
                "YEAR",
                ["2024", "2023", "2019", "2018"],
                lambda: str(random.randint(2018, 2025)),
                "<YEAR>",
            ),
            "URL": EntityPattern(
                "URL",
                [
                    "https://assets.ing.com/m/6e8ace8ade094690/original/Annual-report-2024.pdf"
                ],
                lambda: f"https://assets.ing.com/m/{fake.uuid4().replace('-', '')}/original/{fake.file_name()}",
                "<LINK>",
            ),
            "CORP_KEY": EntityPattern(
                "CORP_KEY", ["CK123456", "CK789012"], generate_corp_key, "<CORPKEY>"
            ),
            "DOCUMENT_REF": EntityPattern(
                "DOCUMENT_REF",
                ["DOC4ZK0HT", "DOCKTE2OF", "DOCL595HW"],
                generate_document_id,
                "<DOC_REF>",
            ),
            "APP_ID": EntityPattern(
                "APP_ID",
                ["IaaSApp_4", "IaaSApp_6", "IaaSApp_14"],
                generate_app_id,
                "<APP>",
            ),
            "TRANSFER_ID": EntityPattern(
                "TRANSFER_ID",
                ["TXN123456789", "TRF987654321"],
                generate_transfer_id,
                "<TRANSFER_ID>",
            ),
        }

    def _define_document_templates(self) -> List[Dict[str, Any]]:
        """Define templates for different types of banking/financial documents"""

        return [
            {
                "category": "financial_reports",
                "templates": [
                    "Write a LinkedIn post announcing the {YEAR} Annual Report. Include the link and 2 bullet points for customers/investors: {URL}.",
                    "Draft an internal newsletter blurb pointing to {YEAR} Pillar 3 Disclosures (Q4) for risk, capital and liquidity metrics: {URL}.",
                    "Prepare an analyst Q&A pack referencing {YEAR} Full Year Results (press room): {URL}.",
                    "Compose an investor FAQ citing {YEAR} Annual Report and {YEAR} Additional Pillar III: {URL} and {URL}.",
                    "Create a press-room index linking {YEAR} Full Year Results and {YEAR} Full Year Results releases.",
                ],
                "entity_types": ["YEAR", "URL", "DOCUMENT_REF"],
            },
            {
                "category": "customer_agreements",
                "templates": [
                    "Prepare a renewal/health check digest for Active customer agreements: {CUSTOMER_ID} ({PERSON} — Credit Card Agreement — {AMOUNT}), {CUSTOMER_ID} ({PERSON} — Insurance Distribution — {AMOUNT}), {CUSTOMER_ID} ({PERSON} — Line of Credit — {AMOUNT}).",
                    "List Terminated Letters of Credit to archive and reconcile: {CUSTOMER_ID} ({PERSON} — {AMOUNT}) and {CUSTOMER_ID} ({PERSON} — {AMOUNT}).",
                    "Review customer portfolio for {PERSON}: Agreement {CUSTOMER_ID}, current balance {AMOUNT}, IBAN {IBAN}.",
                    "Process KYC renewal for {PERSON} (ID: {CUSTOMER_ID}), contact via {EMAIL} or {PHONE_NUMBER}.",
                    "Compliance review needed for high-value customer {PERSON} ({CUSTOMER_ID}) with exposure {AMOUNT}.",
                ],
                "entity_types": [
                    "CUSTOMER_ID",
                    "PERSON",
                    "AMOUNT",
                    "IBAN",
                    "EMAIL",
                    "PHONE_NUMBER",
                ],
            },
            {
                "category": "infrastructure",
                "templates": [
                    "Create a capacity snapshot for Production Running IaaS apps: {APP_ID} (Storage 1795 GB; 8 vCPU/32 GB; Azure), {APP_ID} (1714 GB; 4 vCPU/8 GB; Google Cloud), {APP_ID} (856 GB; 2 vCPU/32 GB; Google Cloud).",
                    "Draft a maintenance window for Test Maintenance IaaS apps: {APP_ID} (Azure), {APP_ID} (AWS), {APP_ID} (Google Cloud). Include patching and agent refresh.",
                    "Write a decommission plan for Production Retired IaaS apps: {APP_ID} (AWS; {DOCUMENT_REF}), {APP_ID} (Azure; {DOCUMENT_REF}).",
                    "Infrastructure capacity review for {APP_ID}: current usage monitoring, contact {PERSON} ({EMAIL}) for approval.",
                    "Security patch deployment for applications {APP_ID}, {APP_ID}, and {APP_ID} scheduled for maintenance.",
                ],
                "entity_types": ["APP_ID", "DOCUMENT_REF", "PERSON", "EMAIL"],
            },
            {
                "category": "payments_transfers",
                "templates": [
                    "Process wire transfer {TRANSFER_ID} from {PERSON} account {IBAN} amount {AMOUNT} to beneficiary {IBAN}.",
                    "Transfer {TRANSFER_ID} completed ({YEAR}), {AMOUNT} → IBAN {IBAN}. Customer {PERSON} notification sent to {EMAIL}.",
                    "Risk Alert: High-value transfer {TRANSFER_ID} — {AMOUNT} → IBAN {IBAN}. Requires dual approval from {PERSON}.",
                    "Reconciliation needed for transfers {TRANSFER_ID}, {TRANSFER_ID}, {TRANSFER_ID} totaling {AMOUNT}.",
                    "IBAN validation failed for {IBAN} in transfer {TRANSFER_ID} from customer {PERSON} ({CUSTOMER_ID}).",
                ],
                "entity_types": [
                    "TRANSFER_ID",
                    "PERSON",
                    "IBAN",
                    "AMOUNT",
                    "EMAIL",
                    "CUSTOMER_ID",
                    "YEAR",
                ],
            },
            {
                "category": "employee_operations",
                "templates": [
                    "On-call rotation for Payments Bridge: Primary: {PERSON} — {PHONE_NUMBER} — {EMAIL} — CorpKey {CORP_KEY}, Backup: {PERSON} — {PHONE_NUMBER} — {EMAIL} — CorpKey {CORP_KEY}.",
                    "INCIDENT P1 — ACK within 10m. If no response: page {PERSON} ({PHONE_NUMBER}) → then {PERSON} ({PHONE_NUMBER}).",
                    "Security Awareness Training reminder for {PERSON} and {PERSON}, tracking via CorpKeys {CORP_KEY}/{CORP_KEY}.",
                    "Create distribution list request: Members {EMAIL}; {EMAIL}; {EMAIL} with internal visibility.",
                    "Employee access review: {PERSON} ({EMAIL}, {CORP_KEY}) requires system access for application {APP_ID}.",
                ],
                "entity_types": [
                    "PERSON",
                    "PHONE_NUMBER",
                    "EMAIL",
                    "CORP_KEY",
                    "APP_ID",
                ],
            },
        ]

    def generate_training_sample(self) -> Dict[str, Any]:
        """Generate a single training sample with original text, redacted text, and entity mappings"""

        # Choose a random document category and template
        doc_category = random.choice(self.document_templates)
        template = random.choice(doc_category["templates"])

        # Generate entity values and create mappings
        entity_mappings = []
        original_text = template
        redacted_text = template

        # Find all entity placeholders in the template
        entity_placeholders = re.findall(r"\{([^}]+)\}", template)
        entity_counts = {}

        for placeholder in entity_placeholders:
            entity_type = placeholder

            # Count occurrences for proper indexing
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            else:
                entity_counts[entity_type] += 1

            # Generate value and create mapping
            if entity_type in self.entity_patterns:
                pattern = self.entity_patterns[entity_type]
                original_value = pattern.generate_value()
                redacted_placeholder = pattern.get_placeholder(
                    entity_counts[entity_type]
                )

                # Replace in texts
                original_text = original_text.replace(
                    f"{{{entity_type}}}", original_value, 1
                )
                redacted_text = redacted_text.replace(
                    f"{{{entity_type}}}", redacted_placeholder, 1
                )

                # Store mapping
                entity_mappings.append(
                    {
                        "entity_type": entity_type,
                        "original": original_value,
                        "placeholder": redacted_placeholder,
                        "start": 0,  # These would be calculated in real implementation
                        "end": len(original_value),
                        "confidence": round(random.uniform(0.85, 0.99), 3),
                    }
                )

        return {
            "original_text": original_text,
            "redacted_text": redacted_text,
            "entity_mappings": entity_mappings,
            "document_category": doc_category["category"],
            "sensitivity_level": self._calculate_sensitivity_level(entity_mappings),
            "timestamp": datetime.now() - timedelta(days=random.randint(0, 365)),
        }

    def _calculate_sensitivity_level(self, entity_mappings: List[Dict]) -> str:
        """Calculate sensitivity level based on entity types present"""
        high_sensitivity_entities = {"IBAN", "AMOUNT", "CUSTOMER_ID", "TRANSFER_ID"}
        medium_sensitivity_entities = {"PERSON", "EMAIL", "PHONE_NUMBER", "CORP_KEY"}

        entity_types = {mapping["entity_type"] for mapping in entity_mappings}

        if entity_types.intersection(high_sensitivity_entities):
            return "HIGH"
        elif entity_types.intersection(medium_sensitivity_entities):
            return "MEDIUM"
        else:
            return "LOW"

    def generate_training_dataset(
        self, num_samples: int = 1000
    ) -> List[Dict[str, Any]]:
        """Generate a complete training dataset"""
        print(f"🤖 Generating {num_samples} synthetic training samples...")

        dataset = []
        for i in range(num_samples):
            sample = self.generate_training_sample()
            dataset.append(sample)

            if (i + 1) % 100 == 0:
                print(f"   ✅ Generated {i + 1} samples...")

        print(f"🎉 Dataset generation completed!")
        return dataset

    def export_to_csv(
        self,
        dataset: List[Dict[str, Any]],
        filename: str = "synthetic_training_data.csv",
    ):
        """Export dataset to CSV for ML training"""

        # Flatten the data for CSV export
        flattened_data = []
        for sample in dataset:
            base_row = {
                "original_text": sample["original_text"],
                "redacted_text": sample["redacted_text"],
                "document_category": sample["document_category"],
                "sensitivity_level": sample["sensitivity_level"],
                "timestamp": sample["timestamp"],
                "num_entities": len(sample["entity_mappings"]),
            }

            # Add entity information
            for i, entity in enumerate(sample["entity_mappings"]):
                base_row.update(
                    {
                        f"entity_{i}_type": entity["entity_type"],
                        f"entity_{i}_original": entity["original"],
                        f"entity_{i}_placeholder": entity["placeholder"],
                        f"entity_{i}_confidence": entity["confidence"],
                    }
                )

            flattened_data.append(base_row)

        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)
        print(f"📊 Dataset exported to {filename}")
        return df

    def export_to_json(
        self,
        dataset: List[Dict[str, Any]],
        filename: str = "synthetic_training_data.json",
    ):
        """Export dataset to JSON format"""

        # Convert datetime objects to strings for JSON serialization
        json_dataset = []
        for sample in dataset:
            json_sample = sample.copy()
            json_sample["timestamp"] = sample["timestamp"].isoformat()
            json_dataset.append(json_sample)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(json_dataset, f, indent=2, ensure_ascii=False)

        print(f"📄 Dataset exported to {filename}")

    def analyze_dataset(self, dataset: List[Dict[str, Any]]):
        """Analyze the generated dataset for quality and distribution"""

        print("\n📈 Dataset Analysis:")
        print(f"   Total samples: {len(dataset)}")

        # Category distribution
        categories = {}
        for sample in dataset:
            cat = sample["document_category"]
            categories[cat] = categories.get(cat, 0) + 1

        print(f"   Document categories:")
        for cat, count in categories.items():
            print(f"     {cat}: {count} ({count/len(dataset)*100:.1f}%)")

        # Sensitivity level distribution
        sensitivity_levels = {}
        for sample in dataset:
            level = sample["sensitivity_level"]
            sensitivity_levels[level] = sensitivity_levels.get(level, 0) + 1

        print(f"   Sensitivity levels:")
        for level, count in sensitivity_levels.items():
            print(f"     {level}: {count} ({count/len(dataset)*100:.1f}%)")

        # Entity type distribution
        entity_types = {}
        for sample in dataset:
            for entity in sample["entity_mappings"]:
                etype = entity["entity_type"]
                entity_types[etype] = entity_types.get(etype, 0) + 1

        print(f"   Entity types (top 10):")
        sorted_entities = sorted(
            entity_types.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for etype, count in sorted_entities:
            print(f"     {etype}: {count}")


def main():
    """Main function to generate synthetic training data"""

    print("🤖 ING Synthetic Data Generator for ML Training")
    print("=" * 50)

    generator = SyntheticDataGenerator()

    # Ask user for dataset size
    try:
        num_samples = int(
            input("How many training samples to generate? (default: 1000): ") or "1000"
        )
    except ValueError:
        num_samples = 1000

    # Generate dataset
    dataset = generator.generate_training_dataset(num_samples)

    # Analyze dataset
    generator.analyze_dataset(dataset)

    # Export options
    print(f"\n💾 Export Options:")
    print(f"1. CSV format (for pandas/sklearn)")
    print(f"2. JSON format (for custom processing)")
    print(f"3. Both formats")

    export_choice = input("Choose export format (1-3): ").strip()

    if export_choice in ["1", "3"]:
        df = generator.export_to_csv(dataset)
        print(f"   CSV shape: {df.shape}")

    if export_choice in ["2", "3"]:
        generator.export_to_json(dataset)

    print(f"\n✨ Synthetic data generation completed!")


if __name__ == "__main__":
    main()
