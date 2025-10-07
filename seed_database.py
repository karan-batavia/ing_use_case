#!/usr/bin/env python3
"""
Database seeding script for ING Use Case - Prompt Scrubber API
Creates realistic audit log entries based on example prompts and responses
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mongodb_service import MongoDBService

# Sample data based on real prompts from the prompts folder
SAMPLE_PROMPTS = [
    {
        "original": "Write a LinkedIn post announcing the 2024 Annual Report. Include the link and 2 bullet points for customers/investors: https://assets.ing.com/m/6e8ace8ade094690/original/Annual-report-2024.pdf.",
        "redacted": "Write a LinkedIn post announcing the <YEAR> <DOC_TYPE>. Include the <LINK> and 2 bullet points for customers/investors.",
        "response": "New: Our 2024 Annual Report is live.\n• Highlights of growth, customer experience, and sustainability progress.\n• Full details here: https://assets.ing.com/m/6e8ace8ade094690/original/Annual-report-2024.pdf",
        "response_redacted": "New: Our <YEAR> <DOC_TYPE> is live.\n• Highlights of growth, CX, and sustainability.\n• Full details: <LINK>",
    },
    {
        "original": "Draft an internal newsletter blurb pointing to 2024 Pillar 3 Disclosures (Q4) for risk, capital and liquidity metrics: https://assets.ing.com/m/5f710bfeff3364d0/original/ING-BE-Pillar-3-Disclosures-Q4-2024.xlsx.",
        "redacted": "Draft an internal newsletter blurb pointing to <YEAR> <DOC_TYPE> for risk/capital/liquidity metrics: <LINK>.",
        "response": "Pillar 3 — Q4 2024: Access core risk, capital and liquidity disclosures here: https://assets.ing.com/m/5f710bfeff3364d0/original/ING-BE-Pillar-3-Disclosures-Q4-2024.xlsx.",
        "response_redacted": "Pillar 3 — <PERIOD> <YEAR>: Access disclosures here: <LINK>.",
    },
    {
        "original": "Prepare a renewal/health check digest for Active customer agreements: CUST-0001 (Emily Davis — Credit Card Agreement — €162,507.76), CUST-0011 (Laura Smith — Insurance Distribution — €263,429.48), CUST-0023 (Emily Williams — Line of Credit — €353,236.74).",
        "redacted": "Prepare a renewal/health check digest for Active customer agreements: <AGREEMENT_1> (<CUSTOMER_1> — <TYPE_1> — <AMOUNT_1>), <AGREEMENT_2> (...), <AGREEMENT_3> (...).",
        "response": "Active Agreements — Renewal/Health Check\n• CUST-0001 — Emily Davis — Credit Card Agreement — €162,507.76\n• CUST-0011 — Laura Smith — Insurance Distribution Agreement — €263,429.48\n• CUST-0023 — Emily Williams — Line of Credit Agreement — €353,236.74\nNext: confirm covenants, pricing, KYC refresh.",
        "response_redacted": "Active Agreements — Renewal/Health Check\n• <AGREEMENT_1> — <CUSTOMER_1> — <TYPE_1> — <AMOUNT_1>\n• <AGREEMENT_2> — <CUSTOMER_2> — <TYPE_2> — <AMOUNT_2>\n• <AGREEMENT_3> — <CUSTOMER_3> — <TYPE_3> — <AMOUNT_3>\nNext: confirm covenants and KYC.",
    },
    {
        "original": "List Terminated Letters of Credit to archive and reconcile: CUST-0002 (Michael Clark — €499,667.34) and CUST-0007 (Chris Garcia — €126,113.91).",
        "redacted": "List Terminated <AGREEMENT_TYPE> to archive and reconcile: <AGREEMENT_ID> (<CUSTOMER> — <AMOUNT>), <AGREEMENT_ID> (...).",
        "response": "Terminated — Letters of Credit\n• CUST-0002 — Michael Clark — €499,667.34\n• CUST-0007 — Chris Garcia — €126,113.91\nAction: archive packs; reconcile collateral.",
        "response_redacted": "Terminated — <AGREEMENT_TYPE>\n• <AGREEMENT_ID> — <CUSTOMER> — <AMOUNT> (x2)\nAction: archive & reconcile.",
    },
    {
        "original": "Create a capacity snapshot for Production Running IaaS apps: IaaSApp_4 (Storage 1795 GB; 8 vCPU/32 GB; Azure), IaaSApp_6 (1714 GB; 4 vCPU/8 GB; Google Cloud), IaaSApp_14 (856 GB; 2 vCPU/32 GB; Google Cloud), IaaSApp_16 (807 GB; 8 vCPU/8 GB; Azure).",
        "redacted": "Create a capacity snapshot for <ENVIRONMENT> <OP_STATUS> IaaS apps: <APP> (Storage <GB> GB; <VCPU_RAM>; <PROVIDER>), ...",
        "response": "Capacity — Prod Running (IaaS)\n• App_4: 1795GB; 8vCPU/32GB; Azure\n• App_6: 1714GB; 4vCPU/8GB; Google Cloud\n• App_14: 856GB; 2vCPU/32GB; Google Cloud\n• App_16: 807GB; 8vCPU/8GB; Azure\nAction: monitor headroom & latency; plan spares.",
        "response_redacted": "Capacity — <ENVIRONMENT> <OP_STATUS> (IaaS)\n• <APP_LIST_WITH_CAPACITY>\nAction: monitor headroom & latency; plan spares.",
    },
    {
        "original": "INCIDENT P1 — ACK within 10m. If no response: page John Smith (555-0123) → then Sarah Johnson (555-0456).",
        "redacted": "INCIDENT P1 — ACK within 10m. If no response: page <EMP1_NAME> (<EMP1_PHONE>) → then <EMP2_NAME> (<EMP2_PHONE>).",
        "response": "P1 escalation confirmed. Primary: John Smith (555-0123), Secondary: Sarah Johnson (555-0456). Response time: 10min max.",
        "response_redacted": "P1 escalation confirmed. Primary: <EMP1_NAME> (<EMP1_PHONE>), Secondary: <EMP2_NAME> (<EMP2_PHONE>). Response time: 10min max.",
    },
]

# Sample user data for realistic seeding
SAMPLE_USERS = [
    {"user_id": "user_001", "username": "analyst1", "role": "analyst"},
    {"user_id": "user_002", "username": "risk_manager", "role": "manager"},
    {"user_id": "user_003", "username": "compliance_officer", "role": "compliance"},
    {"user_id": "user_004", "username": "admin_user", "role": "admin"},
    {"user_id": "user_005", "username": "data_scientist", "role": "analyst"},
]


class DatabaseSeeder:
    def __init__(self):
        self.mongodb_service = MongoDBService()

    async def seed_database(self, num_sessions: int = 20):
        """
        Seed the database with realistic audit log entries

        Args:
            num_sessions: Number of complete workflow sessions to create
        """
        print(f"🌱 Starting database seeding with {num_sessions} sessions...")

        try:
            # Ensure MongoDB connection
            if not self.mongodb_service.is_connected():
                print("❌ MongoDB not connected. Please check connection.")
                return

            sessions_created = 0

            for i in range(num_sessions):
                # Create a complete workflow session
                session_id = str(uuid.uuid4())
                user = random.choice(SAMPLE_USERS)
                prompt_data = random.choice(SAMPLE_PROMPTS)

                # Generate realistic timestamps (last 30 days)
                base_time = datetime.now() - timedelta(days=random.randint(0, 30))
                redact_time = base_time
                predict_time = base_time + timedelta(minutes=random.randint(1, 15))
                descrub_time = predict_time + timedelta(minutes=random.randint(1, 30))

                # 1. Create redaction entry
                redaction_detections = await self._generate_realistic_detections(
                    prompt_data["original"]
                )

                self.mongodb_service.log_interaction(
                    session_id=session_id,
                    user_id=user["user_id"],
                    action="text_redacted",
                    details={
                        "original_text": prompt_data["original"],
                        "redacted_text": prompt_data["redacted"],
                        "detections": redaction_detections,
                        "total_redacted": len(redaction_detections),
                        "endpoint": "/redact",
                    },
                )

                # 2. Create prediction entry (70% of sessions have predictions)
                if random.random() < 0.7:
                    self.mongodb_service.log_interaction(
                        session_id=session_id,
                        user_id=user["user_id"],
                        action="prediction_generated",
                        details={
                            "prompt": prompt_data["redacted"],
                            "response": prompt_data["response_redacted"],
                            "model_used": random.choice(["gemini-pro", "gemini-flash"]),
                            "prompt_length": len(prompt_data["redacted"]),
                            "endpoint": "/predict",
                        },
                    )

                # 3. Create de-scrub entry (30% of sessions are de-scrubbed, admin only)
                if random.random() < 0.3 and user["role"] == "admin":
                    self.mongodb_service.log_interaction(
                        session_id=session_id,
                        user_id=user["user_id"],
                        action="text_de_scrubbed",
                        details={
                            "scrubbed_text": prompt_data["redacted"],
                            "restored_text": prompt_data["original"],
                            "detections_restored": len(redaction_detections),
                            "endpoint": "/de-scrub",
                        },
                    )

                sessions_created += 1
                if sessions_created % 5 == 0:
                    print(f"   ✅ Created {sessions_created} sessions...")

            print(f"🎉 Database seeding completed!")
            print(f"   📊 Total sessions created: {sessions_created}")
            print(f"   👥 Users involved: {len(SAMPLE_USERS)}")
            print(f"   📝 Sample prompt types: {len(SAMPLE_PROMPTS)}")

            # Print summary statistics
            await self._print_seeding_summary()

        except Exception as e:
            print(f"❌ Error during seeding: {str(e)}")
            raise

    async def _generate_realistic_detections(self, text: str) -> List[Dict[str, Any]]:
        """Generate realistic detection data based on the text content"""
        detections = []

        # Common patterns in the sample data
        patterns = [
            {"type": "YEAR", "examples": ["2024", "2023", "2019", "2018"]},
            {
                "type": "URL",
                "examples": ["https://assets.ing.com/...", "http://assets.ing.com/..."],
            },
            {
                "type": "PERSON",
                "examples": [
                    "Emily Davis",
                    "Laura Smith",
                    "John Smith",
                    "Michael Clark",
                ],
            },
            {
                "type": "AMOUNT",
                "examples": ["€162,507.76", "€263,429.48", "€499,667.34"],
            },
            {"type": "PHONE_NUMBER", "examples": ["555-0123", "555-0456", "555-0789"]},
            {"type": "EMAIL", "examples": ["user@ing.com", "analyst@ing.be"]},
            {
                "type": "IBAN",
                "examples": ["BE12 3456 7890 1234", "NL91 ABNA 0123 4567 89"],
            },
            {
                "type": "CUSTOMER_ID",
                "examples": ["CUST-0001", "CUST-0002", "CUST-0023"],
            },
        ]

        # Generate 2-5 realistic detections per text
        num_detections = random.randint(2, 5)
        for i in range(num_detections):
            pattern = random.choice(patterns)
            original_value = random.choice(pattern["examples"])
            placeholder = (
                f"<{pattern['type']}_{i+1}>" if i > 0 else f"<{pattern['type']}>"
            )

            detections.append(
                {
                    "entity_type": pattern["type"],
                    "original": original_value,
                    "placeholder": placeholder,
                    "start": random.randint(10, 100),
                    "end": random.randint(110, 200),
                    "confidence": round(random.uniform(0.85, 0.99), 3),
                }
            )

        return detections

    async def _print_seeding_summary(self):
        """Print summary of seeded data"""
        try:
            # Check if database is connected
            if not self.mongodb_service.is_connected():
                print("   ⚠️  Database not connected, cannot generate summary")
                return

            db = self.mongodb_service.db
            if db is None:
                print("   ⚠️  Database not available, cannot generate summary")
                return

            # Get collection stats
            interactions_collection = db.interactions
            total_interactions = interactions_collection.count_documents({})

            # Count by action type
            redactions = interactions_collection.count_documents(
                {"action": "text_redacted"}
            )
            predictions = interactions_collection.count_documents(
                {"action": "prediction_generated"}
            )
            descrubs = interactions_collection.count_documents(
                {"action": "text_de_scrubbed"}
            )

            # Count unique sessions
            unique_sessions = len(interactions_collection.distinct("session_id"))

            print("\n📈 Seeding Summary:")
            print(f"   Total interactions: {total_interactions}")
            print(f"   Unique sessions: {unique_sessions}")
            print(f"   Redactions: {redactions}")
            print(f"   Predictions: {predictions}")
            print(f"   De-scrubs: {descrubs}")

        except Exception as e:
            print(f"   ⚠️  Could not generate summary: {str(e)}")

    async def clear_existing_data(self):
        """Clear existing interactions data (use with caution!)"""
        print("🗑️  Clearing existing interaction data...")
        try:
            if not self.mongodb_service.is_connected():
                print("   ❌ Database not connected, cannot clear data")
                return

            db = self.mongodb_service.db
            if db is None:
                print("   ❌ Database not available, cannot clear data")
                return

            result = db.interactions.delete_many({})
            print(f"   ✅ Deleted {result.deleted_count} existing interactions")
        except Exception as e:
            print(f"   ❌ Error clearing data: {str(e)}")


async def main():
    """Main seeding function"""
    seeder = DatabaseSeeder()

    # Check for environment variables to run non-interactively
    auto_seed = os.getenv("AUTO_SEED", "false").lower() == "true"
    clear_first = os.getenv("CLEAR_EXISTING", "false").lower() == "true"
    num_sessions = int(os.getenv("SEED_SESSIONS", "25"))

    if auto_seed:
        print("🤖 Running in auto-seed mode...")
        print(f"   📊 Sessions to create: {num_sessions}")
        print(f"   🗑️  Clear existing data: {clear_first}")

        if clear_first:
            await seeder.clear_existing_data()
        await seeder.seed_database(num_sessions=num_sessions)
        print("\n✨ Auto-seeding complete! Database is ready for exploration.")
        return

    # Interactive mode (original behavior)
    print("🌱 ING Use Case - Database Seeder")
    print("This will create realistic audit log entries for testing and exploration.")
    print("\nOptions:")
    print("1. Seed database (keep existing data)")
    print("2. Clear existing data and seed fresh")
    print("3. Just clear existing data")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "2":
        await seeder.clear_existing_data()
        await seeder.seed_database(num_sessions=25)
    elif choice == "1":
        await seeder.seed_database(num_sessions=25)
    elif choice == "3":
        await seeder.clear_existing_data()
    else:
        print("❌ Invalid choice. Exiting.")
        return

    print("\n✨ Seeding complete! You can now explore the audit logs through the API.")


if __name__ == "__main__":
    asyncio.run(main())
