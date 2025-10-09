#!/usr/bin/env python3
"""
Training Script for Enhanced Banking Redaction Model
Uses synthetic data to train NER + classification hybrid model
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our synthetic data generators and enhanced model
try:
    from synthetic_data.generate_synthetic_data import SyntheticDataGenerator
    from synthetic_data.banking_data_generator import BankingDataGenerator
    from src.enhanced_redaction_model_clean import (
        EnhancedRedactionModel,
        create_enhanced_redaction_model,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedModelTrainer:
    """Trainer for the enhanced redaction model"""

    def __init__(self):
        self.model = create_enhanced_redaction_model()
        self.data_generator = SyntheticDataGenerator()
        self.banking_generator = BankingDataGenerator()
        self.training_data = []
        self.metrics = {}

    def load_existing_data(
        self, data_dir: str = "data/classification"
    ) -> List[Dict[str, Any]]:
        """Load existing CSV data and convert to training format"""
        data_path = Path(data_dir)
        training_samples = []

        if not data_path.exists():
            logger.warning(f"Data directory {data_path} not found")
            return []

        # Load CSV files and create training samples
        csv_files = list(data_path.glob("*.csv"))

        for csv_file in csv_files:
            try:
                # Determine sensitivity level from filename
                filename = csv_file.name.lower()
                if "c1" in filename:
                    sensitivity = "C1"
                elif "c2" in filename:
                    sensitivity = "C2"
                elif "c3" in filename:
                    sensitivity = "C3"
                elif "c4" in filename:
                    sensitivity = "C4"
                else:
                    sensitivity = "C2"  # Default

                logger.info(f"Processing {csv_file.name} as {sensitivity}")

                # Read CSV with different separators
                try:
                    df = pd.read_csv(csv_file, sep=";")
                except:
                    try:
                        df = pd.read_csv(csv_file, sep=",")
                    except:
                        logger.error(f"Could not read {csv_file}")
                        continue

                # Convert each row to text for training
                for idx, row in df.iterrows():
                    # Create text representation of the row
                    text_parts = []
                    for col, val in row.items():
                        if pd.notna(val) and str(val).strip():
                            text_parts.append(f"{col}: {val}")

                    if text_parts:
                        original_text = " | ".join(text_parts)

                        training_sample = {
                            "original_text": original_text,
                            "sensitivity_level": sensitivity,
                            "source": csv_file.name,
                            "document_category": self._infer_category(csv_file.name),
                        }
                        training_samples.append(training_sample)

                        # Limit samples per file to prevent imbalance
                        if (
                            len(
                                [
                                    s
                                    for s in training_samples
                                    if s["source"] == csv_file.name
                                ]
                            )
                            >= 50
                        ):
                            break

            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")

        logger.info(f"Loaded {len(training_samples)} samples from existing data")
        return training_samples

    def _infer_category(self, filename: str) -> str:
        """Infer document category from filename"""
        filename_lower = filename.lower()

        if "employee" in filename_lower or "staff" in filename_lower:
            return "employee_data"
        elif "customer" in filename_lower:
            return "customer_data"
        elif "transfer" in filename_lower or "payment" in filename_lower:
            return "transaction_data"
        elif "agreement" in filename_lower or "contract" in filename_lower:
            return "legal_documents"
        elif "policy" in filename_lower or "guideline" in filename_lower:
            return "internal_policies"
        else:
            return "general_banking"

    def load_large_csv_dataset(self) -> List[Dict[str, Any]]:
        """Load existing large CSV dataset if available"""

        # Look for large CSV files in the current directory
        csv_files = []
        for file_path in Path(".").glob("*.csv"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Files larger than 10MB
                csv_files.append((file_path, size_mb))

        if not csv_files:
            logger.info("No large CSV files found in current directory")
            return []

        # Sort by size, largest first
        csv_files.sort(key=lambda x: x[1], reverse=True)
        largest_file, size_mb = csv_files[0]

        logger.info(f"Found large CSV file: {largest_file.name} ({size_mb:.1f} MB)")

        try:
            df = pd.read_csv(largest_file)
            logger.info(f"Loaded {len(df)} rows from {largest_file.name}")

            # Convert to training format
            training_samples = []

            for _, row in df.iterrows():
                if "original_text" in row and "redacted_text" in row:
                    sample = {
                        "original_text": row["original_text"],
                        "redacted_text": row.get("redacted_text", ""),
                        "sensitivity_level": row.get("sensitivity_level", "MEDIUM"),
                        "document_category": row.get(
                            "business_category", "general_banking"
                        ),
                        "source": f"large_csv_{largest_file.name}",
                    }
                    training_samples.append(sample)

            logger.info(f"Converted {len(training_samples)} samples from large CSV")
            return training_samples

        except Exception as e:
            logger.error(f"Error loading large CSV {largest_file.name}: {e}")
            return []

    def generate_and_save_large_dataset(
        self, num_samples: int = 50000, save_to_csv: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate a large dataset and optionally save to CSV"""
        logger.info(f"Generating large dataset with {num_samples} samples...")

        # Generate in batches to avoid memory issues
        batch_size = 1000
        all_samples = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_samples = self.generate_synthetic_data(batch_end - batch_start)
            all_samples.extend(batch_samples)

            logger.info(
                f"Generated batch {batch_start//batch_size + 1}/{(num_samples-1)//batch_size + 1}: {len(batch_samples)} samples"
            )

        if save_to_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"combined_training_data_{timestamp}.csv"

            # Convert to DataFrame and save
            df_data = []
            for sample in all_samples:
                df_data.append(
                    {
                        "original_text": sample["original_text"],
                        "redacted_text": sample.get("redacted_text", ""),
                        "sensitivity_level": sample.get("sensitivity_level", "MEDIUM"),
                        "business_category": sample.get(
                            "document_category", "general_banking"
                        ),
                        "source": sample.get("source", "combined_generators"),
                        "entity_count": sample.get("entity_count", 0),
                    }
                )

            df = pd.DataFrame(df_data)
            df.to_csv(csv_filename, index=False)

            logger.info(f"Saved {len(all_samples)} samples to {csv_filename}")

        return all_samples
        """Generate synthetic training data using both generators"""
        logger.info(
            f"Generating {num_samples} synthetic training samples using both generators..."
        )

        # Split samples between generators
        gen1_samples = num_samples // 2  # Original generator
        gen2_samples = num_samples - gen1_samples  # Banking generator

        all_samples = []

        # Generate samples from original generator (SyntheticDataGenerator)
        logger.info(f"Generating {gen1_samples} samples with SyntheticDataGenerator...")
        for i in range(gen1_samples):
            if i % 500 == 0:
                logger.info(f"Generated {i}/{gen1_samples} samples (Generator 1)...")

            try:
                sample = self.data_generator.generate_training_sample()
                all_samples.append(sample)
            except Exception as e:
                logger.error(f"Error generating sample {i} with Generator 1: {e}")

        # Generate samples from banking generator (BankingDataGenerator)
        logger.info(f"Generating {gen2_samples} samples with BankingDataGenerator...")
        for i in range(gen2_samples):
            if i % 500 == 0:
                logger.info(f"Generated {i}/{gen2_samples} samples (Generator 2)...")

            try:
                banking_sample = self.banking_generator.generate_sample()

                # Convert banking sample format to training format
                converted_sample = {
                    "original_text": banking_sample["original_text"],
                    "redacted_text": banking_sample["redacted_text"],
                    "sensitivity_level": banking_sample["sensitivity_level"],
                    "document_category": banking_sample["business_category"],
                    "source": "banking_generator",
                    "entity_count": banking_sample["entity_count"],
                    "entities": [
                        {
                            "type": entity["entity_type"],
                            "original": entity["original"],
                            "redacted": entity["placeholder"],
                            "sensitivity": entity["sensitivity"],
                            "confidence": entity["confidence"],
                        }
                        for entity in banking_sample["entity_mappings"]
                    ],
                }
                all_samples.append(converted_sample)

            except Exception as e:
                logger.error(f"Error generating sample {i} with Generator 2: {e}")

        logger.info(f"Generated {len(all_samples)} synthetic samples total")
        logger.info(f"  - SyntheticDataGenerator: {gen1_samples} samples")
        logger.info(f"  - BankingDataGenerator: {gen2_samples} samples")

        return all_samples

    def prepare_training_data(
        self, use_existing: bool = True, num_synthetic: int = 2000
    ) -> List[Dict[str, Any]]:
        """Prepare combined training dataset"""
        all_samples = []

        # Load existing data if requested
        if use_existing:
            existing_samples = self.load_existing_data()
            all_samples.extend(existing_samples)

        # Generate synthetic data
        if num_synthetic > 0:
            synthetic_samples = self.generate_synthetic_data(num_synthetic)
            all_samples.extend(synthetic_samples)

        # Analyze data distribution
        self._analyze_data_distribution(all_samples)

        self.training_data = all_samples
        return all_samples

    def _analyze_data_distribution(self, data: List[Dict[str, Any]]):
        """Analyze the distribution of training data"""
        logger.info("=== Training Data Analysis ===")

        # Count by sensitivity level
        sensitivity_counts = {}
        category_counts = {}
        source_counts = {}

        for sample in data:
            sensitivity = sample.get("sensitivity_level", "Unknown")
            category = sample.get("document_category", "Unknown")
            source = sample.get("source", "synthetic_data_generator")

            sensitivity_counts[sensitivity] = sensitivity_counts.get(sensitivity, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1

        logger.info(f"Total samples: {len(data)}")

        logger.info("Data source distribution:")
        for source, count in sorted(source_counts.items()):
            percentage = (count / len(data)) * 100
            logger.info(f"  {source}: {count} ({percentage:.1f}%)")

        logger.info("Sensitivity distribution:")
        for level, count in sorted(sensitivity_counts.items()):
            percentage = (count / len(data)) * 100
            logger.info(f"  {level}: {count} ({percentage:.1f}%)")

        logger.info("Category distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(data)) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")

    def train_model(self) -> Dict[str, Any]:
        """Train the enhanced redaction model"""
        if not self.training_data:
            raise ValueError(
                "No training data available. Call prepare_training_data() first."
            )

        logger.info("Starting model training...")

        # Train the model
        metrics = self.model.train(self.training_data)
        self.metrics = metrics

        logger.info("Training completed!")
        logger.info(f"Final accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def evaluate_model(
        self, test_samples: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate the trained model on test samples"""
        if test_samples is None:
            test_samples = [
                "Dear John Smith, your account NL91ABNA0417164300 has been credited with EUR 5,234.56.",
                "Employee ID: AB123CD, Phone: +31 20 123 4567, Email: john.doe@ing.com",
                "Transaction 550e8400-e29b-41d4-a716-446655440000 processed successfully.",
                "Customer data: Name: Jane Doe, DOB: 1985-03-15, SSN: 123456789",
                "Internal policy document regarding data protection procedures.",
            ]

        logger.info("=== Model Evaluation ===")

        evaluation_results = []

        for i, test_text in enumerate(test_samples):
            logger.info(f"\nTest {i+1}: {test_text[:50]}...")

            result = self.model.redact_text(test_text)

            eval_result = {
                "original": test_text,
                "redacted": result.redacted_text,
                "sensitivity": result.sensitivity_level,
                "confidence": result.confidence,
                "detections": len(result.detections),
                "detection_summary": result.detection_summary,
                "method_summary": result.method_summary,
            }

            evaluation_results.append(eval_result)

            logger.info(
                f"  Sensitivity: {result.sensitivity_level} (confidence: {result.confidence:.2f})"
            )
            logger.info(f"  Redactions: {result.total_redacted}")
            logger.info(f"  Methods: {result.method_summary}")
            logger.info(f"  Redacted: {result.redacted_text}")

        return {"test_results": evaluation_results, "model_metrics": self.metrics}

    def save_results(self, output_dir: str = "models") -> None:
        """Save training results and model artifacts"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save training metrics
        metrics_file = output_path / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Save data analysis
        if self.training_data:
            analysis = {
                "total_samples": len(self.training_data),
                "sensitivity_distribution": {},
                "category_distribution": {},
            }

            for sample in self.training_data:
                sensitivity = sample.get("sensitivity_level", "Unknown")
                category = sample.get("document_category", "Unknown")

                analysis["sensitivity_distribution"][sensitivity] = (
                    analysis["sensitivity_distribution"].get(sensitivity, 0) + 1
                )
                analysis["category_distribution"][category] = (
                    analysis["category_distribution"].get(category, 0) + 1
                )

            analysis_file = output_path / "data_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train Enhanced Banking Redaction Model"
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=20000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        default=True,
        help="Use existing CSV data",
    )
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for models and results"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation after training"
    )

    args = parser.parse_args()

    print("🤖 Enhanced Banking Redaction Model Trainer")
    print("=" * 50)

    # Initialize trainer
    trainer = EnhancedModelTrainer()

    try:
        # Prepare training data
        print(f"📊 Preparing training data...")
        training_data = trainer.prepare_training_data(
            use_existing=args.use_existing, num_synthetic=args.synthetic_samples
        )

        if not training_data:
            print("❌ No training data available!")
            return

        # Train model
        print(f"🔥 Training model...")
        metrics = trainer.train_model()

        print(f"✅ Training completed!")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Train samples: {metrics['train_size']}")
        print(f"   Test samples: {metrics['test_size']}")

        # Evaluate model if requested
        if args.evaluate:
            print(f"🧪 Evaluating model...")
            eval_results = trainer.evaluate_model()

            print(f"✅ Evaluation completed!")
            for i, result in enumerate(eval_results["test_results"]):
                print(
                    f"   Test {i+1}: {result['sensitivity']} ({result['detections']} redactions)"
                )

        # Save results
        print(f"💾 Saving results...")
        trainer.save_results(args.output_dir)

        print(f"🎉 Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
