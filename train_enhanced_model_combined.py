#!/usr/bin/env python3
"""
Training Script for Enhanced Banking Redaction Model - Combined Generators
Uses both synthetic data generators to train NER + classification hybrid model
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


class CombinedDataTrainer:
    """Enhanced trainer using both data generators"""

    def __init__(self):
        self.model = create_enhanced_redaction_model()
        self.original_generator = SyntheticDataGenerator()
        self.banking_generator = BankingDataGenerator()
        self.training_data = []
        self.metrics = {}

    def check_for_existing_large_dataset(self) -> Optional[str]:
        """Check if there's already a large CSV dataset"""

        # Look for large CSV files in current directory
        csv_files = []
        for file_path in Path(".").glob("*.csv"):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Files larger than 10MB
                csv_files.append((file_path, size_mb))

        if csv_files:
            # Sort by size, largest first
            csv_files.sort(key=lambda x: x[1], reverse=True)
            largest_file, size_mb = csv_files[0]
            logger.info(
                f"Found existing large CSV: {largest_file.name} ({size_mb:.1f} MB)"
            )
            return str(largest_file)

        return None

    def load_large_csv_dataset(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load existing large CSV dataset"""

        logger.info(f"Loading large dataset from {csv_path}...")

        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")

            # Convert to training format
            training_samples = []

            for _, row in df.iterrows():
                # Handle different CSV formats
                if "original_text" in row and "redacted_text" in row:
                    sample = {
                        "original_text": row["original_text"],
                        "redacted_text": row.get("redacted_text", ""),
                        "sensitivity_level": row.get("sensitivity_level", "MEDIUM"),
                        "document_category": row.get(
                            "business_category", "general_banking"
                        ),
                        "source": f"large_csv",
                        "entity_count": row.get("entity_count", 0),
                    }
                    training_samples.append(sample)

            logger.info(f"Converted {len(training_samples)} samples from large CSV")
            return training_samples

        except Exception as e:
            logger.error(f"Error loading large CSV {csv_path}: {e}")
            return []

    def load_existing_small_datasets(
        self, data_dir: str = "data/classification"
    ) -> List[Dict[str, Any]]:
        """Load existing small CSV data and convert to training format"""
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
                            "source": f"existing_csv_{csv_file.name}",
                            "document_category": self._infer_category(csv_file.name),
                        }
                        training_samples.append(training_sample)

                        # Limit samples per file to prevent imbalance
                        if (
                            len(
                                [
                                    s
                                    for s in training_samples
                                    if s["source"] == f"existing_csv_{csv_file.name}"
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

    def generate_combined_synthetic_data(
        self, num_samples: int = 10000
    ) -> List[Dict[str, Any]]:
        """Generate synthetic data using both generators"""
        logger.info(f"🤖 Generating {num_samples} samples using BOTH generators...")

        # Split samples between generators (60% banking, 40% original)
        banking_samples = int(num_samples * 0.6)
        original_samples = num_samples - banking_samples

        all_samples = []

        # Generate samples from banking generator (more comprehensive)
        logger.info(
            f"🏦 Generating {banking_samples} samples with BankingDataGenerator..."
        )
        for i in range(banking_samples):
            if i % 1000 == 0:
                logger.info(f"   Banking samples: {i}/{banking_samples}...")

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
                logger.error(f"Error generating banking sample {i}: {e}")

        # Generate samples from original generator
        logger.info(
            f"📝 Generating {original_samples} samples with SyntheticDataGenerator..."
        )
        for i in range(original_samples):
            if i % 1000 == 0:
                logger.info(f"   Original samples: {i}/{original_samples}...")

            try:
                sample = self.original_generator.generate_training_sample()
                # Add source identifier
                sample["source"] = "original_generator"
                all_samples.append(sample)
            except Exception as e:
                logger.error(f"Error generating original sample {i}: {e}")

        logger.info(f"✅ Generated {len(all_samples)} synthetic samples total")
        logger.info(f"   - BankingDataGenerator: {banking_samples} samples")
        logger.info(f"   - SyntheticDataGenerator: {original_samples} samples")

        return all_samples

    def generate_and_save_large_dataset(
        self, num_samples: int = 200000, save_to_csv: bool = True
    ) -> str:
        """Generate a large dataset and save to CSV"""
        logger.info(f"🚀 Generating LARGE dataset with {num_samples} samples...")

        # Generate in batches to avoid memory issues
        batch_size = 5000
        all_samples = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_size_actual = batch_end - batch_start

            logger.info(
                f"Generating batch {batch_start//batch_size + 1}/{(num_samples-1)//batch_size + 1}: {batch_size_actual} samples"
            )

            batch_samples = self.generate_combined_synthetic_data(batch_size_actual)
            all_samples.extend(batch_samples)

        if save_to_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"combined_training_data_{num_samples}_{timestamp}.csv"

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

            size_mb = Path(csv_filename).stat().st_size / (1024 * 1024)
            logger.info(
                f"💾 Saved {len(all_samples)} samples to {csv_filename} ({size_mb:.1f} MB)"
            )

            return csv_filename

        return ""

    def prepare_training_data(
        self,
        use_existing_large: bool = True,
        use_existing_small: bool = True,
        num_synthetic: int = 10000,
        generate_large_dataset: bool = False,
        large_dataset_size: int = 200000,
    ) -> List[Dict[str, Any]]:
        """Prepare comprehensive training dataset"""
        all_samples = []

        # Check for existing large dataset first
        if use_existing_large:
            large_csv = self.check_for_existing_large_dataset()
            if large_csv:
                large_samples = self.load_large_csv_dataset(large_csv)
                all_samples.extend(large_samples)
                logger.info(
                    f"✅ Loaded {len(large_samples)} samples from existing large CSV"
                )
            elif generate_large_dataset:
                logger.info("🚀 No large dataset found, generating new one...")
                large_csv = self.generate_and_save_large_dataset(large_dataset_size)
                if large_csv:
                    large_samples = self.load_large_csv_dataset(large_csv)
                    all_samples.extend(large_samples)

        # Load existing small CSV data if requested
        if use_existing_small:
            existing_samples = self.load_existing_small_datasets()
            all_samples.extend(existing_samples)

        # Generate additional synthetic data if requested
        if num_synthetic > 0 and not all_samples:  # Only if we don't have large dataset
            synthetic_samples = self.generate_combined_synthetic_data(num_synthetic)
            all_samples.extend(synthetic_samples)

        # Analyze data distribution
        self._analyze_data_distribution(all_samples)

        self.training_data = all_samples
        return all_samples

    def _analyze_data_distribution(self, data: List[Dict[str, Any]]):
        """Analyze the distribution of training data"""
        logger.info("=== Training Data Analysis ===")

        # Count by different dimensions
        sensitivity_counts = {}
        category_counts = {}
        source_counts = {}

        for sample in data:
            sensitivity = sample.get("sensitivity_level", "Unknown")
            category = sample.get("document_category", "Unknown")
            source = sample.get("source", "unknown")

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

        logger.info("🔥 Starting model training...")

        # Train the model
        metrics = self.model.train(self.training_data)
        self.metrics = metrics

        logger.info("✅ Training completed!")
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
                "Transaction TXN123456789012 processed successfully for customer CUST-001234.",
                "Customer data: Name: Jane Doe, DOB: 1985-03-15, SSN: 123456789",
                "Internal policy document regarding data protection procedures.",
                "IBAN: BE68 5390 0754 7034, BIC: GKCCBEBB, Amount: €45,678.90",
                "Credit card ending in 1234 was charged $2,500.00 on 2024-01-15",
            ]

        logger.info("🧪 Model Evaluation")

        evaluation_results = []

        for i, test_text in enumerate(test_samples):
            logger.info(f"\nTest {i+1}: {test_text[:60]}...")

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
                "source_distribution": {},
            }

            for sample in self.training_data:
                sensitivity = sample.get("sensitivity_level", "Unknown")
                category = sample.get("document_category", "Unknown")
                source = sample.get("source", "Unknown")

                analysis["sensitivity_distribution"][sensitivity] = (
                    analysis["sensitivity_distribution"].get(sensitivity, 0) + 1
                )
                analysis["category_distribution"][category] = (
                    analysis["category_distribution"].get(category, 0) + 1
                )
                analysis["source_distribution"][source] = (
                    analysis["source_distribution"].get(source, 0) + 1
                )

            analysis_file = output_path / "data_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)

        logger.info(f"💾 Results saved to {output_path}")


def main():
    """Main training function with combined generators"""
    parser = argparse.ArgumentParser(
        description="Train Enhanced Banking Redaction Model with Combined Generators"
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate (if no large dataset)",
    )
    parser.add_argument(
        "--generate-large",
        action="store_true",
        help="Generate large dataset if none exists",
    )
    parser.add_argument(
        "--large-size",
        type=int,
        default=200000,
        help="Size of large dataset to generate",
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

    print("🤖 Enhanced Banking Redaction Model Trainer - Combined Generators")
    print("=" * 70)

    # Initialize trainer
    trainer = CombinedDataTrainer()

    try:
        # Prepare training data
        print(f"📊 Preparing training data...")
        training_data = trainer.prepare_training_data(
            use_existing_large=True,
            use_existing_small=args.use_existing,
            num_synthetic=args.synthetic_samples,
            generate_large_dataset=args.generate_large,
            large_dataset_size=args.large_size,
        )

        if not training_data:
            print("❌ No training data available!")
            return

        print(f"✅ Prepared {len(training_data)} training samples")

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
