#!/usr/bin/env python3
"""
Batch Processing Example - Process large files or datasets with the enhanced model
"""

import sys
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from src.enhanced_redaction_model_clean import create_enhanced_redaction_model
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class BatchRedactionProcessor:
    """Handles batch processing of large datasets"""

    def __init__(self, max_workers: int = 4):
        print("🔄 Loading enhanced redaction model...")
        self.model = create_enhanced_redaction_model()
        self.max_workers = max_workers
        print("✅ Model loaded!")

    def process_text_batch(
        self,
        texts: List[str],
        sensitivity_level: str = "C3",
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process a batch of texts"""

        results = []
        total = len(texts)

        print(f"🔄 Processing {total} texts with sensitivity level {sensitivity_level}")

        def process_single(text_data):
            idx, text = text_data
            try:
                result = self.model.redact_text(text, sensitivity_level)
                return {
                    "index": idx,
                    "original_text": text,
                    "redacted_text": result.redacted_text,
                    "entities_detected": len(result.detections),
                    "confidence": result.confidence,
                    "detection_summary": result.detection_summary,
                    "processing_status": "success",
                }
            except Exception as e:
                return {
                    "index": idx,
                    "original_text": text,
                    "redacted_text": text,  # Return original on error
                    "entities_detected": 0,
                    "confidence": 0.0,
                    "detection_summary": {},
                    "processing_status": f"error: {str(e)}",
                }

        # Process with threading for better performance
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_text = {
                executor.submit(process_single, (i, text)): i
                for i, text in enumerate(texts)
            }

            # Collect results
            completed = 0
            for future in as_completed(future_to_text):
                result = future.result()
                results.append(result)
                completed += 1

                if show_progress and completed % 100 == 0:
                    print(f"  ✅ Processed {completed}/{total} texts...")

        # Sort by original index
        results.sort(key=lambda x: x["index"])

        # Print summary
        successful = sum(1 for r in results if r["processing_status"] == "success")
        total_entities = sum(r["entities_detected"] for r in results)
        avg_confidence = (
            sum(r["confidence"] for r in results) / len(results) if results else 0
        )

        print(f"📊 Batch Processing Summary:")
        print(f"  Total processed: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {total - successful}")
        print(f"  Total entities detected: {total_entities}")
        print(f"  Average confidence: {avg_confidence:.3f}")

        return results

    def process_csv_file(
        self,
        input_file: str,
        output_file: str,
        text_column: str = "text",
        sensitivity_column: Optional[str] = None,
        default_sensitivity: str = "C3",
    ):
        """Process a CSV file"""

        print(f"📁 Processing CSV file: {input_file}")

        # Read CSV
        df = pd.read_csv(input_file)
        print(f"📊 Loaded {len(df)} rows")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")

        # Process each row
        results = []
        for row_num, (row_idx, row) in enumerate(df.iterrows()):
            text = str(row[text_column])

            # Determine sensitivity level
            if sensitivity_column and sensitivity_column in df.columns:
                sensitivity = str(row[sensitivity_column])
            else:
                sensitivity = default_sensitivity

            # Process text
            try:
                result = self.model.redact_text(text, sensitivity)

                # Create result row
                result_row = {
                    "original_index": row_idx,
                    "original_text": text,
                    "redacted_text": result.redacted_text,
                    "sensitivity_level": sensitivity,
                    "entities_detected": len(result.detections),
                    "confidence": result.confidence,
                    "detection_summary": json.dumps(result.detection_summary),
                    "method_summary": json.dumps(result.method_summary),
                    "processing_status": "success",
                }

                # Add original columns
                for col in df.columns:
                    if col not in result_row:
                        result_row[f"original_{col}"] = row[col]

                results.append(result_row)

            except Exception as e:
                results.append(
                    {
                        "original_index": row_idx,
                        "original_text": text,
                        "redacted_text": text,
                        "sensitivity_level": sensitivity,
                        "entities_detected": 0,
                        "confidence": 0.0,
                        "detection_summary": "{}",
                        "method_summary": "{}",
                        "processing_status": f"error: {str(e)}",
                    }
                )

            if (row_num + 1) % 1000 == 0:
                print(f"  ✅ Processed {row_num + 1}/{len(df)} rows...")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

        print(f"💾 Results saved to: {output_file}")
        return results_df

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.txt",
        sensitivity_level: str = "C3",
    ):
        """Process all files in a directory"""

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        files = list(input_path.glob(file_pattern))
        print(f"📁 Found {len(files)} files matching '{file_pattern}'")

        for file_path in files:
            print(f"\n🔄 Processing: {file_path.name}")

            try:
                # Read file
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Process
                result = self.model.redact_text(text, sensitivity_level)

                # Save redacted version
                output_file = output_path / f"redacted_{file_path.name}"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result.redacted_text)

                # Save metadata
                metadata_file = output_path / f"metadata_{file_path.stem}.json"
                metadata = {
                    "original_file": str(file_path),
                    "redacted_file": str(output_file),
                    "sensitivity_level": sensitivity_level,
                    "entities_detected": len(result.detections),
                    "confidence": result.confidence,
                    "detection_summary": result.detection_summary,
                    "method_summary": result.method_summary,
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"  ✅ Saved: {output_file.name}")
                print(
                    f"  📊 Entities: {len(result.detections)}, Confidence: {result.confidence:.3f}"
                )

            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Batch process files with enhanced redaction"
    )
    parser.add_argument(
        "--mode",
        choices=["csv", "directory", "test"],
        default="test",
        help="Processing mode",
    )
    parser.add_argument("--input", help="Input file/directory")
    parser.add_argument("--output", help="Output file/directory")
    parser.add_argument(
        "--sensitivity",
        default="C3",
        choices=["C1", "C2", "C3", "C4"],
        help="Sensitivity level",
    )
    parser.add_argument(
        "--text-column", default="text", help="Text column name for CSV"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )

    args = parser.parse_args()

    # Initialize processor
    processor = BatchRedactionProcessor(max_workers=args.workers)

    if args.mode == "test":
        # Test mode with sample data
        print("🧪 Running test mode...")

        sample_texts = [
            "Dear John Smith, your account NL91ABNA0417164300 has a balance of €50,000.",
            "Employee Sarah Johnson can be reached at s.johnson@ing.com or +31-6-12345678.",
            "Transfer €25,000 from NL91ABNA0417164300 to DE89370400440532013000.",
            "Our new product offers competitive rates. Visit www.ing.nl for details.",
            "Customer ID: CUST789456, Email: customer@example.com, Phone: +31-20-1234567",
        ]

        results = processor.process_text_batch(
            sample_texts, sensitivity_level=args.sensitivity
        )

        print(f"\n📋 Sample Results:")
        for i, result in enumerate(results[:3]):  # Show first 3
            print(f"\n{i+1}. Original: {result['original_text']}")
            print(f"   Redacted: {result['redacted_text']}")
            print(f"   Entities: {result['entities_detected']}")

    elif args.mode == "csv":
        if not args.input or not args.output:
            print("❌ CSV mode requires --input and --output arguments")
            return

        processor.process_csv_file(
            input_file=args.input,
            output_file=args.output,
            text_column=args.text_column,
            default_sensitivity=args.sensitivity,
        )

    elif args.mode == "directory":
        if not args.input or not args.output:
            print("❌ Directory mode requires --input and --output arguments")
            return

        processor.process_directory(
            input_dir=args.input,
            output_dir=args.output,
            sensitivity_level=args.sensitivity,
        )

    print("\n✅ Batch processing completed!")


if __name__ == "__main__":
    main()
