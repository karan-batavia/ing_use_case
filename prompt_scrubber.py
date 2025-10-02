import os
import pandas as pd
from typing import List, Dict


class PromptScrubber:
    def __init__(self):
        self.classifications_dir = "./data/classification"
        self.classifications = self._load_classifications()

    def _load_classifications(self) -> Dict[str, Dict[str, str]]:
        """Load all classification files (CSV and XLSX) and their contents."""
        classifications = {}

        if not os.path.exists(self.classifications_dir):
            print(
                f"Warning: Classifications directory '{self.classifications_dir}' not found"
            )
            return classifications

        for filename in os.listdir(self.classifications_dir):
            file_path = os.path.join(self.classifications_dir, filename)

            # Only process CSV and XLSX files for now
            # TODO: handle pdf, potentially other format
            if os.path.isfile(file_path):
                try:
                    if filename.endswith(".csv"):
                        # Try to detect delimiter first
                        with open(file_path, "r", encoding="utf-8") as f:
                            first_line = f.readline()
                            if ";" in first_line and first_line.count(
                                ";"
                            ) > first_line.count(","):
                                df = pd.read_csv(file_path, delimiter=";")
                            else:
                                df = pd.read_csv(file_path)
                    elif filename.endswith(".xlsx"):
                        df = pd.read_excel(file_path)
                    else:
                        continue  # Skip non-CSV/XLSX files

                    # Create a mapping from value to column name
                    value_to_column = {}
                    for column in df.columns:
                        column_values = df[column].dropna().astype(str).tolist()
                        for value in column_values:
                            value = value.strip()
                            if value:  # Only add non-empty values
                                value_to_column[value] = column

                    classifications[filename] = value_to_column

                except Exception as e:
                    print(f"Error reading file {filename}: {e}")

        return classifications

    def scrub(self, prompt: str) -> Dict[str, List[str]]:
        """
        Check if the prompt contains any values from classification files.

        Args:
            prompt: The input prompt to check

        Returns:
            Dictionary with filename as key and list of found matches as value
        """
        found_matches = {}
        prompt_lower = prompt.lower()

        for filename, values in self.classifications.items():
            matches = []
            for value, column_name in values.items():
                value_lower = value.lower()
                # Check for exact matches or matches as whole words
                # For URLs and longer strings, use exact substring matching
                # For shorter strings, check if they appear as whole words
                if len(value) > 10 or value.startswith("http"):
                    # For URLs and longer strings, use exact substring match
                    if value_lower in prompt_lower:
                        matches.append(value)
                else:
                    # For shorter strings, check if they appear as whole words
                    import re

                    pattern = r"\b" + re.escape(value_lower) + r"\b"
                    if re.search(pattern, prompt_lower):
                        matches.append(value)

            if matches:
                found_matches[filename] = matches

        return found_matches

    def scrub_prompt(self, prompt: str) -> str:
        """
        Returns the prompt with classified values replaced by their column names in <> brackets.

        Args:
            prompt: The input prompt to scrub

        Returns:
            The scrubbed prompt with values replaced by <column_name>
        """
        scrubbed_prompt = prompt

        # Collect all matches with their column names
        matches_with_columns = []

        for filename, values in self.classifications.items():
            for value, column_name in values.items():
                value_lower = value.lower()
                prompt_lower = prompt.lower()

                # Check for exact matches or matches as whole words
                if len(value) > 10 or value.startswith("http"):
                    # For URLs and longer strings, use exact substring match
                    if value_lower in prompt_lower:
                        matches_with_columns.append((value, column_name))
                else:
                    # For shorter strings, check if they appear as whole words
                    import re

                    pattern = r"\b" + re.escape(value_lower) + r"\b"
                    if re.search(pattern, prompt_lower):
                        matches_with_columns.append((value, column_name))

        # Sort matches by length (longest first) to avoid partial replacements
        matches_with_columns.sort(key=lambda x: len(x[0]), reverse=True)

        # Replace matches with column names in brackets
        for value, column_name in matches_with_columns:
            # Use case-insensitive replacement but preserve original case structure
            import re

            pattern = re.compile(re.escape(value), re.IGNORECASE)
            scrubbed_prompt = pattern.sub(f"<{column_name}>", scrubbed_prompt)

        return scrubbed_prompt

    def get_all_classifications(self) -> Dict[str, Dict[str, str]]:
        """Return all loaded classifications."""
        return self.classifications
