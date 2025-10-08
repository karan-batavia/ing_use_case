
#!/usr/bin/env python3
"""
predict_snippets.py — quick tester for the sensitivity classifier

Usage examples:
  python predict_snippets.py --model ./sensitivity_classifier.joblib --text "Draft a LinkedIn post for the 2024 Annual Report."
  python predict_snippets.py --model ./sensitivity_classifier.joblib --file ./samples.txt
  echo "Customer account balance is 12,500 EUR" | python predict_snippets.py --model ./sensitivity_classifier.joblib --stdin

If --model is omitted, it tries ./sensitivity_classifier.joblib next to this script.
"""

import sys
import json
import argparse
from pathlib import Path
import joblib

# Preload model defs so joblib can unpickle custom classes
for candidate in ("ml_setup_chatgpt", "ml_setup"):
    try:
        mod = __import__(candidate)
        if hasattr(mod, "RuleFeatureizer"):
            sys.modules["__main__"].RuleFeatureizer = mod.RuleFeatureizer
            break
    except Exception:
        pass

def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    data = joblib.load(str(path))  
    pipe = data["pipeline"]
    labels = data.get("labels", ["C1","C2","C3","C4"])
    return pipe, labels

def iter_inputs(args):
    # Priority: --text, --file, --stdin
    if args.text is not None:
        yield args.text.strip()
        return
    if args.file is not None:
        p = Path(args.file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if line:
                yield line
        return
    if args.stdin:
        for line in sys.stdin:
            line=line.strip()
            if line:
                yield line
        return
    # If nothing provided, print help
    raise SystemExit("Provide --text, --file, or --stdin (see -h).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="Path to sensitivity_classifier.joblib")
    ap.add_argument("--text", type=str, default=None, help="Single text to classify")
    ap.add_argument("--file", type=str, default=None, help="Path to a file with one text per line")
    ap.add_argument("--stdin", action="store_true", help="Read texts from STDIN (one per line)")
    ap.add_argument("--proba", action="store_true", help="Show class probabilities if available")
    args = ap.parse_args()

    model_path = Path(args.model) if args.model else (Path(__file__).parent / "sensitivity_classifier.joblib")
    pipe, labels = load_model(model_path)

    # Try to get probability support
    has_proba = args.proba and hasattr(getattr(pipe, "named_steps", {}).get("clf", pipe), "predict_proba")

    inputs = list(iter_inputs(args))
    preds = pipe.predict(inputs)

    if has_proba:
        proba = pipe.predict_proba(inputs)
    else:
        proba = None

    for i, text in enumerate(inputs):
        label = preds[i]
        print(f"[{label}] {text}")
        if proba is not None:
            probs = {labels[j]: float(proba[i][j]) for j in range(len(labels))}
            print("  probs:", json.dumps(probs))

if __name__ == "__main__":
    main()
