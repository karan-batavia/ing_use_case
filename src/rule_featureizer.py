# src/rule_featureizer.py
import re
import sys

# --- Patch for joblib model pickles that reference __main__.RuleFeatureizer ---
from types import ModuleType

# Dynamically create a dummy module named '__main__' for unpickling
# if RuleFeatureizer was defined in __main__ during training
if '__main__' not in sys.modules:
    sys.modules['__main__'] = ModuleType('__main__')

class RuleFeatureizer:
    def __init__(self, *args, **kwargs): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# Attach it to __main__ and also to src.rule_featureizer for future imports
sys.modules['__main__'].RuleFeatureizer = RuleFeatureizer
sys.modules['src.rule_featureizer'] = sys.modules['__main__']
