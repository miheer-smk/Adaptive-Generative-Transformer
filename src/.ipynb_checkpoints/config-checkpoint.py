"""
AGT-v3 POC — Central Configuration

Source of truth:
AGT-v3 blueprint pre-freeze v0.1 (30 Sep 2026)

Provisional values are explicitly marked.
"""

# ============================================================
# Model dimensions — PROVISIONAL
# ============================================================

N_BLOCKS = 8
WIDTH = 256
MEMORY_SLOTS = 16

# Species order/mix — PROVISIONAL
# 2 : 2 : 2 : 2
SPECIES_MIX = {
    "GQA": 2,
    "GDN": 2,
    "mHC": 2,
    "MEMORY": 2,
}

# ============================================================
# Execution — PROVISIONAL
# ============================================================

MAX_EXECUTION_DEPTH = 256

# Backpropagation truncation
K_BP = 4

# Controller / halting
HALT_EPSILON = 1e-6

# ============================================================
# Numerical settings
# ============================================================

DEFAULT_DTYPE = "float32"

# Stage 0 alarm threshold — provisional
NORM_GROWTH_ALARM_FACTOR = 100.0

# ============================================================
# Reproducibility
# ============================================================

DEFAULT_SEED = 42

# ============================================================
# Validation
# ============================================================

EXPECTED_PARAMETER_COUNT = 86_381_128
