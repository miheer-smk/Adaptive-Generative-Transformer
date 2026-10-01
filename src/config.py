"""
AGT-v3 POC — Central Configuration

Source of truth:
AGT-v3 blueprint pre-freeze v0.1 (30 Sep 2026)
"""

# ============================================================
# Model dimensions
# ============================================================

N_BLOCKS = 8
WIDTH = 768
N_HC = 2
MEMORY_SLOTS = 16

# ============================================================
# Heterogeneous block bank
#
# Required Milestone-1 composition:
#   2 SWA
#   2 GQA-global
#   2 GDN
#   2 FFN-heavy
# ============================================================

SPECIES_MIX = {
    "SWA": 2,
    "GQA": 2,
    "GDN": 2,
    "FFN": 2,
}

# ============================================================
# Execution
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