"""Single source of truth for paths and business constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_card_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "reports" / "figures"
POWERBI_DIR = ROOT / "powerbi" / "data"
AUDIT_LOG = LOG_DIR / "fraud_audit.jsonl"

RANDOM_STATE = 42

# Cost matrix (from the project report)
COST_FN = 5000   # missed fraud
COST_FP = 200    # wrongly alerted genuine transaction

# Business rules (report section 5.1.2)
SPIKE_MULTIPLIER = 3
VERY_LARGE_AMOUNT = 100_000
VELOCITY_GAP_MIN = 5
VELOCITY_AMOUNT = 5_000
TRAVEL_GAP_MIN = 30
RULE_WEIGHTS = {"spike": 0.35, "very_large": 0.40, "velocity": 0.25, "travel": 0.40}

DEFAULT_GAP_MIN = 1440.0  # used when a user has no previous transaction
