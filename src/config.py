import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STUDENT_ROOT = Path(r"D:\CLG\HACK\Student folder\Student folder")
TRAIN_DIR = STUDENT_ROOT / "train"
TEST_DIR = STUDENT_ROOT / "student_test"

# Output directories
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models_saved"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure dirs exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Geodesic & Spatial Constants
R_EARTH = 6371000.0  # Earth radius in meters
CANDIDATE_RADIUS_METERS = 45.0  # Search radius around GPS points
MAX_SPEED_MPS = 50.0  # ~180 km/h sanity threshold
RANDOM_STATE = 42

# Matplotlib cache fix
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".cache" / "matplotlib")
