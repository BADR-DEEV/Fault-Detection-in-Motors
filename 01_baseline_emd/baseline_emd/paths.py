from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "data" / "paper_data"
FIGURES_DIR = PROJECT_ROOT / "figures" / "02_multiclass_xai"
MODELS_DIR = PROJECT_ROOT / "models" / "02_multiclass_xai"
PLAYGROUND_MODELS_DIR = PROJECT_ROOT / "playground_models" / "02_multiclass_xai"
ARTIFACTS_DIR = PHASE_ROOT / "artifacts"
