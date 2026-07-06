from pathlib import Path


PHASE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw_mafulda"
FIGURES_DIR = PROJECT_ROOT / "figures" / "02_multiclass_xai"
MODELS_DIR = PROJECT_ROOT / "models" / "02_multiclass_xai"
PLAYGROUND_MODELS_DIR = PROJECT_ROOT / "playground_models" / "02_multiclass_xai"
