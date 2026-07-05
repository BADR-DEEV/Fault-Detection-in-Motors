# AI-Driven Vibrations Motor

Universal project README for the three production phases:

- `01_baseline_emd`
- `02_multiclass_xai`
- `03_ordetracking_cnn`

## Overview

This repository contains three organized fault-diagnosis pipelines built around vibration analysis for rotating machinery.

- Phase 1 is a classical baseline using EMD plus handcrafted features plus SVM-style modeling.
- Phase 2 is a paper-reproduction pipeline for binary healthy vs faulty classification using EMD-derived feature sets and model comparison.
- Phase 3 is an order-tracking CNN pipeline for deep learning on multi-axis vibration spectra.

The work uses two datasets:

- `MaFaulDa` for the multiclass and deep learning phases.
- `Hasan` dataset for the binary paper-reproduction phase.

## Repository Layout

```text
.
├── 01_baseline_emd/        Phase 1 production code
├── 02_multiclass_xai/      Phase 2 production code
├── 03_ordetracking_cnn/    Phase 3 production code
├── models/                 Final production models
├── playground_models/      Extra and legacy model artifacts
├── figures/                Centralized figures by phase
├── playground/             Old experiments and non-production files
├── src/                    Remaining app and support code
└── data/                   Local datasets (not included here)
```

## Dataset Layout

Expected local dataset structure:

```text
data/
├── raw_mafulda/
│   ├── normal/
│   ├── imbalance/
│   ├── horizontal-misalignment/
│   ├── vertical-misalignment/
│   └── underhang/
│       ├── ball_fault/
│       └── outer_race/
└── paper_data/
    ├── Healthy/
    └── Faulty/
```

Interpretation:

- `raw_mafulda` = MaFaulDa dataset
- `paper_data` = Hasan dataset used in the paper-style reproduction pipeline

## Installation

Use the existing virtual environment or install from `requirements.txt`.

```powershell
pip install -r requirements.txt
```

## How To Run

### Phase 1: Baseline EMD

MaFaulDa multiclass baseline with handcrafted physics-aware features.

```powershell
python 01_baseline_emd\run_phase1.py
```

Outputs:

- Model: `models/01_baseline_emd/`
- Figures: `figures/01_baseline_emd/`
- Extra reports/artifacts: `01_baseline_emd/artifacts/`

### Phase 2: Multiclass XAI / Paper Reproduction

Binary healthy vs faulty paper-reproduction pipeline using the Hasan dataset in `data/paper_data`.

```powershell
python 02_multiclass_xai\train_paper_repro.py
```

Outputs:

- Tables and summaries: `02_multiclass_xai/artifacts/`
- Confusion matrices and plots: `figures/02_multiclass_xai/`
- Final promoted models: `models/02_multiclass_xai/`

### Phase 3: Order-Tracking CNN

Deep learning pipeline on MaFaulDa using order tracking and CNN classification.

```powershell
python 03_ordetracking_cnn\mafaulda_deep.py
```

Outputs:

- Final model weights and preprocessing artifacts: `models/03_ordetracking_cnn/`
- Figures: `figures/03_ordetracking_cnn/`
- Reports and metrics: `03_ordetracking_cnn/artifacts/`

## Phase Summary

### `01_baseline_emd`

- Dataset: MaFaulDa
- Method: EMD + handcrafted features + SVM pipeline
- Focus: interpretable classical baseline

### `02_multiclass_xai`

- Dataset: Hasan `paper_data`
- Method: EMD reconstruction + multiple feature families + model comparison
- Focus: paper reproduction and feature-set benchmarking

### `03_ordetracking_cnn`

- Dataset: MaFaulDa
- Method: order tracking + spectral preprocessing + CNN
- Focus: deep learning production pipeline

## Artifacts Policy

- Final production-ready model files are stored in `models/`.
- Non-final or historical model files are stored in `playground_models/`.
- Centralized figures are stored in `figures/`.
- Old notebooks, exploratory scripts, and legacy copies were moved to `playground/`.

## Notes

- The datasets are local and are not bundled in this repository.
- Phase folder names start with digits for ordering, but the internal Python packages use valid import names.
- Root-level app code such as `src/app/ai_server.py` now points to the new production model location for phase 1.

## Per-Phase READMEs

For phase-specific details:

- [01_baseline_emd/README.md](01_baseline_emd/README.md)
- [02_multiclass_xai/README.md](02_multiclass_xai/README.md)
- [03_ordetracking_cnn/README.md](03_ordetracking_cnn/README.md)
