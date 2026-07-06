


# Phase 1: Baseline EMD

This folder contains the production EMD plus SVM baseline moved out of `src`.

## Run

From the repository root:

```powershell
python 01_baseline_emd\train_paper_repro.py
```

## Inputs

- Dataset root: `data/paper_data`

## Outputs

- Final pipeline: `models/01_baseline_emd/svm_pipeline_physics_validated.pkl`
- Cache files: `playground_models/01_baseline_emd/`
- Generated figures: `figures/01_baseline_emd/`
