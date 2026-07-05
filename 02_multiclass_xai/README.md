# Phase 2: Multiclass XAI

This folder contains the paper-reproduction pipeline moved out of `src/training/train_paper_repro.py` and split into modular files.

## Run

From the repository root:

```powershell
python 02_multiclass_xai\train_paper_repro.py
```

## Inputs

- Dataset root: `data/paper_data`

## Outputs

- Fold tables and ranking summary: `02_multiclass_xai/artifacts/`
- Confusion matrices: `figures/02_multiclass_xai/confusion_matrices/`
- Final `.pkl` production artifacts, if any: `models/02_multiclass_xai/`
- Non-production `.pkl` artifacts: `playground_models/02_multiclass_xai/`
