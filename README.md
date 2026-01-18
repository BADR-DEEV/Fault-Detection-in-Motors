# Vibration-Based Fault Detection using EMD and Machine Learning

## 📌 Overview
This project implements a machine learning pipeline to detect faults in rotating machines using vibration sensor data. The system utilizes **Empirical Mode Decomposition (EMD)** to break down non-stationary vibration signals into Intrinsic Mode Functions (IMFs).

From these IMFs, a hybrid set of **Time-Domain** and **Spectral** features are extracted to train classifiers (SVM, KNN, LDA) to distinguish between "Healthy" and "Faulty" states. This repository is designed for research collaboration and modular experimentation.

## 📂 Dataset Setup
The project relies on the **Vibration Faults Dataset for Rotating Machines**.

**1. Download the Dataset:**
[Kaggle: Vibration Faults Dataset for Rotating Machines](https://www.kaggle.com/datasets/sumairaziz/vibration-faults-dataset-for-rotating-machines)

**2. Directory Structure:**
To ensure the code runs without path errors, please organize your directories exactly as follows relative to the code files. The code expects the dataset to be in a folder one level up (`../Dataset`) or modified in `load_data.py`.

## Updates are available in our Notion page:
https://www.notion.so/2b9caed544b780619e39f8c00c7fa335

```text
Project_Root/
├── Dataset/                 <-- Downloaded data folder
│   ├── Healthy/             <-- Contains .mat files for healthy data
│   └── Faulty/              <-- Contains .mat files for faulty data
├── src/                     <-- Your python scripts
│   ├── main.py              (Main execution script)
│   ├── load_data.py         (Data loading logic)
│   └── emd_imfs.py          (EMD signal processing)
├── Results/                 <-- Generated automatically on run
├── requirements.txt
└── README.md