# ILs-screening-tm

An interactive, modular screening tool for ionic liquid cation generation, structural filtering, synthetic accessibility scoring, and machine learning-driven melting point ($T_m$) prediction.

This repository decouples core chemical and deep learning calculations from the interactive user interface, providing a clean production pipeline suitable for virtual materials discovery.

You can run the entire interactive pipeline directly in your browser without installing anything locally. Click the badge below to launch the pre-configured environment in Google Colab (remember to run the setup cell at the top to initialize the environment and display the interactive UI):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Toucouere-Lorris/ILs-screening-tm/blob/main/tests/test_pipeline.ipynb)
---

## 📂 Repository Structure

The repository is organized into distinct structural layers following software development standards:

```text
Git/
├── ils_screening_tm/          # Core production Python package
│   ├── __init__.py
│   ├── main.py                # Pipeline orchestrator & Interactive UI (ipywidgets)
│   │
│   ├── database/              # Fixed chemical databases
│   │   ├── base_cations.csv
│   │   ├── substituents_library.csv
│   │   └── anions_library.csv
│   │
│   ├── models/                # Deep learning weights & feature scalers
│   │   ├── pscnn_fold_1.keras ... pscnn_fold_5.keras
│   │   ├── scaler_mordred.pkl
│   │   └── for-external.pkl
│   │
│   ├── Generation/            # Step 1: Combinatorial generation engine
│   ├── SAScore/               # Step 2: RDKit accessibility screening & pairing
│   ├── Prediction_tm/         # Step 3: Mordred descriptor & CNN inference
│   └── Display/               # Step 4: Visual analytics & structural rendering
│
├── output/                    # Generated pipeline checkpoints and final datasets
│   ├── generated_cations_raw.csv     # Output from Step 1
│   ├── ionic_liquids_raw_pairs.csv   # Output from Step 2
│   └── ionic_liquids_filtered_tm.csv # Ultimate Target Dataset (Step 3 & 4)
│
├── tests/                     # Interactive testing & Validation notebooks
│   └── test_pipeline.ipynb    # Demo notebook to launch the UI
│
├── training/                  # Research environment for model development
│   ├── dataset/
│   │   └── tm_data.csv        # Curated experimental benchmark training set
│   └── train_tm_model.py      # Dual-Input Parallel-Scaffold CNN training script
│
├── pyproject.toml
└── README.md
```
---

## ⚙️ The 4-Step Screening Pipeline

When executed via the interactive dashboard, the package orchestrates a sequential 4-step pipeline using the `output/` directory to store intermediate results:

### Step 1: Combinatorial Cation Generation (`Generation/`)
- Takes an atom-map-encoded scaffold selected in the UI.
- Applies combinatorial functional group grafting using local connection matrix rules.
- Filters out non-compliant structures based on custom structural SMARTS filters.
- Saves unique generated scaffolds to `output/generated_cations_raw.csv`.

### Step 2: Synthetic Accessibility Filtering & Anion Pairing (`SAScore/`)
- Computes SAScores (via RDKit contributions) for every single unique cation.
- Enforces a strict synthesis gate (<= 6.0) to discard overly complex or unstable chemical entities.
- Performs a cross-join (product cartesian) between surviving cations and the standard anion library.
- Saves fully-paired salt structures alongside their cation SAScore to `output/ionic_liquids_raw_pairs.csv`.

### Step 3: Deep Learning Tm Prediction (`Prediction_tm/`)
- Computes 209 mathematical Mordred structural descriptors for both the cation and anion blocks.
- Standardizes feature blocks using pre-trained scalers.
- Feeds data into a 5-Fold Cross-Validation Ensemble of Parallel-Scaffold Convolutional Neural Networks (PSCNN).
- Computes the ensemble average melting point and applies a room-temperature/low-melting screening threshold (Tm <= 100°C).
- Saves the final screened dataset containing both Tm and SAScore properties to `output/ionic_liquids_filtered_tm.csv`.

### Step 4: Visual Analytics & Reporting (`Display/`)
- Outputs a formatted cross-statistical summary report in the terminal (min, max, average values for both Tm and SAScores).
- Performs a randomized sample extraction from the top stable candidates.
- Renders a 2D high-resolution molecular grid (Cation alongside Anion) inside the notebook layout.

---
## 🚀 Execution & Interactive UI

Testing and running the pipeline is containerized inside the `tests/` directory to protect production code from volatile Jupyter execution paths.

### 1. Launching the Interactive Test Notebook
Navigate to the `tests/` directory and open `test_pipeline.ipynb`. Create a cell with the following block to append the local package and launch the graphical interface:

```python
import os
import sys
import pandas as pd

# Append parent directory to sys.path so Python detects the package locally
sys.path.append(os.path.abspath('..'))

from ils_screening_tm.main import start_screening_interface

# Load the scaffold baseline database
df = pd.read_csv('../ils_screening_tm/database/base_cations.csv')

# Render the interactive dashboard
start_screening_interface(df)
```

### 2. Live Statistics Dashboard Output
Upon execution, a real-time tracking panel will render, enabling live slicing, atom cutting, and symmetry management. Clicking the "Launch Full Pipeline" button runs the 4 steps and outputs the following comprehensive analysis:

```text
==================================================
 📊 SCREENING PIPELINE FINAL SUMMARY
==================================================
 Total stable ionic liquids retained : 295
--------------------------------------------------
 🌡️ Predicted Melting Point (Tm):
   Minimum Predicted Tm              : -7.47°C (265.68 K)
   Maximum Predicted Tm              : 99.62°C (372.77 K)
   Average Predicted Tm              : 44.69°C (317.84 K)
--------------------------------------------------
 🧪 Synthetic Accessibility Score (SAScore):
   Minimum SAScore (Easiest)         : 5.65
   Maximum SAScore (Hardest)         : 5.98
   Average SAScore                   : 5.82
   (Scale: 1 = Very Easy, 10 = Extremely Difficult)
==================================================
```
## 🛠️ Installation

To install the production screening package and its absolute dependencies locally in editable development mode, clone the repository, navigate to the root folder containing `pyproject.toml`, and run:

```bash
pip install -e .
```
