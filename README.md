# ILs-screening-tm

An interactive, modular screening tool for ionic liquid cation generation, structural filtering, synthetic accessibility scoring, and machine learning-driven melting point ($T_m$) prediction.

This repository decouples core chemical and deep learning calculations from the interactive user interface, providing a clean production pipeline suitable for virtual materials discovery.

You can run the entire interactive pipeline directly in your browser without installing anything locally. Click the badge below to launch the pre-configured environment in Google Colab (remember to run the setup cell at the top to initialize the environment and display the interactive UI):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Toucouere-Lorris/ILs-screening-tm/blob/main/tests/test_pipeline_GC.ipynb)

---
![Graphical Abstract](docs/graphical_abstract.svg)
---

## 📂 Repository Structure

The repository is organized into distinct structural layers following software development standards:

```text
Git/
├── ils_screening_tm/          # Core production Python package
│   ├── __init__.py
│   ├── main.py                # Class ILsScreening, Pipeline orchestrator & UI (ipywidgets)
│   │
│   ├── data/                  # Fixed chemical source libraries
│   │   ├── base_cations.csv
│   │   ├── substituents_library.csv
│   │   └── anions_library.csv
│   │
│   └── Models/                # Deep learning weights & feature scalers
│       ├── pscnn_fold_1.keras ... pscnn_fold_5.keras
│       ├── scaler_mordred.pkl
│       └── for-external.pkl
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

## ⚙️ The Fluent Screening Pipeline Workflow

The package orchestrates a sequential pipeline using an object-oriented API with method chaining (`.generation().sascore().tm()`). All calculations modify an internal `self.df` state registry in memory, bypassing unnecessary disk read/write overhead:

### 1. Scaffold Selection & UI Interaction (`.generation()`)
* **Interactive Trigger:** If no scaffold is pre-set, calling this method automatically launches the interactive visual ipywidgets environment to choose a base cation, cut specific atom indices, and configure local anchor symmetries/groups dynamically.
* **Library Expansion:** Generates a library of combinatorially substituted cations using functional group fragments and connection rules.

### 2. Synthetic Accessibility Filtering & Anion Pairing (`.sascore()`)
* Computes SAScores (via RDKit contributions) for every single unique cation.
* Enforces a strict synthesis gate (default `threshold=6.0`) to discard overly complex or unstable chemical entities.
* Automatically triggers a cross-join (product cartesian) with the package's internal anion library to build the complete salt matrix.

### 3. Deep Learning $T_m$ Prediction (`.tm()`)
* Computes 209 mathematical Mordred structural descriptors for both the cation and anion blocks.
* Standardizes feature blocks using pre-trained scalers and feeds data into a 5-Fold Cross-Validation Ensemble of Parallel-Scaffold Convolutional Neural Networks (PSCNN).
* Computes the ensemble average melting point and applies a room-temperature/low-melting screening threshold (default $T_m \le 100^\circ\text{C}$).

### 4. Advanced Diagnostics & Visual Reporting (`.heatmap()` & `.show()`)
* **`.heatmap()`:** Dynamically adapts to the data footprint, plotting a matrix pivot heatmap for smaller runs or switching to an automated violin distribution layout across anion families for large-scale datasets.
* **`.show()`:** Extracts a random sample of the remaining low-melting candidates and renders a 2D high-resolution molecular grid (Cation alongside Anion) inside the notebook layout.

## 🎛️ Interactive Dashboard Features

The screening pipeline comes with an intuitive ipywidgets-based graphical interface that allows you to easily configure your structural modifications and screening constraints in real-time.

### 1. Screening Bounds Settings (Advanced Performance)
At the top of the interface, two numeric entry boxes protect your workstation or server memory against combinatorics explosions:
* **Max Subts:** Controls the maximum number of unique substituents loaded from your library for each active chemical site family.
* **Max Comb:** Sets a hard absolute ceiling for the total number of generated cation structures. If the combinatorics estimation exceeds this number, the generation safely truncates to prevent Out of Memory crashes.

### 2. Scaffold Selection & Core Slicing
* **Cation Core Dropdown:** Select your available base cation scaffolds from the loaded database: Pyrrolidinium, Pyridinium, Piperidinium, Ammonium, Phosphonium, Sulfonium, Imidazolium.
* **Cut Atoms Selector:** Select one or multiple atom indices to dynamically remove them from the structure, opening up new coordination sites for branching.

### 3. Symmetries & Validation
* **Detected Symmetries:** Once an anchor placeholder (`[*]`) is detected on the remaining core structure, dropdown menus automatically appear to configure architectural symmetries and local substitution groups dynamically.
* **Validate & Launch Screening Button:** Automatically triggers the full sequence (`.set_scaffold()`, `.generation()`, `.sascore()`, `.tm()`) using your custom graphic selections.

## 🚀 Execution & Interactive UI

Testing and running the pipeline is containerized inside the `tests/` directory to protect production code from volatile Jupyter execution paths.

### 1. Initializing the Notebook Environment
To test the pipeline locally, open `tests/test_pipeline.ipynb` and execute the following setup block. It includes the `%autoreload` magic extension to seamlessly catch local `main.py` source modifications:

```python
import os
import sys

# Activate Jupyter auto-reload
%load_ext autoreload
%autoreload 2

# Append parent directory to sys.path so Python detects the package locally
sys.path.append(os.path.abspath('..'))

from ils_screening_tm.main import ILsScreening

# Instantiate the pipeline
ils = ILsScreening()

# Launch the interactive interface
ils.generation()
```

### 2. Running a Standalone Python Script (Bypassing UI)
You can entirely bypass the graphical dashboard for non-interactive automated scripts or massive workflows by pre-defining a target atom-mapped scaffold string:

```python
from ils_screening_tm.main import ILsScreening

# Fluent method chaining
ils = ILsScreening()
ils.set_scaffold("C1=[N+]([*:201])CCN1[*:202]") \
   .generation(limit_nb=2) \
   .sascore(threshold=5.5) \
   .tm(threshold_c=80.0) \
   .heatmap(mode='tm') \
   .show(sample_size=4)
```

## 📦 Installation & Local Usage

### Standard Installation (via PyPI)
If you prefer to run the screening pipeline locally on your machine, the package is officially hosted on PyPI. You can install it and all its core dependencies with a single command:

```bash
pip install ils-screening-tm
```

### Local Installation
To install the package in editable development mode (useful if you want to modify the source code), clone the repository and run from the root directory:

```bash
pip install -e .
```
