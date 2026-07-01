# ILs-screening-tm

An interactive screening tool for ionic liquid cation generation, structural filtering, synthetic accessibility scoring, and machine learning-driven melting point ($T_m$) prediction.

---

## Repository Structure

The repository is divided into two main parts:
1. **`ils_screening_tm/`**: The core production Python package used to run the interactive molecular generation and screening pipeline.
2. **`training/`**: Research scripts and benchmark datasets used to train and evaluate the Deep Learning models.

---

## Dataset Structure

### 1. Screening Generation Libraries
Located inside the package at `ils_screening_tm/database/raw_ions/`:
- `base_cations.csv`: Core cationic scaffolds used as starting points for the generative library.
- `substituents_library.csv`: List of functional groups and chains to be grafted onto the core scaffolds.
- `anions_library.csv`: Standard library of complementary anions used to pair with the generated cations.

### 2. Model Training Data
Located in the research directory at `training/dataset/`:
- `tm_data.csv`: Curated experimental benchmark dataset used for training, cross-validation, and testing of the deep learning model.

---

## Code & Execution

### 1. Production Pipeline (Screening)
The main screening pipeline is located at `ils_screening_tm/screening_pipeline.py`. It handles:
- Combinatorial generation of new cations.
- Structure-based filtering and SMARTS compliance checking.
- Synthetic accessibility (SA) scoring.
- Ensemble-based $T_m$ prediction using the pre-trained weights stored in `ils_screening_tm/models/`.

### 2. Model Training (Research)
The script `training/train_tm_model.py` contains the complete deep learning pipeline (Parallel-Scaffold CNN architecture) used to retrain the 5-fold ensemble model and export the standard scalers.

---

## Installation & Usage

To install the production screening package and its dependencies locally, clone this repository, navigate to the root directory, and run:

```bash
pip install .
