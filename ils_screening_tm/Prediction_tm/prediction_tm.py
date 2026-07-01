import os
import pickle
import logging
import warnings
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from tensorflow.keras.models import load_model

# Mute Keras/TensorFlow warnings for cleaner terminal output
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Setup logging
logger = logging.getLogger("il_screening.prediction_tm")

# --- CONFIGURATION & PATHS -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # Points to Git/ root

N_MODEL_FOLDS = 5
N_CATION_DESCRIPTORS = 209
TM_THRESHOLD_C = 100.0

# Input files
FICHIER_SORTIE_FUSION = os.path.join(BASE_DIR, 'output', 'ionic_liquids_raw_pairs.csv')
CH_MODELS = os.path.join(BASE_DIR, 'ils_screening_tm', 'models')

# Output file
FICHIER_SORTIE_FINALE = os.path.join(BASE_DIR, 'output', 'ionic_liquids_filtered_tm.csv')


# --- PREDICTIVE UTILITIES --------------------------------------------------

def compute_descriptors_df(smiles_series: pd.Series, calc: Calculator, col_descriptors: List[str]) -> np.ndarray:
    """Computes specific Mordred descriptors for a pandas Series of SMILES."""
    mols = [
        Chem.MolFromSmiles(s) if pd.notna(s) and isinstance(s, str) else None
        for s in smiles_series
    ]
    
    n_failed = sum(1 for m in mols if m is None)
    if n_failed:
        logger.warning(f"{n_failed}/{len(mols)} SMILES strings failed to parse before descriptor calculation.")
        
    df_calc = calc.pandas(mols, quiet=True)
    df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_calc[col_descriptors].values


def load_prediction_models(models_dir: str) -> List:
    """Loads all 5 available fold models from the package directory."""
    models = []
    for i in range(1, N_MODEL_FOLDS + 1):
        path = os.path.join(models_dir, f'pscnn_fold_{i}.keras')
        if os.path.exists(path):
            try:
                models.append(load_model(path, compile=False))
            except Exception as exc:
                logger.warning(f"Failed to load model fold {i} at {path}: {exc}")
        else:
            logger.warning(f"Expected model file missing: {path}")
            
    if not models:
        raise RuntimeError(
            f"No Deep Learning model files (.keras) could be loaded from '{models_dir}'. "
            f"Please ensure pscnn_fold_1.keras through pscnn_fold_{N_MODEL_FOLDS}.keras are present."
        )
    return models

# --- ENTRY POINT FUNCTION --------------------------------------------------

def run_tm_prediction() -> pd.DataFrame:
    """
    Executes Step 3 of the pipeline:
    1. Loads the generated salt pairs from Step 2.
    2. Computes 209 Mordred descriptors for both cations and anions.
    3. Standardizes features and runs the 5-fold PSCNN ensemble prediction.
    4. Filters out ionic liquids with predicted Tm > 100°C (373.15 K).
    5. Saves the ultimate screened target dataset to output/.
    """
    logger.info("Executing Step 3: Deep Learning Tm Prediction & Thermal Filtering...")

    # 1. Load paired structures
    if not os.path.exists(FICHIER_SORTIE_FUSION):
        raise FileNotFoundError(f"Input pair file from Step 2 not found at: {FICHIER_SORTIE_FUSION}")
        
    df_final = pd.read_csv(FICHIER_SORTIE_FUSION)
    logger.info(f"Loaded {len(df_final)} candidate ionic liquids for evaluation.")

    # 2. Load scaler and descriptor configuration
    try:
        with open(os.path.join(CH_MODELS, 'for-external.pkl'), 'rb') as f:
            _ = pickle.load(f)
            col_descriptors = pickle.load(f)
            
        with open(os.path.join(CH_MODELS, 'scaler_mordred.pkl'), 'rb') as f:
            my_scaler = pickle.load(f)
    except Exception as exc:
        raise FileNotFoundError(f"Error loading descriptor feature names or scaler pickle files from {CH_MODELS}: {exc}")

    # 3. Compute structural descriptors
    logger.info("Calculating Mordred descriptors for cations and anions (this might take a moment)...")
    calc = Calculator(descriptors, ignore_3D=True)
    
    try:
        X_cat_raw = compute_descriptors_df(df_final['Cation_SMILES'], calc, col_descriptors)
        X_an_raw = compute_descriptors_df(df_final['Anion_SMILES'], calc, col_descriptors)
        
        # Merge matrices and scale
        X_phys_complete = np.concatenate([X_cat_raw, X_an_raw], axis=1)
        X_phys_std = my_scaler.transform(X_phys_complete)
        
        # Split blocks back for the dual-input CNN
        X_cat_final = X_phys_std[:, :N_CATION_DESCRIPTORS]
        X_an_final = X_phys_std[:, N_CATION_DESCRIPTORS:]
    except Exception as exc:
        raise RuntimeError(f"Descriptor computation or matrix transformation failed: {exc}")

    # 4. Load models & predict via Ensemble Average
    models = load_prediction_models(CH_MODELS)
    logger.info(f"Running predictions using an ensemble of {len(models)} fold models...")
    
    predictions = []
    for i, model in enumerate(models, start=1):
        try:
            pred = model.predict([X_cat_final, X_an_final], verbose=0).flatten()
            predictions.append(pred)
        except Exception as exc:
            logger.warning(f"Model fold {i} failed to evaluate the dataset: {exc}")

    if not predictions:
        raise RuntimeError("All available model folds failed to generate predictions.")

    # Average out the predictions from cross-validation folds
    mean_predictions_K = np.mean(predictions, axis=0)
    
    df_final['Predicted_Tm_K'] = mean_predictions_K
    df_final['Predicted_Tm_C'] = mean_predictions_K - 273.15

    # 5. Apply temperature screening gate (Tm <= 100°C)
    # MODIFICATION ICI : On s'assure de faire une copie propre du filtre sur df_final,
    # qui contient déjà naturellement la colonne SAScore héritée du fichier de l'étape 2.
    df_filtered = df_final[df_final['Predicted_Tm_C'] <= TM_THRESHOLD_C].copy()
    excluded_count = len(df_final) - len(df_filtered)
    
    logger.info(f"Thermal filter applied: Excluded {excluded_count} salts with Tm > {TM_THRESHOLD_C}°C.")
    logger.info(f"Total valid low-melting ionic liquids retained: {len(df_filtered)}")

    # 6. Save final checkpoint file
    os.makedirs(os.path.dirname(FICHIER_SORTIE_FINALE), exist_ok=True)
    df_filtered.to_csv(FICHIER_SORTIE_FINALE, index=False)
    
    logger.info(f"Step 3 Complete: Final target database successfully saved to {FICHIER_SORTIE_FINALE}")
    return df_filtered
