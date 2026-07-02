import os
import sys
import logging
from typing import Optional
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig

# Setup logging
logger = logging.getLogger("il_screening.sascore")

# Dynamically load sascorer from RDKit contributions
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError as exc:
    raise ImportError(
        "Unable to load 'sascorer'. Ensure RDKit is properly installed "
        "with contributions (RDContrib)."
    ) from exc

# --- CONFIGURATION & PATHS -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # Points to Git/ root

SASCORE_THRESHOLD = 6.0

# Input files
FICHIER_ENTREE = os.path.join(BASE_DIR, 'output', 'generated_cations_raw.csv')
FICHIER_ANIONS = os.path.join(BASE_DIR, 'ils_screening_tm', 'database', 'anions_library.csv')

# Output file
FICHIER_SORTIE_FUSION = os.path.join(BASE_DIR, 'output', 'ionic_liquids_raw_pairs.csv')


# --- SASCORE UTILITIES -----------------------------------------------------

def compute_sascore_for_smiles(smiles: str) -> Optional[float]:
    """Calculates the Synthetic Accessibility Score (SAScore) for a given SMILES."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return round(sascorer.calculateScore(mol), 2)
        logger.debug("SAScore: could not parse SMILES %s", smiles)
    except Exception as exc:
        logger.debug("SAScore calculation failed for %s: %s", smiles, exc)
    return None


# --- ENTRY POINT FUNCTION --------------------------------------------------

def run_sascore_filtering() -> pd.DataFrame:
    """
    Executes Step 2 of the pipeline:
    1. Loads generated cations from Step 1.
    2. Calculates SAScores and filters out structures with a score > 6.0.
    3. Performs a cross-join with the anion library to generate complete ionic liquid pairs.
    4. Saves the raw pairs to output/.
    """
    logger.info("Executing Step 2: SAScore Filtering & Anion Pairing...")

    # 1. Load generated cations
    if not os.path.exists(FICHIER_ENTREE):
        raise FileNotFoundError(f"Input file from Step 1 not found at: {FICHIER_ENTREE}")
    
    df_cations = pd.read_csv(FICHIER_ENTREE)
    
    # 2. Compute SAScore and filter
    logger.info("Calculating SAScores for unique cations...")
    df_cations['SAScore'] = df_cations['SMILES'].apply(compute_sascore_for_smiles)
    
    # Sort by ease of synthesis (lowest score first)
    df_cations = df_cations.sort_values(by='SAScore', ascending=True, na_position='last')
    
    # Apply threshold filtering
    df_cations_filtered = df_cations[df_cations['SAScore'] <= SASCORE_THRESHOLD].copy()
    
    logger.info(f"Cations surviving SAScore filter (<= {SASCORE_THRESHOLD}): {len(df_cations_filtered)} / {len(df_cations)}")
    
    if df_cations_filtered.empty:
        raise ValueError(f"No cations survived the SAScore filter (<= {SASCORE_THRESHOLD}). Pipeline stopped.")

    # 3. Pair with Anions (Cross Join)
    if not os.path.exists(FICHIER_ANIONS):
        raise FileNotFoundError(f"Anion library database not found at: {FICHIER_ANIONS}")
        
    df_anions = pd.read_csv(FICHIER_ANIONS)
    
    # Prepare dataframes for the cross join
    df_cations_prep = df_cations_filtered[['SMILES', 'SAScore']].rename(columns={'SMILES': 'Cation_SMILES'})
    
    # MODIFICATION HERE: Only extract and rename 'SMILES' since 'Abbreviation' was removed
    df_anions_prep = df_anions[['SMILES']].rename(columns={'SMILES': 'Anion_SMILES'})
    
    # The cross-join operation automatically propagates the SAScore to each generated pair
    df_pairs = pd.merge(df_cations_prep, df_anions_prep, how='cross')
    
    # MODIFICATION HERE: Removed 'Anion_Name' from the final columns footprint
    df_final_pairs = df_pairs[['Cation_SMILES', 'Anion_SMILES', 'SAScore']].copy()
    
    # 4. Save checkpoint to output/
    os.makedirs(os.path.dirname(FICHIER_SORTIE_FUSION), exist_ok=True)
    df_final_pairs.to_csv(FICHIER_SORTIE_FUSION, index=False)
    
    logger.info(f"Step 2 Complete: Generated {len(df_final_pairs)} full ionic liquid combinations saved to {FICHIER_SORTIE_FUSION}")
    return df_final_pairs
