import os
import logging
from pathlib import Path

import pandas as pd
from IPython.display import display as ipy_display
from rdkit import Chem
from rdkit.Chem import Draw

# Setup logging
logger = logging.getLogger("il_screening.display")

# --- CONFIGURATION & PATHS -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # Points to Git/ root

# Input file (the final result of the pipeline)
FICHIER_SORTIE_FINALE = os.path.join(BASE_DIR, 'output', 'ionic_liquids_filtered_tm.csv')


# --- ENTRY POINT FUNCTION --------------------------------------------------

def run_visualization(sample_size: int = 4) -> None:
    """
    Executes Step 4 of the pipeline:
    1. Loads the final filtered ionic liquids dataset.
    2. Prints a summary report of the screened candidates (Tm and SAScore).
    3. Randomly samples and renders a 2D chemical grid of the top candidates (Cation next to Anion).
    """
    logger.info("Executing Step 4: Visual Analytics & Reporting...")

    if not os.path.exists(FICHIER_SORTIE_FINALE):
        raise FileNotFoundError(
            f"Final screened dataset not found at: {FICHIER_SORTIE_FINALE}. "
            f"Please run the prediction step first."
        )

    df_filtered = pd.read_csv(FICHIER_SORTIE_FINALE)

    # 1. Print Summary Statistics
    print("\n" + "=" * 50)
    print(" 📊 SCREENING PIPELINE FINAL SUMMARY")
    print("=" * 50)
    print(f" Total stable ionic liquids retained : {len(df_filtered)}")
    
    if len(df_filtered) > 0:
        print("-" * 50)
        print(" 🌡️ Predicted Melting Point (Tm):")
        print(f"   Minimum Predicted Tm              : {df_filtered['Predicted_Tm_C'].min():.2f}°C ({df_filtered['Predicted_Tm_K'].min():.2f} K)")
        print(f"   Maximum Predicted Tm              : {df_filtered['Predicted_Tm_C'].max():.2f}°C ({df_filtered['Predicted_Tm_K'].max():.2f} K)")
        print(f"   Average Predicted Tm              : {df_filtered['Predicted_Tm_C'].mean():.2f}°C ({df_filtered['Predicted_Tm_K'].mean():.2f} K)")
        
        # Section de statistiques pour le SAScore
        if 'SAScore' in df_filtered.columns:
            print("-" * 50)
            print(" 🧪 Synthetic Accessibility Score (SAScore):")
            print(f"   Minimum SAScore (Easiest)         : {df_filtered['SAScore'].min():.2f}")
            print(f"   Maximum SAScore (Hardest)         : {df_filtered['SAScore'].max():.2f}")
            print(f"   Average SAScore                   : {df_filtered['SAScore'].mean():.2f}")
            print("   (Scale: 1 = Very Easy, 10 = Extremely Difficult)")
            
    print("=" * 50 + "\n")

    if df_filtered.empty:
        logger.warning("The filtered dataset is empty. No structures to display.")
        return

    # 2. Render Molecular Grid
    actual_sample_size = min(sample_size, len(df_filtered))
    df_sample = df_filtered.sample(n=actual_sample_size, random_state=42)

    mols_to_display = []
    legends_to_display = []

    for idx, row in df_sample.iterrows():
        mol_cat = Chem.MolFromSmiles(row['Cation_SMILES'])
        mol_an = Chem.MolFromSmiles(row['Anion_SMILES'])
        
        if mol_cat and mol_an:
            mols_to_display.extend([mol_cat, mol_an])
            
            # Injection dynamique de la valeur SAScore individuelle sous la structure du Cation
            sas_info = f" | SAScore: {row['SAScore']:.1f}" if 'SAScore' in row else ""
            
            # Légende pour le Cation, et chaîne vide pour masquer totalement la légende de l'anion
            legends_to_display.extend([
                f"Candidate {idx} - Cation{sas_info}",
                "", 
            ])

    if mols_to_display:
        print(f"Showing a random sample of {actual_sample_size} screened Ionic Liquid pairs (Cation alongside Anion):")
        
        img_grid = Draw.MolsToGridImage(
            mols_to_display, 
            molsPerRow=2, 
            subImgSize=(300, 300), 
            legends=legends_to_display
        )
        
        ipy_display(img_grid)
    else:
        logger.error("Failed to parse chemical structures for the selected sample.")
