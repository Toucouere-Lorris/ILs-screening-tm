#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Interactive screening tool for ionic liquid cation generation, structural
filtering, synthetic accessibility scoring, and melting point prediction.

Refactored version: pipeline logic is decoupled from the UI, errors are
logged instead of silently swallowed, and edge cases (empty model list,
runaway combinatorics) are guarded explicitly.
"""

import io
import os
import sys
import pickle
import logging
import warnings
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as widgets
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, RDConfig
from mordred import Calculator, descriptors
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message=".*SettingWithCopyWarning.*")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("il_screening")

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
try:
    import sascorer
except ImportError as exc:
    raise ImportError(
        "Unable to load 'sascorer'. Ensure RDKit is properly installed "
        "with contributions (RDContrib)."
    ) from exc

BASE_DIR = Path(__file__).resolve().parent

# --- 1. CONFIGURATION & CONSTANTS ---------------------------------------

LIMIT_NB = 2
MAX_COMBINATIONS = 20_000  # hard ceiling to avoid runaway combinatorics

FORBIDDEN_CONNECTIONS = {
    'N': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S', 'N'],
    'P': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S'],
    'S': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'N'],
    'O': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S', 'N', 'P'],
}
SMARTS_INTERDITS = ["N[CX3]=[O,S]"]

CH_MODELS = os.path.join(BASE_DIR, 'models')
N_MODEL_FOLDS = 5
N_CATION_DESCRIPTORS = 209  # split point between cation / anion descriptor blocks

FICHIER_CATIONS = os.path.join(BASE_DIR, 'database', 'raw_ions', 'base_cations.csv')
FICHIER_LIGANDS = os.path.join(BASE_DIR, 'database', 'raw_ions', 'substituents_library.csv')
FICHIER_ANIONS = os.path.join(BASE_DIR, 'database', 'raw_ions', 'anions_library.csv')

FICHIER_ENTREE = os.path.join(BASE_DIR, 'database', 'generated_cations_raw.csv')
FICHIER_SASCORE = os.path.join(BASE_DIR, 'database', 'generated_cations_sascore.csv')
FICHIER_SORTIE_FUSION = os.path.join(BASE_DIR, 'database', 'ionic_liquids_raw_pairs.csv')
FICHIER_SORTIE_FINALE = os.path.join(BASE_DIR, 'database', 'ionic_liquids_filtered_tm.csv')

SASCORE_THRESHOLD = 6
TM_THRESHOLD_C = 100

resultat_selection = {"smiles": None, "original_index": None}


# --- 2. UTILITY FUNCTIONS (RDKit / display) ------------------------------

def pil_to_widget(pil_image):
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format='PNG')
    return widgets.Image(value=byte_io.getvalue(), format='png', width=350, height=350)


def remove_atoms_interactive(mol, indices):
    """Remove the given atom indices from mol. Returns the original mol
    if removal or sanitization fails."""
    if not indices:
        return mol
    try:
        editable = Chem.EditableMol(mol)
        for idx in sorted(indices, reverse=True):
            editable.RemoveAtom(idx)
        new_mol = editable.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception as exc:
            logger.debug("Sanitization failed after atom removal: %s", exc)
        return new_mol
    except Exception as exc:
        logger.warning("Atom removal failed (indices=%s): %s", indices, exc)
        return mol


def get_labeled_image(mol):
    dopts = Draw.MolDrawOptions()
    dopts.addAtomIndices = True
    dopts.bondLineWidth = 2
    return Draw.MolToImage(mol, size=(400, 400), options=dopts)


# --- 3. FILTERING & COMBINATION FUNCTIONS --------------------------------

def check_final_smarts_filter(mol) -> bool:
    for smarts in SMARTS_INTERDITS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return False
    return True


def get_scaffold_host_atom(mol, type_int: int) -> str:
    for atom in mol.GetAtoms():
        box_num = atom.GetAtomMapNum()
        if box_num > 100 and (box_num // 100) == type_int:
            neighbors = atom.GetNeighbors()
            if neighbors:
                return neighbors[0].GetSymbol()
    return 'C'


def get_ligand_head_atom(pattern: str, typ: str) -> Optional[str]:
    m = Chem.MolFromSmiles(pattern) if typ == 'SMILES' else Chem.MolFromSmarts(pattern)
    if not m:
        return None
    for atom in m.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            if neighbors:
                return neighbors[0].GetSymbol()
    return m.GetAtomWithIdx(0).GetSymbol()


def load_substituants_filtered(substituants_df, ligand_type_str, host_symbol):
    if 'Ligand' not in substituants_df.columns:
        return []
    raw_list = substituants_df[substituants_df['Ligand'] == ligand_type_str][['Type', 'Pattern']].values.tolist()
    forbidden = FORBIDDEN_CONNECTIONS.get(host_symbol, [])

    filtered_list = []
    for item in raw_list:
        head = get_ligand_head_atom(item[1], item[0])
        if head not in forbidden:
            filtered_list.append(item)
            if len(filtered_list) >= LIMIT_NB:
                break
    return filtered_list


def filter_invalid_substituants(df: pd.DataFrame) -> pd.DataFrame:
    valid_rows = []
    for _, row in df.iterrows():
        try:
            m = Chem.MolFromSmiles(row['Pattern']) if row['Type'] == 'SMILES' else Chem.MolFromSmarts(row['Pattern'])
            if m:
                valid_rows.append(row)
            else:
                logger.debug("Invalid substituent pattern dropped: %s", row['Pattern'])
        except Exception as exc:
            logger.debug("Error parsing substituent pattern %s: %s", row.get('Pattern'), exc)
    return pd.DataFrame(valid_rows)


def calculer_sascore_pour_smiles(smiles) -> Optional[float]:
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


@dataclass
class CombinationStats:
    total_generes: int = 0
    exclus_smarts: int = 0
    exclus_erreur: int = 0
    combinatoire_tronquee: bool = False


class CombinationEncoded:
    """Generates substituted cation structures from an atom-map-encoded
    scaffold and a substituent library, applying SMARTS exclusion rules."""

    def __init__(self):
        self.stats = CombinationStats()

    def __call__(self, base_smiles_encoded: str, substituants_df: pd.DataFrame):
        base_mol = Chem.MolFromSmiles(base_smiles_encoded)
        if not base_mol:
            logger.error("Could not parse base scaffold SMILES.")
            return []

        groups_by_type = {"X": set(), "L": set(), "Z": set()}
        type_map = {1: "Z", 2: "X", 3: "L"}

        for atom in base_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 100:
                m_type_int = map_num // 100
                grp_id = map_num % 100
                if m_type_int in type_map:
                    groups_by_type[type_map[m_type_int]].add(grp_id)

        sub_df = filter_invalid_substituants(substituants_df)
        combs_config = {}
        type_to_int = {"Z": 1, "X": 2, "L": 3}

        for m_type in ["X", "L", "Z"]:
            unique_groups = sorted(groups_by_type[m_type])
            if not unique_groups:
                combs_config[m_type] = [()]
                continue
            host = get_scaffold_host_atom(base_mol, type_to_int[m_type])
            subs_list = load_substituants_filtered(sub_df, m_type, host)
            if not subs_list:
                logger.warning("No valid substituents found for site %s (host=%s).", m_type, host)
            combs_config[m_type] = list(product(subs_list, repeat=len(unique_groups))) or [()]

        # Guard against combinatorial explosion before doing any work.
        estimated_total = len(combs_config["X"]) * len(combs_config["L"]) * len(combs_config["Z"])
        if estimated_total > MAX_COMBINATIONS:
            logger.warning(
                "Estimated %d combinations exceeds cap of %d; truncating.",
                estimated_total, MAX_COMBINATIONS,
            )
            self.stats.combinatoire_tronquee = True

        working_mol = Chem.RWMol(base_mol)
        for atom in working_mol.GetAtoms():
            if atom.GetAtomMapNum() > 100:
                atom.SetIsotope(10000 + atom.GetAtomMapNum())
        base_tagged_mol = working_mol.GetMol()

        results = []
        for cX, cL, cZ in product(combs_config["X"], combs_config["L"], combs_config["Z"]):
            self.stats.total_generes += 1
            if self.stats.total_generes > MAX_COMBINATIONS:
                break

            try:
                mol = base_tagged_mol
                legend_parts = []
                tasks = [
                    (2, sorted(groups_by_type["X"]), cX),
                    (3, sorted(groups_by_type["L"]), cL),
                    (1, sorted(groups_by_type["Z"]), cZ),
                ]

                for type_int, unique_groups, chosen_comb in tasks:
                    if not unique_groups:
                        continue
                    for grp_id, (typ, pattern) in zip(unique_groups, chosen_comb):
                        target_iso = 10000 + (type_int * 100) + grp_id
                        sub_mol = Chem.MolFromSmiles(pattern) if typ == 'SMILES' else Chem.MolFromSmarts(pattern)
                        if not sub_mol:
                            logger.debug("Substituent pattern failed to parse: %s", pattern)
                            continue

                        target_idx = next(
                            (at.GetIdx() for at in mol.GetAtoms() if at.GetIsotope() == target_iso),
                            None,
                        )
                        if target_idx is None:
                            continue

                        rw = Chem.RWMol(mol)
                        rw.GetAtomWithIdx(target_idx).SetIsotope(999)
                        res = AllChem.ReplaceSubstructs(
                            rw.GetMol(), Chem.MolFromSmarts("[999*]"), sub_mol,
                            replacementConnectionPoint=0,
                        )
                        if res:
                            mol = res[0]
                            try:
                                Chem.SanitizeMol(mol)
                            except Exception as exc:
                                logger.debug("Sanitization failed during substitution: %s", exc)
                        legend_parts.append(pattern)

                if check_final_smarts_filter(mol):
                    smi = Chem.MolToSmiles(mol)
                    clean_mol = Chem.MolFromSmiles(smi)
                    if clean_mol:
                        for at in clean_mol.GetAtoms():
                            at.SetIsotope(0)
                            at.SetAtomMapNum(0)
                        canonical_smi = Chem.MolToSmiles(clean_mol)
                        results.append((canonical_smi, " + ".join(legend_parts)))
                    else:
                        self.stats.exclus_erreur += 1
                else:
                    self.stats.exclus_smarts += 1
            except Exception as exc:
                logger.debug("Combination generation error: %s", exc)
                self.stats.exclus_erreur += 1

        return results


# --- 4. PIPELINE (UI-independent) ----------------------------------------

@dataclass
class PipelineResult:
    success: bool
    message: str = ""
    df_filtre: Optional[pd.DataFrame] = None
    df_final: Optional[pd.DataFrame] = None
    report_lines: list = field(default_factory=list)


def calculer_descripteurs_df(smiles_series: pd.Series, calc: Calculator, col_descriptors) -> np.ndarray:
    mols = [
        Chem.MolFromSmiles(s) if pd.notna(s) and isinstance(s, str) else None
        for s in smiles_series
    ]
    n_failed = sum(1 for m in mols if m is None)
    if n_failed:
        logger.warning("%d/%d SMILES failed to parse before descriptor calculation.", n_failed, len(mols))
    df_calc = calc.pandas(mols, quiet=True)
    df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_calc[col_descriptors].values


def load_prediction_models(models_dir: str):
    """Loads all available fold models. Returns the list of loaded models,
    or raises if none could be loaded."""
    models = []
    for i in range(1, N_MODEL_FOLDS + 1):
        path = os.path.join(models_dir, f'pscnn_fold_{i}.keras')
        if os.path.exists(path):
            try:
                models.append(load_model(path, compile=False))
            except Exception as exc:
                logger.warning("Failed to load model %s: %s", path, exc)
        else:
            logger.warning("Model file missing: %s", path)
    if not models:
        raise RuntimeError(
            f"No model files could be loaded from '{models_dir}'. "
            f"Expected files like 'pscnn_fold_1.keras' .. 'pscnn_fold_{N_MODEL_FOLDS}.keras'."
        )
    return models


def run_screening_pipeline(final_smiles_encoded: str, df_substituants: pd.DataFrame) -> PipelineResult:
    """Runs the full generation -> filtering -> Tm prediction pipeline.
    Pure function of its inputs; no UI dependency, so it can be unit tested."""

    if df_substituants.empty:
        return PipelineResult(False, "The substituents DataFrame is empty.")

    # 1. Combination + SMARTS structural filtering
    combination = CombinationEncoded()
    substituted_results = combination(final_smiles_encoded, df_substituants)
    if not substituted_results:
        return PipelineResult(False, "No results: everything was filtered out during substitution.")

    nb_cations_generes_initial = combination.stats.total_generes
    nb_apres_smarts = len(substituted_results)

    # Deduplication before SAScore
    df_brut = pd.DataFrame(substituted_results, columns=['SMILES', 'Legend'])
    df_res = df_brut.drop_duplicates(subset=['SMILES'], keep='first').copy()
    nb_exclus_doublons = nb_apres_smarts - len(df_res)

    os.makedirs(os.path.dirname(FICHIER_ENTREE), exist_ok=True)
    df_res.to_csv(FICHIER_ENTREE, index=False)

    # 2. SAScore calculation and filtering
    df_res['SAScore'] = df_res['SMILES'].apply(calculer_sascore_pour_smiles)
    nb_sascore_echec = df_res['SAScore'].isna().sum()
    df_res = df_res.sort_values(by='SAScore', ascending=True, na_position='last')
    df_res.to_csv(FICHIER_SASCORE, index=False)

    df_cations_filtres = df_res[df_res['SAScore'] <= SASCORE_THRESHOLD]
    exclus_sascore = len(df_res) - len(df_cations_filtres) - nb_sascore_echec

    if df_cations_filtres.empty:
        return PipelineResult(False, f"No cations have a SAScore <= {SASCORE_THRESHOLD}. Process terminated.")

    # 3. Cross join with anion library
    if not os.path.exists(FICHIER_ANIONS):
        return PipelineResult(False, f"File '{FICHIER_ANIONS}' not found.")

    df_anions = pd.read_csv(FICHIER_ANIONS)
    df_cations_prep = df_cations_filtres[['SMILES']].rename(columns={'SMILES': 'Cation_SMILES'})
    df_anions_prep = df_anions[['Abbreviation', 'SMILES']].rename(
        columns={'Abbreviation': 'Anion_Name', 'SMILES': 'Anion_SMILES'}
    )
    df_comb = pd.merge(df_cations_prep, df_anions_prep, how='cross')
    df_final = df_comb[['Anion_Name', 'Cation_SMILES', 'Anion_SMILES']].copy()
    nb_combinaisons_anions = len(df_final)
    df_final.to_csv(FICHIER_SORTIE_FUSION, index=False)

    # 4. Descriptor calculation & melting point prediction
    try:
        with open(os.path.join(CH_MODELS, 'for-external.pkl'), 'rb') as f:
            _ = pickle.load(f)
            col_descriptors = pickle.load(f)
        with open(os.path.join(CH_MODELS, 'scaler_mordred.pkl'), 'rb') as f:
            mon_scaler = pickle.load(f)
    except Exception as exc:
        return PipelineResult(False, f"Error loading descriptor/scaler pickle files: {exc}")

    try:
        models = load_prediction_models(CH_MODELS)
    except RuntimeError as exc:
        return PipelineResult(False, str(exc))

    calc = Calculator(descriptors, ignore_3D=True)
    try:
        X_cat_raw = calculer_descripteurs_df(df_final['Cation_SMILES'], calc, col_descriptors)
        X_an_raw = calculer_descripteurs_df(df_final['Anion_SMILES'], calc, col_descriptors)
        X_phys_complet = np.concatenate([X_cat_raw, X_an_raw], axis=1)
        X_phys_std = mon_scaler.transform(X_phys_complet)
        X_cat_final = X_phys_std[:, :N_CATION_DESCRIPTORS]
        X_an_final = X_phys_std[:, N_CATION_DESCRIPTORS:]
    except Exception as exc:
        return PipelineResult(False, f"Descriptor calculation failed: {exc}")

    predictions = []
    for model in models:
        try:
            pred = model.predict([X_cat_final, X_an_final], verbose=0).flatten()
            predictions.append(pred)
        except Exception as exc:
            logger.warning("Prediction failed for one fold model: %s", exc)

    if not predictions:
        return PipelineResult(False, "All fold models failed to produce predictions.")

    predictions_moyennes_K = np.mean(predictions, axis=0)
    df_final.loc[:, 'Predicted_Tm_K'] = predictions_moyennes_K
    df_final.loc[:, 'Predicted_Tm_C'] = predictions_moyennes_K - 273.15

    df_filtre = df_final[df_final['Predicted_Tm_C'] <= TM_THRESHOLD_C]
    exclus_tm = len(df_final) - len(df_filtre)
    df_filtre.to_csv(FICHIER_SORTIE_FINALE, index=False)

    report_lines = [
        "=" * 50,
        "SCREENING & EXCLUSIONS REPORT BY STEP",
        "=" * 50,
        f"Cations initially generated                 : {nb_cations_generes_initial}"
        + (" (capped, combinatorics truncated)" if combination.stats.combinatoire_tronquee else ""),
        f"Excluded by structural rules (SMARTS)        : {combination.stats.exclus_smarts}",
        f"Excluded by generation errors                : {combination.stats.exclus_erreur}",
        f"Cations retained after SMARTS filters        : {nb_apres_smarts}",
        f"Excluded duplicate structures (SMILES)       : {nb_exclus_doublons}",
        f"Unique cations retained for screening        : {len(df_res)}",
        f"Excluded: SAScore calculation failed         : {nb_sascore_echec}",
        f"Excluded by SAScore (> {SASCORE_THRESHOLD})                     : {exclus_sascore}",
        f"Final validated cations (SAScore <= {SASCORE_THRESHOLD})        : {len(df_cations_prep)}",
        "-" * 50,
        f"Combinations made with anions                : {nb_combinaisons_anions}",
        f"Folds used for Tm prediction                 : {len(predictions)}/{N_MODEL_FOLDS}",
        f"Salts excluded by Melting Point (> {TM_THRESHOLD_C} C)       : {exclus_tm}",
        "-" * 50,
        f"Raw combination file saved                   : {FICHIER_SORTIE_FUSION}",
        f"Filtered file (<= {TM_THRESHOLD_C} C) saved                : {FICHIER_SORTIE_FINALE}",
        f"Total valid ionic liquids retained           : {len(df_filtre)}",
        "=" * 50,
    ]

    return PipelineResult(True, "OK", df_filtre=df_filtre, df_final=df_final, report_lines=report_lines)


# --- 5. INTERACTIVE GRAPHICAL INTERFACE -----------------------------------

def demarrer_interface_strict(df_cations: pd.DataFrame, df_substituants: pd.DataFrame):
    if df_cations.empty:
        print("Error: The Cations DataFrame is empty.")
        return

    mol_slider = widgets.IntSlider(value=6, min=0, max=len(df_cations) - 1, description='Mol Index:')
    main_container = widgets.Output()

    def interface_dynamique(change=None):
        main_container.clear_output(wait=True)
        idx = mol_slider.value
        try:
            smiles = df_cations['SMILES'].iloc[idx]
        except Exception as exc:
            logger.error("Could not read cation at index %d: %s", idx, exc)
            return

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if not mol:
            logger.error("Could not parse SMILES at index %d: %s", idx, smiles)
            return

        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        new_selector = widgets.SelectMultiple(
            options=atom_indices, value=(), description='Cut Atoms:', disabled=False, rows=10,
        )
        result_container = widgets.VBox([])

        def on_selection_change(change):
            selected_atoms = change['new']
            new_mol = remove_atoms_interactive(mol, selected_atoms)

            anchors_found = {1: [], 2: [], 3: []}
            for atom in new_mol.GetAtoms():
                raw_type = atom.GetIsotope() or atom.GetAtomMapNum()
                if raw_type in anchors_found:
                    anchors_found[raw_type].append(atom.GetIdx())

            widgets_groupes = [widgets.HTML("<b>3. Detected Symmetries</b>")]
            current_dd_widgets = {}
            has_anchors = False
            type_names = {1: "Z (Z Site)", 2: "X (X Site)", 3: "L (L Site)"}

            for type_code in [2, 3, 1]:
                indices = anchors_found[type_code]
                if not indices:
                    continue
                has_anchors = True
                widgets_groupes.append(widgets.Label(f"--- {type_names[type_code]} : {len(indices)} sites ---"))
                nb_options = min(len(indices), 10)
                opts = list(range(1, max(4, nb_options + 1)))

                for i, at_idx in enumerate(indices):
                    default_val = (i % len(opts)) + 1
                    dd = widgets.Dropdown(
                        options=opts, value=default_val, description=f"Atom {at_idx}:",
                        layout=widgets.Layout(width='180px'), style={'description_width': '80px'},
                    )
                    current_dd_widgets[at_idx] = {'widget': dd, 'type': type_code}
                    widgets_groupes.append(dd)

            btn = widgets.Button(description="Validate & Launch Screening", button_style='success')

            def click_save(b):
                btn.disabled = True
                btn.description = "Processing..."
                btn.button_style = 'warning'

                with main_container:
                    clear_output(wait=True)
                    display(widgets.HTML(
                        "<h3 style='color: #d97706;'>Screening and calculations in progress... "
                        "Please wait. Do not close this tab.</h3>"
                    ))

                    working_mol = Chem.RWMol(new_mol)
                    for at in working_mol.GetAtoms():
                        at.SetIsotope(0)
                        at.SetAtomMapNum(0)

                    for at_idx, data in current_dd_widgets.items():
                        atom = working_mol.GetAtomWithIdx(at_idx)
                        encoded_map = data['type'] * 100 + int(data['widget'].value)
                        atom.SetAtomMapNum(encoded_map)
                        atom.SetIsotope(0)

                    final_smiles_encoded = Chem.MolToSmiles(working_mol)
                    resultat_selection["smiles"] = final_smiles_encoded

                    result = run_screening_pipeline(final_smiles_encoded, df_substituants)

                    clear_output(wait=True)
                    if not result.success:
                        print(f"Screening stopped: {result.message}")
                        btn.disabled = False
                        btn.description = "Validate & Launch Screening"
                        btn.button_style = 'success'
                        return

                    print("\n".join(result.report_lines))

                    df_visu = result.df_filtre if len(result.df_filtre) > 0 else result.df_final
                    nb_echantillon = min(4, len(df_visu))
                    df_sample = df_visu.sample(n=nb_echantillon, random_state=42)

                    mols_a_afficher, legendes_a_afficher = [], []
                    for _, row in df_sample.iterrows():
                        mol_cat = Chem.MolFromSmiles(row['Cation_SMILES'])
                        mol_an = Chem.MolFromSmiles(row['Anion_SMILES'])
                        if mol_cat and mol_an:
                            mols_a_afficher.extend([mol_cat, mol_an])
                            legendes_a_afficher.extend([
                                "Cation",
                                f"Anion: {row['Anion_Name']}",
                            ])

                    if mols_a_afficher:
                        print("\nRandomly sampled Ionic Liquids (Cation next to Anion):")
                        img_grid = Draw.MolsToGridImage(
                            mols_a_afficher, molsPerRow=2, subImgSize=(250, 250), legends=legendes_a_afficher,
                        )
                        display(img_grid)

            btn.on_click(click_save)

            if has_anchors:
                widgets_groupes.append(btn)
            else:
                widgets_groupes.append(widgets.Label("No anchors [*] detected."))

            img_res = pil_to_widget(get_labeled_image(new_mol))
            result_container.children = (
                widgets.Label("Result"), img_res, widgets.VBox(widgets_groupes),
            )

        new_selector.observe(on_selection_change, names='value')
        img_base = pil_to_widget(get_labeled_image(mol))

        with main_container:
            display(widgets.HBox([
                widgets.VBox([widgets.Label("1. Base"), img_base]),
                widgets.VBox([widgets.Label("2. Cut"), new_selector]),
                result_container,
            ]))
            on_selection_change({'new': ()})

    mol_slider.observe(interface_dynamique, names='value')
    display(mol_slider)
    display(main_container)
    interface_dynamique()


# --- 6. DATA LOADING AND SCRIPT ENTRY POINT -------------------------------

def load_required_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to load '%s': %s", path, exc)
        return pd.DataFrame()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dc = load_required_csv(FICHIER_CATIONS)
    df_sub = load_required_csv(FICHIER_LIGANDS)
    demarrer_interface_strict(df_cations=dc, df_substituants=df_sub)


# In[ ]:




