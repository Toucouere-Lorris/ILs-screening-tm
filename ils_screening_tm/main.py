import io
import os
import sys
import pickle
import logging
import warnings
from itertools import product
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# --- GLOBALS & STATIC UTILITIES -------------------------------------------

FORBIDDEN_CONNECTIONS = {
    'N': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S', 'N'],
    'P': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S'],
    'S': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'N'],
    'O': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S', 'N', 'P'],
}
SMARTS_INTERDITS = ["N[CX3]=[O,S]"]
N_MODEL_FOLDS = 5


def pil_to_widget(pil_image):
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format='PNG')
    return widgets.Image(value=byte_io.getvalue(), format='png', width=350, height=350)


def remove_atoms_interactive(mol, indices):
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
            logger.debug("Sanitization failed after atom removal (kept unsanitized mol): %s", exc)
        return new_mol
    except Exception as exc:
        logger.warning("Atom removal failed: %s", exc)
        return mol


def get_labeled_image(mol):
    dopts = Draw.MolDrawOptions()
    dopts.addAtomIndices = True
    dopts.bondLineWidth = 2
    return Draw.MolToImage(mol, size=(400, 400), options=dopts)


def calculer_sascore_pour_smiles(smiles) -> Optional[float]:
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return round(sascorer.calculateScore(mol), 2)
    except Exception as exc:
        logger.debug("SAScore computation failed for '%s': %s", smiles, exc)
    return None


def calculer_descripteurs_df(smiles_series: pd.Series, calc: Calculator, col_descriptors) -> np.ndarray:
    mols = [Chem.MolFromSmiles(s) if pd.notna(s) and isinstance(s, str) else None for s in smiles_series]
    df_calc = calc.pandas(mols, quiet=True)
    df_calc = df_calc.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_calc[col_descriptors].values


def load_prediction_models(models_dir: str) -> List:
    models = []
    for i in range(1, N_MODEL_FOLDS + 1):
        path = os.path.join(models_dir, f'pscnn_fold_{i}.keras')
        if os.path.exists(path):
            try:
                models.append(load_model(path, compile=False))
            except Exception as exc:
                logger.warning("Failed to load fold model %s: %s", path, exc)
    if not models:
        raise RuntimeError(f"No fold models could be loaded from '{models_dir}'.")
    return models


@dataclass
class CombinationStats:
    total_generes: int = 0
    exclus_smarts: int = 0
    exclus_erreur: int = 0
    combinatoire_tronquee: bool = False


class CombinationEncoded:
    """Generates functionalized cations by substituting tagged anchor points
    (X, L, Z) on a scaffold with fragments drawn from a substituent library.

    Substituent compatibility (FORBIDDEN_CONNECTIONS) is now evaluated
    per individual anchor site rather than per anchor *type*, since two
    sites of the same type (e.g. two 'X' anchors) can sit on chemically
    different host atoms.
    """

    def __init__(self):
        self.stats = CombinationStats()
        

    def __call__(self, base_smiles_encoded: str, substituants_df: pd.DataFrame, limit_nb: int, max_combinations: int):
        base_mol = Chem.MolFromSmiles(base_smiles_encoded)
        if not base_mol:
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

        def check_final_smarts_filter(m) -> bool:
            for smarts in SMARTS_INTERDITS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and m.HasSubstructMatch(pattern):
                    return False
            return True

        def get_site_host_atom(m, t_int: int, grp_id: int) -> str:
            """Host atom symbol for one specific anchor site (type + group id),
            rather than for the whole type. This lets each site apply its own
            FORBIDDEN_CONNECTIONS filter."""
            target_map_num = t_int * 100 + grp_id
            for at in m.GetAtoms():
                if at.GetAtomMapNum() == target_map_num:
                    neighbors = at.GetNeighbors()
                    if neighbors:
                        return neighbors[0].GetSymbol()
            return 'C'

        def get_ligand_head_atom(pattern: str, typ: str) -> Optional[str]:
            m_lig = Chem.MolFromSmiles(pattern) if typ == 'SMILES' else Chem.MolFromSmarts(pattern)
            if not m_lig:
                return None
            for at in m_lig.GetAtoms():
                if at.GetSymbol() == '*':
                    neighbors = at.GetNeighbors()
                    if neighbors:
                        return neighbors[0].GetSymbol()
            return m_lig.GetAtomWithIdx(0).GetSymbol()

        def is_valid_pattern(row) -> bool:
            try:
                m_test = Chem.MolFromSmiles(row['Pattern']) if row['Type'] == 'SMILES' else Chem.MolFromSmarts(row['Pattern'])
                return m_test is not None
            except Exception as exc:
                logger.debug("Invalid substituent pattern '%s': %s", row.get('Pattern'), exc)
                return False

        # Vectorized validity mask instead of a manual iterrows loop.
        valid_mask = substituants_df.apply(is_valid_pattern, axis=1)
        sub_df = substituants_df[valid_mask].copy()

        type_to_int = {"Z": 1, "X": 2, "L": 3}
        combs_config = {}

        for m_type in ["X", "L", "Z"]:
            unique_groups = sorted(groups_by_type[m_type])
            if not unique_groups:
                combs_config[m_type] = [()]
                continue

            if 'Ligand' in sub_df.columns:
                candidates = sub_df[sub_df['Ligand'] == m_type][['Type', 'Pattern']].values.tolist()
            else:
                candidates = []

            site_options: List[List[Tuple[str, str]]] = []
            for grp_id in unique_groups:
                host = get_site_host_atom(base_mol, type_to_int[m_type], grp_id)
                forbidden = FORBIDDEN_CONNECTIONS.get(host, [])
                subs_for_site = []
                for item in candidates:
                    head = get_ligand_head_atom(item[1], item[0])
                    if head not in forbidden:
                        subs_for_site.append(tuple(item))
                        if len(subs_for_site) >= limit_nb:
                            break
                if not subs_for_site:
                    logger.warning(
                        "No compatible substituents found for %s site (host atom '%s'); "
                        "this site cannot be functionalized.", m_type, host
                    )
                site_options.append(subs_for_site)

            if all(site_options):
                combs_config[m_type] = list(product(*site_options))
            else:
                combs_config[m_type] = []

        estimated_total = len(combs_config["X"]) * len(combs_config["L"]) * len(combs_config["Z"])
        if estimated_total == 0:
            logger.warning("No valid combinations can be generated for this scaffold/substituent set.")
        elif estimated_total > max_combinations:
            logger.warning(
                "Estimated %d combinations exceeds cap of %d; generation will stop early once the cap is reached.",
                estimated_total, max_combinations,
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
            if self.stats.total_generes > max_combinations:
                self.stats.combinatoire_tronquee = True
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
                            continue

                        target_idx = next((at.GetIdx() for at in mol.GetAtoms() if at.GetIsotope() == target_iso), None)
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
                                logger.debug(
                                    "Sanitization failed mid-substitution (grp %s, pattern '%s'): %s",
                                    grp_id, pattern, exc,
                                )
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
                        logger.debug("Discarded structure: final SMILES failed to re-parse ('%s').", smi)
                else:
                    self.stats.exclus_smarts += 1
            except Exception as exc:
                self.stats.exclus_erreur += 1
                logger.debug("Discarded structure due to exception during substitution: %s", exc)

        return results


# --- 1. THE MAIN SCREENING CLASS ------------------------------------------

class ILsScreening:
    def __init__(self):
        """Initializes internal active database storage configurations."""
        self.df: Optional[pd.DataFrame] = None
        self.encoded_smiles: Optional[str] = None

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Configuration des chemins absolus stables pour éviter les FileNotFoundError
        self.fichier_cations = os.path.join(base_dir, 'data', 'base_cations.csv')
        self.fichier_substituants = os.path.join(base_dir, 'data', 'substituents_library.csv')
        self.fichier_anions = os.path.join(base_dir, 'data', 'anions_library.csv')
        self.ch_models = os.path.join(base_dir, 'Models')

    def __repr__(self) -> str:
        """Custom clean string representation for Jupyter notebooks display."""
        if self.df is None:
            return "ILsScreening Pipeline (Status: Empty Sandbox)"

        # Détection dynamique de l'état d'avancement de la base
        if 'Predicted_Tm_C' in self.df.columns:
            status = f"Screened Library ({len(self.df)} salts with predicted Tm)"
        elif 'Anion_SMILES' in self.df.columns:
            status = f"Paired Library ({len(self.df)} salt configurations)"
        else:
            status = f"Generated Cations Registry ({len(self.df)} structures)"

        return f"ILsScreening Pipeline (Status: {status})"

    def set_scaffold(self, smiles: str) -> "ILsScreening":
        """Sets the active molecular scaffold target structure string."""
        self.encoded_smiles = smiles
        print(f"🎯 Scaffold structure explicitly set to: {self.encoded_smiles}")
        return self

    def generation(self, limit_nb: int = 2, max_combinations: int = 20000) -> "ILsScreening":
        """Step 1: Generates combinatorially branched unique cations.

        Automatically resets previous screening state to allow safe reruns.
        If no scaffold is predefined, triggers the visual interactive UI.

        Note: when the interactive UI path is used (no scaffold set), this
        method returns immediately and `self.df` stays `None` until the
        user finishes the widget interaction and clicks "Validate & Launch
        Screening". Chaining `.generation().sascore()` in a non-notebook
        script without a scaffold set will therefore raise a ValueError in
        `.sascore()` rather than silently doing nothing.
        """
        # --- RESET STRATEGIC ENTRY POINT ---
        # Clears any existing dataframe memory from previous pipeline runs
        self.df = None

        # 🛡️ SÉCURITÉ ABSOLUE DIRECTE : Recalcul forcé des chemins absolus au moment de l'appel
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.fichier_cations = os.path.join(base_dir, 'data', 'base_cations.csv')
        self.fichier_substituants = os.path.join(base_dir, 'data', 'substituents_library.csv')
        self.fichier_anions = os.path.join(base_dir, 'data', 'anions_library.csv')
        self.ch_models = os.path.join(base_dir, 'Models')

        if not self.encoded_smiles:
            print("💡 No scaffold layout defined. Launching interactive Widget interface...")
            try:
                dc_cations = pd.read_csv(self.fichier_cations)
                df_substituants = pd.read_csv(self.fichier_substituants)
                demarrer_interface_strict(df_cations=dc_cations, df_substituants=df_substituants, instance_screening=self)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"❌ Impossible to launch UI, required local files missing: {e}")
            return self

        if not os.path.exists(self.fichier_substituants):
            raise FileNotFoundError(f"❌ Substituent library file missing at {self.fichier_substituants}")

        df_sub = pd.read_csv(self.fichier_substituants)
        engine = CombinationEncoded()
        raw_structures = engine(self.encoded_smiles, df_sub, limit_nb, max_combinations)

        df_brut = pd.DataFrame(raw_structures, columns=['SMILES', 'Legend'])
        self.df = df_brut.drop_duplicates(subset=['SMILES'], keep='first').copy()

        print(f"🧬 [Generation] Generated {len(self.df)} unique functionalized cations.")
        return self

    def sascore(self, threshold: float = 6.0) -> "ILsScreening":
        """Step 2: Evaluates structural ease of synthesis using standard RDKit SAScores."""
        if self.df is None or 'SMILES' not in self.df.columns:
            raise ValueError("❌ No active cation registry located. Execute .generation() first.")

        print("🧪 [SAScore] Calculating chemical accessibility vectors...")
        self.df['SAScore'] = self.df['SMILES'].apply(calculer_sascore_pour_smiles)
        self.df = self.df.sort_values(by='SAScore', ascending=True, na_position='last')

        before_count = len(self.df)
        self.df = self.df[self.df['SAScore'] <= threshold].copy()
        print(f"✂️ [SAScore Filter] {len(self.df)} / {before_count} cations retained under threshold (<= {threshold}).")
        return self

    def pair_with_anions(self) -> "ILsScreening":
        """Intermediate Core Step: Couples survived cations matrices against the raw anion dataset.
        Enables fluent chaining and standalone pairing analysis.
        """
        if self.df is None or ('SMILES' not in self.df.columns and 'Cation_SMILES' not in self.df.columns):
            raise ValueError("❌ Cation dataframe sequence empty or missing. Run .generation() first.")

        # Si les molécules sont déjà associées aux anions, on ne fait rien pour éviter de dupliquer la matrice
        if 'Cation_SMILES' in self.df.columns and 'Anion_SMILES' in self.df.columns:
            return self

        if not os.path.exists(self.fichier_anions):
            raise FileNotFoundError(f"❌ Anions source library missing at {self.fichier_anions}")

        df_anions = pd.read_csv(self.fichier_anions)

        # On prépare le dataframe des cations existants (en gardant le SAScore s'il a été calculé)
        cols_to_keep = ['SMILES']
        if 'SAScore' in self.df.columns:
            cols_to_keep.append('SAScore')

        df_cat_prep = self.df[cols_to_keep].rename(columns={'SMILES': 'Cation_SMILES'})
        df_an_prep = df_anions[['SMILES']].rename(columns={'SMILES': 'Anion_SMILES'})

        # Produit cartésien (Cross Join) pour créer toutes les combinaisons possibles de sels
        self.df = pd.merge(df_cat_prep, df_an_prep, how='cross')
        print(f"🔗 [Anion Pairing] Matrix built: Generated {len(self.df)} full salt configurations.")

        return self  # <-- TRÈS IMPORTANT : Permet le chaînage fluide

    def tm(self, threshold_c: float = 100.0) -> "ILsScreening":
        """Step 3: Extracts molecular descriptors and returns deep learning melting points predictions."""
        if self.df is not None and 'Cation_SMILES' not in self.df.columns:
            self.pair_with_anions()

        if self.df is None or self.df.empty:
            raise ValueError("❌ No structured matrix ready to receive prediction tensors.")

        print("Tm Prediction running... Extracting Mordred vectors and calling fold layers...")
        try:
            with open(os.path.join(self.ch_models, 'for-external.pkl'), 'rb') as f:
                _ = pickle.load(f)
                col_descriptors = pickle.load(f)
            with open(os.path.join(self.ch_models, 'scaler_mordred.pkl'), 'rb') as f:
                mon_scaler = pickle.load(f)
            models = load_prediction_models(self.ch_models)
        except Exception as exc:
            raise RuntimeError(f"❌ Critical error loading external neural network assets: {exc}")

        calc = Calculator(descriptors, ignore_3D=True)
        X_cat_raw = calculer_descripteurs_df(self.df['Cation_SMILES'], calc, col_descriptors)
        X_an_raw = calculer_descripteurs_df(self.df['Anion_SMILES'], calc, col_descriptors)

        X_phys_complet = np.concatenate([X_cat_raw, X_an_raw], axis=1)
        X_phys_std = mon_scaler.transform(X_phys_complet)

        # Use the actual descriptor count from the loaded pickle instead of a
        # hardcoded literal, so the slicing stays correct if the descriptor
        # set used to train the scaler/models ever changes.
        n_desc = len(col_descriptors)
        X_cat_final = X_phys_std[:, :n_desc]
        X_an_final = X_phys_std[:, n_desc:]

        predictions = [model.predict([X_cat_final, X_an_final], verbose=0).flatten() for model in models]
        predictions_moyennes_K = np.mean(predictions, axis=0)

        self.df['Predicted_Tm_C'] = predictions_moyennes_K - 273.15

        before_count = len(self.df)
        self.df = self.df[self.df['Predicted_Tm_C'] <= threshold_c].copy()
        print(f"📉 [Tm Filter] {len(self.df)} / {before_count} liquid salts survived ceiling check (<= {threshold_c}°C).")
        return self

    def plot(self, prop: str = 'auto', kind: str = 'hist', max_display: int = 50, save_path: str = None) -> "ILsScreening":
        """
        Generates diagnostic plots for the screening workflow.

        Parameters:
        -----------
        prop : str, default 'auto'
            The chemical or physical property to analyze ('sascore' or 'tm').
        kind : str, default 'hist'
            The visualization style:
            - 'hist'      : Statistical distribution of the property.
            - 'matrix'    : Heatmap grid of the property values (requires ILs).
            - 'similarity': Tanimoto structural similarity matrix.
        max_display : int, default 50
            Maximum number of molecules to display in matrix or similarity plots.
        save_path : str, default None
            If provided, saves the figure to the specified path (e.g., 'plot.png').
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        if self.df is None or self.df.empty:
            raise ValueError("❌ Database is empty.")

        prop = prop.lower()
        kind = kind.lower()
        smiles_col = 'SMILES' if 'SMILES' in self.df.columns else 'Cation_SMILES'
        
        if prop == 'auto':
            prop = 'tm' if 'Predicted_Tm_C' in self.df.columns else 'sascore'

        # Helper to handle saving
        def finalize_plot(path):
            if path:
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"💾 [Plot] Figure saved to {path}")
            plt.show()

        # --- KIND: HISTOGRAM (Statistical Distribution) ---
        if kind == 'hist':
            col_to_plot = 'SAScore' if prop == 'sascore' else 'Predicted_Tm_C'
            if col_to_plot not in self.df.columns:
                raise ValueError(f"❌ {col_to_plot} missing. Run .{prop}() first.")
            
            plt.figure(figsize=(10, 5))
            sns.histplot(data=self.df, x=col_to_plot, kde=True, color="teal", bins=20)
            plt.title(f"Distribution of {col_to_plot}")
            finalize_plot(save_path)

        # --- KIND: MATRIX (Value Grid Heatmap) ---
        elif kind == 'matrix':
            if 'Anion_SMILES' in self.df.columns:
                # Map unique cations to an index for cleaner visualization
                unique_cations = self.df[smiles_col].unique()
                cat_to_idx = {smile: i for i, smile in enumerate(unique_cations)}
                
                plot_df = self.df.copy()
                plot_df['Cation_Index'] = plot_df[smiles_col].map(cat_to_idx)
                
                # Filter to display only the requested number of cations
                if len(unique_cations) > max_display:
                    print(f"📈 [Plot] Library contains {len(unique_cations)} unique cations. Displaying first {max_display}.")
                    plot_df = plot_df[plot_df['Cation_Index'] < max_display]
                
                matrix = plot_df.pivot_table(
                    index='Cation_Index', 
                    columns='Anion_SMILES', 
                    values='SAScore' if prop == 'sascore' else 'Predicted_Tm_C'
                )
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(matrix, cmap="YlOrRd" if prop == 'sascore' else "coolwarm", annot=False)
                plt.title(f"Property Matrix: {prop.upper()} (Cation Indices)")
                plt.xlabel("Anion SMILES")
                plt.ylabel("Cation Index")
                finalize_plot(save_path)
            else:
                print("⚠️ No Anions found. Matrix kind requires Ionic Liquids (Cation + Anion).")

        # --- KIND: SIMILARITY (Tanimoto Structural Similarity) ---
        elif kind == 'similarity':
            unique_cats = self.df.drop_duplicates(subset=[smiles_col]).head(max_display)
            mols = [Chem.MolFromSmiles(s) for s in unique_cats[smiles_col]]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols if m is not None]
            
            sim_matrix = np.zeros((len(fps), len(fps)))
            for i in range(len(fps)):
                for j in range(len(fps)):
                    sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

            plt.figure(figsize=(10, 8))
            # Added extent and labels to match the index-style matrix plot
            sns.heatmap(sim_matrix, cmap="viridis", annot=False, xticklabels=range(len(fps)), yticklabels=range(len(fps)))
            plt.title("Structural Similarity Matrix (Tanimoto Coefficients)")
            finalize_plot(save_path)

    def show(self, sample_size: int = 4, random_state: Optional[int] = None) -> "ILsScreening":
        """Displays high-quality RDKit molecular grids without the pandas DataFrame table.

        Automatically adapts whether to display standalone cations or cation/anion pairs.

        Parameters:
        -----------
        sample_size : int, default 4
            Number of random rows to display.
        random_state : Optional[int], default None
            Seed for the sampling. Left as None by default so repeated calls
            show different candidates (matches original behavior); pass an
            int for reproducible output, e.g. during debugging.
        """
        if self.df is None or self.df.empty:
            print("⚠️ Current active datastore sandbox is empty.")
            return self

        nb_echantillon = min(sample_size, len(self.df))
        df_sample = self.df.sample(n=nb_echantillon, random_state=random_state)

        mols, legends = [], []

        # Cas 1 : Après appariement (Colonnes Cation_SMILES et Anion_SMILES présentes)
        if 'Cation_SMILES' in self.df.columns and 'Anion_SMILES' in self.df.columns:
            print(f"🎨 Displaying {nb_echantillon} random Screened Salt Pairs (Left: Cation | Right: Anion):")
            for idx, row in df_sample.iterrows():
                mol_cat = Chem.MolFromSmiles(row['Cation_SMILES'])
                mol_an = Chem.MolFromSmiles(row['Anion_SMILES'])
                if mol_cat and mol_an:
                    mols.extend([mol_cat, mol_an])
                    sas_info = f" | SAS: {row['SAScore']:.1f}" if 'SAScore' in row else ""
                    tm_info = f" | Tm: {row['Predicted_Tm_C']:.1f}°C" if 'Predicted_Tm_C' in row else ""
                    legends.extend([f"Cand {idx} - Cation{sas_info}{tm_info}", f"Cand {idx} - Anion"])

            if mols:
                # 2 molécules par ligne = 1 paire complète (Cation + Anion) par ligne de la grille
                grid = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(300, 300), legends=legends)
                display(grid)

        # Cas 2 : Avant appariement (Seule la colonne SMILES des cations générés est présente)
        elif 'SMILES' in self.df.columns:
            print(f"🎨 Displaying {nb_echantillon} random generated Cations structures:")
            for idx, row in df_sample.iterrows():
                mol_cat = Chem.MolFromSmiles(row['SMILES'])
                if mol_cat:
                    mols.append(mol_cat)
                    sas_info = f" | SAS: {row['SAScore']:.1f}" if 'SAScore' in row else ""
                    legends.append(f"Cation {idx}{sas_info}")

            if mols:
                # Pour les cations seuls, on peut les afficher sur 3 colonnes pour économiser de l'espace
                grid = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(250, 250), legends=legends)
                display(grid)

        return self

    def save(self, filename: str, columns: list = None):
        """
        Saves the current internal dataframe state (self.df) to a file.
        Supports both CSV (.csv) and Excel (.xlsx) formats based on the file extension.

        Parameters:
        -----------
        filename : str
            The target file path (e.g., 'output_results.csv' or 'data/screened_ils.xlsx').
        columns : list, optional
            A specific list of columns to export. If None, exports everything available.
            Examples of useful columns:
            - After .generation(): ['cation_smiles']
            - After .sascore():    ['cation_smiles', 'anion_smiles', 'sascore']
            - After .tm():         ['cation_smiles', 'anion_smiles', 'sascore', 'predicted_tm']

        Returns:
        --------
        self : ILsScreening
            Returns self to preserve the fluent method chaining syntax.
        """
        if self.df is None or self.df.empty:
            print("⚠ Warning: The tracking dataframe is empty or has not been initialized yet. Nothing to save.")
            return self

        # Determine which columns to save
        df_to_save = self.df[columns] if columns is not None else self.df

        # Check format based on file extension
        if filename.endswith('.csv'):
            df_to_save.to_csv(filename, index=False)
            print(f"✓ Successfully exported {len(df_to_save)} rows to CSV: '{filename}'")
        elif filename.endswith('.xlsx'):
            try:
                # openpyxl is required for pandas excel export
                df_to_save.to_excel(filename, index=False, engine='openpyxl')
                print(f"✓ Successfully exported {len(df_to_save)} rows to Excel: '{filename}'")
            except ImportError:
                print("❌ Error: 'openpyxl' package is missing. Please run 'pip install openpyxl' to export to Excel.")
        else:
            print(f"❌ Error: Unsupported file format for '{filename}'. Please use a '.csv' or '.xlsx' extension.")

        return self


def demarrer_interface_strict(df_cations: pd.DataFrame, df_substituants: pd.DataFrame, instance_screening: Optional[ILsScreening] = None):
    """Fallback interactive UI environment anchor using widgets."""
    if df_cations.empty:
        print("Error: The Cations DataFrame is empty.")
        return

    # Configuration des options du menu déroulant (Nom à afficher: Index réel du DataFrame)
    familles_options = [
        ('Pyrrolidinium', 0),
        ('Pyridinium', 1),
        ('Piperidinium', 2),
        ('Ammonium', 3),
        ('Phosphonium', 4),
        ('Sulfonium', 5),
        ('Imidazolium', 6)
    ]

    limit_nb_widget = widgets.IntText(value=2, description='Max Subts:')
    max_comb_widget = widgets.IntText(value=20000, description='Max Comb:')
    config_panel = widgets.HBox([limit_nb_widget, max_comb_widget], layout=widgets.Layout(margin='0px 0px 15px 0px', padding='10px', border='1px dashed #ccc'))

    mol_dropdown = widgets.Dropdown(options=familles_options, value=0, description='Cation Core:')
    main_container = widgets.Output()

    def interface_dynamique(change=None):
        main_container.clear_output(wait=True)
        idx = mol_dropdown.value
        try:
            smiles = df_cations['SMILES'].iloc[idx]
        except Exception as exc:
            logger.warning("Could not retrieve scaffold SMILES for dropdown index %s: %s", idx, exc)
            return

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if not mol:
            return

        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        new_selector = widgets.SelectMultiple(options=atom_indices, value=(), description='Cut Atoms:', rows=10)
        result_container = widgets.VBox([])

        def on_selection_change(change_event):
            selected_atoms = change_event['new']
            new_mol = remove_atoms_interactive(mol, selected_atoms)
            anchors_found = {1: [], 2: [], 3: []}
            for atom in new_mol.GetAtoms():
                raw_type = atom.GetIsotope() or atom.GetAtomMapNum()
                
                # Si c'est un grand tag (ex: 201, 301), on prend la centaine
                if raw_type > 100:
                    raw_type = raw_type // 100
                    
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
                opts = list(range(1, max(4, min(len(indices), 10) + 1)))

                for i, at_idx in enumerate(indices):
                    dd = widgets.Dropdown(options=opts, value=(i % len(opts)) + 1, description=f"Atom {at_idx}:", layout=widgets.Layout(width='180px'))
                    current_dd_widgets[at_idx] = {'widget': dd, 'type': type_code}
                    widgets_groupes.append(dd)

            btn = widgets.Button(description="Validate & Launch Screening", button_style='success')

            def click_save(b):
                btn.disabled = True
                with main_container:
                    clear_output(wait=True)
                    display(widgets.HTML("<h3 style='color: #0284c7;'>🧬 Cation generation in progress from UI selection...</h3>"))
                    working_mol = Chem.RWMol(new_mol)
                    for at in working_mol.GetAtoms():
                        at.SetIsotope(0)
                        at.SetAtomMapNum(0)
                    for at_idx, data in current_dd_widgets.items():
                        atom = working_mol.GetAtomWithIdx(at_idx)
                        atom.SetAtomMapNum(data['type'] * 100 + int(data['widget'].value))

                    final_smi = Chem.MolToSmiles(working_mol)

                    scr = instance_screening if instance_screening is not None else ILsScreening()
                    scr.set_scaffold(final_smi).generation(limit_nb_widget.value, max_comb_widget.value)

            btn.on_click(click_save)
            widgets_groupes.append(btn) if has_anchors else widgets_groupes.append(widgets.Label("No anchors detected."))
            result_container.children = (widgets.Label("Result"), pil_to_widget(get_labeled_image(new_mol)), widgets.VBox(widgets_groupes))

        new_selector.observe(on_selection_change, names='value')
        with main_container:
            display(widgets.HBox([widgets.VBox([widgets.Label("1. Base Scaffold"), pil_to_widget(get_labeled_image(mol))]), widgets.VBox([widgets.Label("2. Cut Atoms"), new_selector]), result_container]))
            on_selection_change({'new': ()})

    mol_dropdown.observe(interface_dynamique, names='value')
    display(widgets.Label("🔧 Screening Bounds Settings:"), config_panel, mol_dropdown, main_container)
    interface_dynamique()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
