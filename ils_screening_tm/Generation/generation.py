import os
import logging
from itertools import product
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logger = logging.getLogger("il_screening.generation")

# --- CONFIGURATION & PATHS -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # Points to Git/ root

LIMIT_NB = 2
MAX_COMBINATIONS = 20_000  # Hard ceiling to avoid runaway combinatorics

FORBIDDEN_CONNECTIONS = {
    'N': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S', 'N'],
    'P': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S'],
    'S': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'N'],
    'O': ['F', 'Cl', 'Br', 'I', 'At', 'Ts', 'O', 'S', 'N', 'P'],
}
FORBIDDEN_SMARTS = ["N[CX3]=[O,S]"]

# Input databases (now directly inside database/)
FICHIER_LIGANDS = os.path.join(BASE_DIR, 'ils_screening_tm', 'database', 'substituents_library.csv')

# Output destination (now in output/ at the root)
FICHIER_ENTREE = os.path.join(BASE_DIR, 'output', 'generated_cations_raw.csv')

# --- STRUCTURAL FILTERING UTILITIES ----------------------------------------

def check_final_smarts_filter(mol) -> bool:
    """Check if the molecule contains any forbidden chemical patterns."""
    for smarts in FORBIDDEN_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return False
    return True


def get_scaffold_host_atom(mol, type_int: int) -> str:
    """Find the symbol of the host atom adjacent to the mapped attachment site."""
    for atom in mol.GetAtoms():
        box_num = atom.GetAtomMapNum()
        if box_num > 100 and (box_num // 100) == type_int:
            neighbors = atom.GetNeighbors()
            if neighbors:
                return neighbors[0].GetSymbol()
    return 'C'


def get_ligand_head_atom(pattern: str, typ: str) -> Optional[str]:
    """Determine the head atom symbol of a substituent pattern."""
    m = Chem.MolFromSmiles(pattern) if typ == 'SMILES' else Chem.MolFromSmarts(pattern)
    if not m:
        return None
    for atom in m.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            if neighbors:
                return neighbors[0].GetSymbol()
    return m.GetAtomWithIdx(0).GetSymbol()


def load_substituents_filtered(substituents_df: pd.DataFrame, ligand_type_str: str, host_symbol: str) -> List[List[str]]:
    """Load and filter substituents that comply with local structural rules."""
    if 'Ligand' not in substituents_df.columns:
        return []
    raw_list = substituents_df[substituents_df['Ligand'] == ligand_type_str][['Type', 'Pattern']].values.tolist()
    forbidden = FORBIDDEN_CONNECTIONS.get(host_symbol, [])

    filtered_list = []
    for item in raw_list:
        head = get_ligand_head_atom(item[1], item[0])
        if head not in forbidden:
            filtered_list.append(item)
            if len(filtered_list) >= LIMIT_NB:
                break
    return filtered_list


def filter_invalid_substituents(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-filter the substituent library to ensure all RDKit patterns are valid."""
    valid_rows = []
    for _, row in df.iterrows():
        try:
            m = Chem.MolFromSmiles(row['Pattern']) if row['Type'] == 'SMILES' else Chem.MolFromSmarts(row['Pattern'])
            if m:
                valid_rows.append(row)
        except Exception as exc:
            logger.debug("Error parsing substituent pattern %s: %s", row.get('Pattern'), exc)
    return pd.DataFrame(valid_rows)


# --- GENERATION LOGIC ------------------------------------------------------

@dataclass
class CombinationStats:
    total_generated: int = 0
    excluded_smarts: int = 0
    excluded_error: int = 0
    combinatorics_truncated: bool = False


class CombinationEncoded:
    """Generates substituted cation structures from an atom-map-encoded scaffold."""

    def __init__(self):
        self.stats = CombinationStats()

    def __call__(self, base_smiles_encoded: str, substituents_df: pd.DataFrame) -> List[Tuple[str, str]]:
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

        sub_df = filter_invalid_substituents(substituents_df)
        combs_config = {}
        type_to_int = {"Z": 1, "X": 2, "L": 3}

        for m_type in ["X", "L", "Z"]:
            unique_groups = sorted(groups_by_type[m_type])
            if not unique_groups:
                combs_config[m_type] = [()]
                continue
            host = get_scaffold_host_atom(base_mol, type_to_int[m_type])
            subs_list = load_substituents_filtered(sub_df, m_type, host)
            if not subs_list:
                logger.warning("No valid substituents found for site %s (host=%s).", m_type, host)
            combs_config[m_type] = list(product(subs_list, repeat=len(unique_groups))) or [()]

        estimated_total = len(combs_config["X"]) * len(combs_config["L"]) * len(combs_config["Z"])
        if estimated_total > MAX_COMBINATIONS:
            logger.warning("Estimated %d combinations exceeds cap of %d; truncating.", estimated_total, MAX_COMBINATIONS)
            self.stats.combinatorics_truncated = True

        working_mol = Chem.RWMol(base_mol)
        for atom in working_mol.GetAtoms():
            if atom.GetAtomMapNum() > 100:
                atom.SetIsotope(10000 + atom.GetAtomMapNum())
        base_tagged_mol = working_mol.GetMol()

        results = []
        for cX, cL, cZ in product(combs_config["X"], combs_config["L"], combs_config["Z"]):
            self.stats.total_generated += 1
            if self.stats.total_generated > MAX_COMBINATIONS:
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
                            except Exception:
                                pass
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
                        self.stats.excluded_error += 1
                else:
                    self.stats.excluded_smarts += 1
            except Exception as exc:
                logger.debug("Combination generation error: %s", exc)
                self.stats.excluded_error += 1

        return results


# --- ENTRY POINT FUNCTION --------------------------------------------------

def run_generation(final_smiles_encoded: str) -> pd.DataFrame:
    """
    Executes Step 1 of the pipeline: Generates the combinatorial cation structures,
    applies structural rules, removes duplicates, and saves the result to output/.
    """
    logger.info("Executing Step 1: Combinatorial Cation Generation...")
    
    if not os.path.exists(FICHIER_LIGANDS):
        raise FileNotFoundError(f"Substituent library not found at: {FICHIER_LIGANDS}")
        
    df_substituants = pd.read_csv(FICHIER_LIGANDS)
    
    combiner = CombinationEncoded()
    generated_results = combiner(final_smiles_encoded, df_substituants)
    
    if not generated_results:
        raise ValueError("No structures survived the generation and initial structural filters.")

    # Create DataFrame and drop duplicate SMILES
    df_raw = pd.DataFrame(generated_results, columns=['SMILES', 'Legend'])
    df_unique = df_raw.drop_duplicates(subset=['SMILES'], keep='first').copy()
    
    # Save checkpoint to output/
    os.makedirs(os.path.dirname(FICHIER_ENTREE), exist_ok=True)
    df_unique.to_csv(FICHIER_ENTREE, index=False)
    
    logger.info(f"Step 1 Complete: {len(df_unique)} unique cations generated and saved to {FICHIER_ENTREE}")
    return df_unique
