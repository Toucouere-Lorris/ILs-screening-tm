import io
import os
import sys
import logging
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as widgets
from rdkit import Chem
from rdkit.Chem import Draw

# Import the 4 sequential modules we created
from ils_screening_tm.Generation.generation import run_generation
from ils_screening_tm.SAScore.sascore import run_sascore_filtering
from ils_screening_tm.Prediction_tm.prediction_tm import run_tm_prediction
from ils_screening_tm.Display.display import run_visualization

# Mute warnings for a cleaner interface
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("il_screening.main")

# --- CONFIGURATION & PATHS -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

FICHIER_CATIONS = os.path.join(BASE_DIR, 'database', 'base_cations.csv')
FICHIER_LIGANDS = os.path.join(BASE_DIR, 'database', 'substituents_library.csv')

# Global state to keep track of the interactive encoding
selection_state = {"encoded_smiles": None}


# --- UI INTERACTIVE UTILITIES ----------------------------------------------

def pil_to_widget(pil_image) -> widgets.Image:
    """Converts a PIL image into an IPython image widget."""
    byte_io = io.BytesIO()
    pil_image.save(byte_io, format='PNG')
    return widgets.Image(value=byte_io.getvalue(), format='png', width=350, height=350)


def remove_atoms_interactive(mol, indices) -> Chem.Mol:
    """Removes atom indices from a molecule, safely handling sanitization."""
    if not indices:
        return mol
    try:
        editable = Chem.EditableMol(mol)
        for idx in sorted(indices, reverse=True):
            editable.RemoveAtom(idx)
        new_mol = editable.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception:
            pass
        return new_mol
    except Exception as exc:
        logger.warning(f"Atom removal failed: {exc}")
        return mol


def get_labeled_image(mol):
    """Generates a 2D depiction of the molecule with atom indices visible."""
    dopts = Draw.MolDrawOptions()
    dopts.addAtomIndices = True
    dopts.bondLineWidth = 2
    return Draw.MolToImage(mol, size=(400, 400), options=dopts)


# --- INTERACTIVE INTERFACE LAUNCHER ----------------------------------------

def start_screening_interface(df_scaffolds: pd.DataFrame) -> None:
    """Builds and renders the full modular ipywidgets dashboard."""
    if df_scaffolds.empty:
        print("Error: The base Cations/Scaffolds DataFrame is empty.")
        return

    mol_slider = widgets.IntSlider(value=0, min=0, max=len(df_scaffolds) - 1, description='Scaffold:')
    main_container = widgets.Output()

    def refresh_dashboard(change=None):
        main_container.clear_output(wait=True)
        idx = mol_slider.value
        
        try:
            smiles = df_scaffolds['SMILES'].iloc[idx]
        except Exception as exc:
            logger.error(f"Could not read scaffold at index {idx}: {exc}")
            return

        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if not mol:
            logger.error(f"Could not parse SMILES at index {idx}")
            return

        atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        cut_selector = widgets.SelectMultiple(
            options=atom_indices, value=(), description='Cut Atoms:', rows=10
        )
        result_container = widgets.VBox([])

        def on_cut_change(change_event):
            selected_atoms = change_event['new']
            new_mol = remove_atoms_interactive(mol, selected_atoms)

            # Detect remaining anchor placeholders
            anchors_found = {1: [], 2: [], 3: []}
            for atom in new_mol.GetAtoms():
                raw_type = atom.GetIsotope() or atom.GetAtomMapNum()
                if raw_type in anchors_found:
                    anchors_found[raw_type].append(atom.GetIdx())

            widgets_group = [widgets.HTML("<b>3. Configure Site Symmetries</b>")]
            current_dropdowns = {}
            has_anchors = False
            type_names = {1: "Z (Z Site)", 2: "X (X Site)", 3: "L (L Site)"}

            for type_code in [2, 3, 1]:
                indices = anchors_found[type_code]
                if not indices:
                    continue
                has_anchors = True
                widgets_group.append(widgets.Label(f"--- {type_names[type_code]} : {len(indices)} sites ---"))
                
                options_list = list(range(1, max(4, len(indices) + 1)))

                for i, at_idx in enumerate(indices):
                    default_val = (i % len(options_list)) + 1
                    dd = widgets.Dropdown(
                        options=options_list, value=default_val, description=f"Atom {at_idx}:",
                        layout=widgets.Layout(width='180px'), style={'description_width': '80px'}
                    )
                    current_dropdowns[at_idx] = {'widget': dd, 'type': type_code}
                    widgets_group.append(dd)

            btn_launch = widgets.Button(description="Launch Full Pipeline", button_style='success')

            def on_click_launch(b):
                btn_launch.disabled = True
                btn_launch.description = "Processing Pipeline..."
                btn_launch.button_style = 'warning'

                with main_container:
                    clear_output(wait=True)
                    display(widgets.HTML(
                        "<h3 style='color: #d97706;'>🚀 Full Screening Pipeline in Progress... "
                        "Calculating descriptors and predictions. Please wait.</h3>"
                    ))

                    # Encode atom map mappings back into the molecule structures
                    working_mol = Chem.RWMol(new_mol)
                    for at in working_mol.GetAtoms():
                        at.SetIsotope(0)
                        at.SetAtomMapNum(0)

                    for at_idx, data in current_dropdowns.items():
                        atom = working_mol.GetAtomWithIdx(at_idx)
                        encoded_map = data['type'] * 100 + int(data['widget'].value)
                        atom.SetAtomMapNum(encoded_map)

                    final_smiles_encoded = Chem.MolToSmiles(working_mol)
                    selection_state["encoded_smiles"] = final_smiles_encoded

                    # --- PIPELINE ORCHESTRATION EXECUTION ---
                    try:
                        # Step 1: Combinatorial Generation
                        run_generation(final_smiles_encoded)
                        
                        # Step 2: SAScore filtering & Anion pairing
                        run_sascore_filtering()
                        
                        # Step 3: Deep Learning Prediction & Tm Filter
                        run_tm_prediction()
                        
                        # Step 4: Display Results & Plot samples
                        clear_output(wait=True)
                        run_visualization(sample_size=4)
                        
                    except Exception as pipeline_error:
                        clear_output(wait=True)
                        print(f"❌ Pipeline Execution Failed: {pipeline_error}")
                        btn_launch.disabled = False
                        btn_launch.description = "Launch Full Pipeline"
                        btn_launch.button_style = 'success'

            btn_launch.on_click(on_click_launch)

            if has_anchors:
                widgets_group.append(btn_launch)
            else:
                widgets_group.append(widgets.Label("No active map placeholders found."))

            img_res = pil_to_widget(get_labeled_image(new_mol))
            result_container.children = (
                widgets.Label("Resulting Structure"), img_res, widgets.VBox(widgets_group),
            )

        cut_selector.observe(on_cut_change, names='value')
        img_base = pil_to_widget(get_labeled_image(mol))

        with main_container:
            display(widgets.HBox([
                widgets.VBox([widgets.Label("1. Choose Scaffold Base"), img_base]),
                widgets.VBox([widgets.Label("2. Select Atoms to Remove"), cut_selector]),
                result_container,
            ]))
            on_cut_change({'new': ()})

    mol_slider.observe(refresh_dashboard, names='value')
    display(mol_slider)
    display(main_container)
    refresh_dashboard()


# --- APP TRIGGER -----------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(FICHIER_CATIONS):
        print(f"CRITICAL ERROR: Scaffold base file not found at {FICHIER_CATIONS}")
    else:
        df_scaffolds = pd.read_csv(FICHIER_CATIONS)
        start_screening_interface(df_scaffolds)
