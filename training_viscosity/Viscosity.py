import numpy as np
import pandas as pd
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit import RDLogger
import lightgbm as lgb
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error

RDLogger.DisableLog("rdApp.*")

SMILES_COL = "compound_smiles"
TEMP_COL = "temperature_K"
INV_TEMP_COL = "inv_temperature_K"
PRESSURE_COL = "pressure_kPa"
LOG_PRESSURE_COL = "log10_pressure_kPa"
TARGET_COL = "viscosity_value"
UNIT_COL = "viscosity_unit"

TARGET_UNIT = "Viscosity, Pa&#8226;s => Liquid"

FP_RADIUS = 3
FP_NBITS = 1024
HUGE_THRESHOLD = 1e15
RANDOM_STATE = 42

PRODUCTION_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_estimators': 1611,
    'learning_rate': 0.22054952807433104,
    'num_leaves': 77,
    'max_depth': 5,
    'min_child_samples': 7,
    'subsample': 0.6861802325421853,
    'colsample_bytree': 0.5004968120923114,
    'reg_alpha': 0.002875599443489704,
    'reg_lambda': 0.013915148176133508
}

def clean_smiles(raw_smiles: str):
    if not isinstance(raw_smiles, str): return None
    frags = [f.strip() for f in raw_smiles.split("|")]
    frags = [f for f in frags if f]
    if not frags: return None
    combined = ".".join(frags)
    mol = Chem.MolFromSmiles(combined)
    return combined if mol else None

def calc_mol_descriptors(mol):
    result = {}
    for name, func in Descriptors._descList:
        try: result[name] = func(mol)
        except: result[name] = np.nan
    return result

def featurize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    desc_dict = calc_mol_descriptors(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fp_dict = {f"fp_{i}": int(arr[i]) for i in range(FP_NBITS)}
    return {**desc_dict, **fp_dict, "n_fragments": smiles.count(".") + 1}

def build_feature_matrix(df: pd.DataFrame):
    df = df.copy()
    df["_clean_smiles"] = df[SMILES_COL].apply(clean_smiles)
    df = df[df["_clean_smiles"].notna()].copy()
    
    unique_smiles = df["_clean_smiles"].unique()
    feat_rows = {smi: featurize_smiles(smi) for smi in unique_smiles if featurize_smiles(smi)}

    feat_df = pd.DataFrame.from_dict(feat_rows, orient="index").select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.mask(feat_df.abs() > HUGE_THRESHOLD).fillna(0)

    df = df[df["_clean_smiles"].isin(feat_df.index)].reset_index(drop=True)
    feat_cols = feat_df.loc[df["_clean_smiles"]].reset_index(drop=True)

    temp_pressure = df[[TEMP_COL, PRESSURE_COL]].reset_index(drop=True).copy()
    temp_pressure[INV_TEMP_COL] = 1.0 / temp_pressure[TEMP_COL]
    temp_pressure[LOG_PRESSURE_COL] = np.log10(temp_pressure[PRESSURE_COL])

    X = pd.concat([feat_cols, temp_pressure], axis=1).drop(columns=[PRESSURE_COL])
    return X, df[TARGET_COL].values

def clean_thermo_anomalies(df):
    df = df[(df[PRESSURE_COL].notna()) & (df[PRESSURE_COL] <= 200000) & 
            (df[TARGET_COL] <= 10.0) & (df[TARGET_COL] >= 1e-4)]
    return df

def remove_group_outliers(df, log_threshold=0.3):
    d = df.copy()
    d["_log_visc"] = np.log10(d[TARGET_COL])
    d["_T_rounded"] = d[TEMP_COL].round()
    medians = d.groupby([SMILES_COL, "_T_rounded"])["_log_visc"].transform("median")
    return d[(d["_log_visc"] - medians).abs() <= log_threshold].drop(columns=["_log_visc", "_T_rounded"])

def run_training():
    data_path = "dataset/Ils_Viscosity.csv"
    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df)} rows.")

    if UNIT_COL in df.columns: df = df[df[UNIT_COL] == TARGET_UNIT]
    df = clean_thermo_anomalies(df)
    df = remove_group_outliers(df)
    df.to_csv("il_thermo_cleaned_data.csv", index=False)

    X_full, y_raw_full = build_feature_matrix(df)
    y_full = np.log10(y_raw_full)

    scaler = StandardScaler()
    cols_to_scale = [TEMP_COL, INV_TEMP_COL, LOG_PRESSURE_COL]
    X_full_scaled = X_full.copy()
    X_full_scaled[cols_to_scale] = scaler.fit_transform(X_full[cols_to_scale])

    model = lgb.LGBMRegressor(**PRODUCTION_PARAMS, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_full_scaled, y_full)

    artifacts = {"model": model, "scaler": scaler, "feature_names": X_full.columns.tolist(), "cols_to_scale": cols_to_scale}
    with open("lgbm_viscosity_production.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    print("✅ Training complete. Artifacts saved to 'lgbm_viscosity_production.pkl'.")

if __name__ == "__main__":
    run_training()
