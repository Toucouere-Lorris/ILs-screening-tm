import math
import fractions
fractions.gcd = math.gcd

import pandas as pd
import numpy as np
import pickle
import joblib
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler


df_final = pd.read_csv('dataset/tm_data.csv')

with open('../ils_screening_tm/models/for-external.pkl', 'rb') as f:
    _ = pickle.load(f)                
    col_descriptors = pickle.load(f) 

calc = Calculator(descriptors, ignore_3D=True)

def compute_209_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    df_all = calc.pandas(mols, quiet=True)
    df_all = df_all.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_all[col_descriptors].values

print("\nCalculating Mordred descriptors for Cations...")
X_cat_raw = compute_209_descriptors(df_final['Molecule1']) 

print("Calculating Mordred descriptors for Anions...")
X_an_raw = compute_209_descriptors(df_final['Molecule2']) 

X_phys_raw = np.concatenate([X_cat_raw, X_an_raw], axis=1)

scaler_phys = StandardScaler()
X_phys_std = scaler_phys.fit_transform(X_phys_raw)

joblib.dump(scaler_phys, '../ils_screening_tm/models/scaler_mordred.pkl')

X_cat_mordred = X_phys_std[:, :209]
X_an_mordred = X_phys_std[:, 209:]

y = df_final['mpK'].values

print("\n=== DATA PREPARATION COMPLETED ===")
print(f"X_cat_mordred shape : {X_cat_mordred.shape} (209 features for Cation)")
print(f"X_an_mordred shape  : {X_an_mordred.shape} (209 features for Anion)")
print(f"Target y shape      : {y.shape}")


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

idx = np.arange(len(y))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

X_cat_train_full = X_cat_mordred[idx_train]
X_an_train_full = X_an_mordred[idx_train]
y_train_full = y[idx_train]

X_cat_test = X_cat_mordred[idx_test]
X_an_test = X_an_mordred[idx_test]
y_test = y[idx_test]

def build_pscnn():
    NB_FEATURES = 209

    input_cation = Input(shape=(NB_FEATURES,), name="Mordred_Cation")
    rc = Reshape((NB_FEATURES, 1))(input_cation)
    c = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(rc)
    c = MaxPooling1D(pool_size=2)(c)
    c = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(c)
    c = MaxPooling1D(pool_size=2)(c)
    c = Flatten()(c)

    input_anion = Input(shape=(NB_FEATURES,), name="Mordred_Anion")
    ra = Reshape((NB_FEATURES, 1))(input_anion)
    a = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(ra)
    a = MaxPooling1D(pool_size=2)(a)
    a = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(a)
    a = MaxPooling1D(pool_size=2)(a)
    a = Flatten()(a)

    fusion = Concatenate(name="Ion_Fusion")([c, a])

    z = Dense(256, activation='relu')(fusion)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)

    z = Dense(128, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)

    output_tm = Dense(1, activation='linear', name="Prediction_Tm")(z)

    model = Model(inputs=[input_cation, input_anion], outputs=output_tm)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    return model


N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_test_predictions = [] 
fold_no = 1

print(f"\n🚀 Training and Saving the {N_SPLITS}-Fold Ensemble Model...")

for train_index, val_index in kf.split(X_cat_train_full):
    print(f"\n--- Training Fold Model {fold_no} / {N_SPLITS} ---")
    
    X_cat_f_train, X_cat_f_val = X_cat_train_full[train_index], X_cat_train_full[val_index]
    X_an_f_train, X_an_f_val = X_an_train_full[train_index], X_an_train_full[val_index]
    y_f_train, y_f_val = y_train_full[train_index], y_train_full[val_index]
    
    model = build_pscnn()
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=0)
    
    model.fit(
        [X_cat_f_train, X_an_f_train], 
        y_f_train,
        validation_data=([X_cat_f_val, X_an_f_val], y_f_val),
        epochs=150,
        batch_size=32,
        callbacks=[reduce_lr, early_stopping],
        verbose=0 
    )
    
    filename = f'../ils_screening_tm/models/pscnn_fold_{fold_no}.keras'
    model.save(filename)
    print(f"💾 Fold {fold_no} model successfully saved as: {filename}")
    
    print(f"✅ Calculating test predictions for Fold {fold_no}...")
    pred_test = model.predict([X_cat_test, X_an_test], verbose=0).flatten()
    all_test_predictions.append(pred_test)
    
    fold_no += 1


y_pred_ensemble = np.mean(all_test_predictions, axis=0)

rmse_m = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
mae_m = mean_absolute_error(y_test, y_pred_ensemble)
r2_m = r2_score(y_test, y_pred_ensemble)

print("\n" + "="*45)
print(" 🎯 ENSEMBLE MODEL RESULTS (Average of 5 Folds):")
print("="*45)
print(f" RMSE : {rmse_m:.4f} K")
print(f" MAE  : {mae_m:.4f} K")
print(f" R²   : {r2_m:.4f}")
print("="*45)


