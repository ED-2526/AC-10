import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ────────────────────────────────────────────────
# RUTES
# ────────────────────────────────────────────────

RAW_PATH = "dataset-millorat/data/raw/SpotifyFeatures.csv"
PROC_DIR = "dataset-millorat/data/processat2"
FINAL_CLEAN_PATH = os.path.join(PROC_DIR, "05_final_clean_dataset.csv")
FINAL_PREP_PATH = os.path.join(PROC_DIR, "06_dataset_prepared_for_clustering.csv")

os.makedirs(PROC_DIR, exist_ok=True)


# ────────────────────────────────────────────────
# FUNCIONS AUXILIARS
# ────────────────────────────────────────────────

def save_step(df: pd.DataFrame, filename: str):
    """Guarda un DataFrame i printa shape."""
    path = os.path.join(PROC_DIR, filename)
    df.to_csv(path, index=False)
    print(f"\n✔ Guardat {filename} — Shape = {df.shape}")
    return df


# ────────────────────────────────────────────────
# 1. CARREGAR CSV
# ────────────────────────────────────────────────

def carregar_csv():
    df = pd.read_csv(RAW_PATH)
    print("\nCSV original carregat.")
    return save_step(df, "01_original_loaded.csv")


# ────────────────────────────────────────────────
# 2. ELIMINAR NaN
# ────────────────────────────────────────────────

def eliminar_nan(df):
    df2 = df.dropna()
    eliminats = len(df) - len(df2)
    print(f"Files eliminades per NaN: {eliminats}")
    return save_step(df2, "02_without_NaN.csv")


# ────────────────────────────────────────────────
# 3. AGRUPAR PER track_id I FER MITJANES
# ────────────────────────────────────────────────

def agrupar_track_id(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Agrupant per track_id i resolent duplicats ---")

    agg_dict = {
        "artist_name": "first",
        "track_name": "first",
        "popularity": "mean",
        "acousticness": "mean",
        "danceability": "mean",
        "duration_ms": "mean",
        "energy": "mean",
        "instrumentalness": "mean",
        "liveness": "mean",
        "loudness": "mean",
        "speechiness": "mean",
        "tempo": "mean",
        "valence": "mean",
        "mode": "first",
        "key": "first",
        "time_signature": "first",
    }

    df_grouped = df.groupby("track_id").agg(agg_dict).reset_index()

    print(f"Files originals: {len(df)}")
    print(f"Files després d'agrupar: {len(df_grouped)}")

    return df_grouped


# ────────────────────────────────────────────────
# 5. DEFINIR COLUMNES INFO I FEATURES
# ────────────────────────────────────────────────

# Eliminar les columnes info (artist_name, track_name) si no es volen usar per al clustering
INFO_COLUMNS = [
    "track_id",  # Només afegim track_id com a columna informativa
]

# NUMÈRIQUES PER CLUSTERING (afegim només les característiques numèriques)
BASE_FEATURE_COLUMNS = [
    "popularity",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]

# Si vols incloure les columnes de gènere multilabel, les afegirem aquí
# Eliminar columnes de gènere si no es volen considerar
# Els noms de les columnes de gènere multilabel es poden mantenir per a la selecció de característiques
# Això dependrà de la configuració que es vulgui seguir.

# ────────────────────────────────────────────────
# 6. PREPARACIÓ PER CLUSTERING (escala + selecció)
# ────────────────────────────────────────────────

def preparar_clustering(df_clean):

    # Eliminar les columnes no necessàries (per exemple, gèneres i altres no numèriques)
    CATEGORICAL_DROP = ["genre", "key", "mode", "time_signature"]
    df_clean = df_clean.drop(columns=CATEGORICAL_DROP, errors="ignore")

    # Seleccionar només les columnes necessàries per al clustering (numèriques i track_id)
    features_df = df_clean[BASE_FEATURE_COLUMNS].copy()  # Escollim només les columnes numèriques
    info_df = df_clean[INFO_COLUMNS].copy()  # Mantenim només track_id com a info

    # Escalar les característiques numèriques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Guardar el dataset processat escalat
    df_scaled_export = pd.DataFrame(X_scaled, columns=BASE_FEATURE_COLUMNS)
    df_scaled_export.insert(0, "track_id", df_clean["track_id"])

    save_step(df_scaled_export, "06_dataset_prepared_for_clustering.csv")

    print("\nDataset preparat per clustering:")
    print(f" - Files: {len(df_clean)}")
    print(f" - Features numèriques escalades: {len(BASE_FEATURE_COLUMNS)}")

    return info_df, X_scaled, scaler



# ────────────────────────────────────────────────
# PIPELINE COMPLET
# ────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n===== INICIANT NETEJA I PREPROCESSAMENT COMPLET =====\n")

    df0 = carregar_csv()
    df1 = eliminar_nan(df0)
    df2 = agrupar_track_id(df1)

    save_step(df2, "05_final_clean_dataset.csv")

    info_df, X_scaled, scaler = preparar_clustering(df2)

    print("\n===== PROCÉS ACABAT! Dataset llest per clustering =====\n")
