import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ────────────────────────────────────────────────
# RUTES
# ────────────────────────────────────────────────

RAW_PATH = "dataset-millorat/data/raw/SpotifyFeatures.csv"
PROC_DIR = "dataset-millorat/data/processat"
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
        "genre": lambda x: list(sorted(set(x))),   
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

    if "track_id" in agg_dict:
        del agg_dict["track_id"]

    df_grouped = df.groupby("track_id").agg(agg_dict).reset_index()

    print(f"Files originals: {len(df)}")
    print(f"Files després d'agrupar: {len(df_grouped)}")

    return df_grouped



# ────────────────────────────────────────────────
# 4. CREAR MULTI-LABEL DEL GÈNERE
# ────────────────────────────────────────────────

def genere_multilabel(df_original, df_grouped):
    generes = sorted(df_original["genre"].unique())

    matriu = pd.DataFrame(0, index=df_grouped["track_id"], columns=generes)

    for tid, group in df_original.groupby("track_id"):
        for g in group["genre"].unique():
            matriu.loc[tid, g] = 1

    matriu.reset_index(inplace=True)

    df_final = df_grouped.merge(matriu, on="track_id")

    print("One-hot multilabel de gèneres aplicat.")
    return save_step(df_final, "04_genres_multilabel.csv")


# ────────────────────────────────────────────────
# 5. DEFINIR COLUMNES INFO I FEATURES
# ────────────────────────────────────────────────

INFO_COLUMNS = [
    "artist_name",
    "track_name",
    "track_id",
]

# NUMÈRIQUES PER CLUSTERING
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

# Elsa columnes de gènere s’afegeixen després


# ────────────────────────────────────────────────
# 6. PREPARACIÓ PER CLUSTERING (escala + selecció)
# ────────────────────────────────────────────────

def preparar_clustering(df_clean):

    # Eliminar columnes categòriques no codificades
    CATEGORICAL_DROP = ["genre", "key", "mode", "time_signature"]
    df_clean = df_clean.drop(columns=CATEGORICAL_DROP, errors="ignore")

    # Detectar automàticament les columnes de gèneres one-hot
    genre_cols = [
        c for c in df_clean.columns
        if c not in INFO_COLUMNS + BASE_FEATURE_COLUMNS
    ]

    # Construïm la llista final de features
    FEATURES = BASE_FEATURE_COLUMNS + genre_cols

    # Separem info i features
    info_df = df_clean[INFO_COLUMNS].copy()
    features_df = df_clean[FEATURES].copy()

    # Escalar només números
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Guardar CSV escalat
    df_scaled_export = pd.DataFrame(X_scaled, columns=FEATURES)
    df_scaled_export.insert(0, "track_id", df_clean["track_id"])

    save_step(df_scaled_export, "06_dataset_prepared_for_clustering.csv")

    print("\nDataset preparat per clustering:")
    print(f" - Files: {len(df_clean)}")
    print(f" - Features numèriques escalades: {len(FEATURES)}")
    print(f" - Columnes de gènere multilabel: {len(genre_cols)}")

    return info_df, X_scaled, scaler




# ────────────────────────────────────────────────
# PIPELINE COMPLET
# ────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n===== INICIANT NETEJA I PREPROCESSAMENT COMPLET =====\n")

    df0 = carregar_csv()
    df1 = eliminar_nan(df0)
    df2 = agrupar_track_id(df1)
    df3 = genere_multilabel(df0, df2)

    save_step(df3, "05_final_clean_dataset.csv")

    info_df, X_scaled, scaler = preparar_clustering(df3)

    print("\n===== PROCÉS ACABAT! Dataset llest per clustering =====\n")
