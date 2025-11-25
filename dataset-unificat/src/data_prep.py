"""
data_prep.py

Fitxer amb funcions bàsiques per:
- Carregar el CSV original de Spotify
- Fer una mica de neteja
- Seleccionar les columnes de features
- Escalar les variables numèriques

Més endavant hi podem afegir:
- Encoding de variables categòriques (key, mode, time_signature...)
- Guardar el dataset processat en un altre CSV
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Configuració bàsica
# -----------------------------

# Ruta al fitxer CSV original (la puc adaptar si cal)
RAW_DATA_PATH = "SpotifyFeatures.csv"


# Llista de columnes que vull conservar com a "info" (no com a features)
INFO_COLUMNS = [
    "genre",
    "artist_name",
    "track_name",
    "track_id",
]

# Llista de columnes numèriques que faran servir com a features pel model
# (aquesta llista la puc anar ajustant quan faci EDA)
FEATURE_COLUMNS = [
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
    # Més endavant potser afegeixo version codificada de "key", "mode", "time_signature", etc.
]


# -----------------------------
# 2. Funció per carregar les dades
# -----------------------------

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega el CSV original de Spotify.

    Parameters
    ----------
    path : str
        Ruta al fitxer CSV.

    Returns
    -------
    df : pd.DataFrame
        DataFrame amb les dades originals.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No he trobat el fitxer a: {path}")

    df = pd.read_csv(path)
    return df


# -----------------------------
# 3. Neteja bàsica
# -----------------------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica alguns passos de neteja molt bàsics.

    De moment:
    - Elimino files amb valors nuls a les columnes de features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame netejat.
    """
    # Em quedo només amb les columnes que m'interessen (info + features)
    columns_to_keep = INFO_COLUMNS + FEATURE_COLUMNS
    df_clean = df[columns_to_keep].copy()

    # Elimino files amb NaN a les columnes de features
    df_clean = df_clean.dropna(subset=FEATURE_COLUMNS)

    # Reset de l'índex per tenir-lo net
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


# -----------------------------
# 4. Separar info i features
# -----------------------------

def split_info_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame en:
    - info_df: columnes informatives (genre, artist_name, etc.)
    - features_df: columnes numèriques per al model

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame netejat.

    Returns
    -------
    info_df : pd.DataFrame
        Columnes d'informació (no s'escalen).
    features_df : pd.DataFrame
        Columnes de features numèriques.
    """
    info_df = df[INFO_COLUMNS].copy()
    features_df = df[FEATURE_COLUMNS].copy()
    return info_df, features_df


# -----------------------------
# 5. Escalar les features
# -----------------------------

def scale_features(features_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Escala les columnes numèriques fent servir StandardScaler.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame amb les columnes numèriques.

    Returns
    -------
    X_scaled : np.ndarray
        Matriu numpy amb les features escalades.
    scaler : StandardScaler
        Objecte scaler entrenat (per poder reutilitzar-lo més endavant).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.values)
    return X_scaled, scaler


# -----------------------------
# 6. Funció "end-to-end"
# -----------------------------

def prepare_dataset(path: str = RAW_DATA_PATH) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Pipeline complet bàsic:
    - Carregar CSV
    - Netejar
    - Separar info i features
    - Escalar features

    Parameters
    ----------
    path : str
        Ruta al fitxer CSV.

    Returns
    -------
    info_df : pd.DataFrame
        Dades informatives (sense escalar).
    X_scaled : np.ndarray
        Matriu amb les features escalades (llesta per fer clustering).
    scaler : StandardScaler
        Scaler entrenat sobre les dades.
    """
    # 1. Carrego el CSV
    df_raw = load_raw_data(path)

    # 2. Neteja bàsica
    df_clean = basic_cleaning(df_raw)

    # 3. Separo info i features
    info_df, features_df = split_info_features(df_clean)

    # 4. Escalo les features
    X_scaled, scaler = scale_features(features_df)

    return info_df, X_scaled, scaler


# -----------------------------
# 7. Prova ràpida (si executo aquest fitxer directament)
# -----------------------------

if __name__ == "__main__":
    # Això només s'executa si faig: python src/data_prep.py
    info_df, X_scaled, scaler = prepare_dataset()

    print("Mostro les primeres files d'info:")
    print(info_df.head())

    print("\nForma de la matriu de features escalades:")
    print(X_scaled.shape)
