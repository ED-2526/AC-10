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

import os #Serveix per interactuar amb el sistema operatiu. Aquí s'usa per comprovar si el fitxer existeix abans d'intentar obrir-lo (gestió d'errors bàsica).
from typing import Tuple #No canvia l'execució, però serveix per indicar (pistes de tipus) què retorna una funció. Ajuda a que el codi sigui més llegible i que l'IDE t'avisi d'errors.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler #Serveix per posar totes les dades en la mateixa escala (mitjana 0, desviació 1).


# -----------------------------
# 1. Configuració bàsica
# -----------------------------

# Ruta al fitxer CSV 
RAW_DATA_PATH = "SpotifyFeatures.csv"


# Llista de columnes que volem conservar com a "info" (no com a features)
#Dades per a humans (Títol cançó, Artista). No serveixen per calcular distàncies matemàtiques.
INFO_COLUMNS = [
    "genre",
    "artist_name",
    "track_name",
    "track_id",
]

# Llista de columnes numèriques que faran servir com a features pel model
#Dades per a la màquina (Energy, Loudness). Són nombres amb els quals farem el clustering.
# (aquesta llista la podem anar ajustant quan faci EDA)
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
    # Més endavant potser afegim version codificada de "key", "mode", "time_signature", etc.
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
    - Eliminem files amb valors nuls a les columnes de features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame netejat.
    """
    # Ens quedem només amb les columnes que interessen (info + features)
    columns_to_keep = INFO_COLUMNS + FEATURE_COLUMNS
    
    df_clean = df[columns_to_keep].copy() 
    # Utilitzem .copy() per evitar que df_clean sigui una referencia al DF original, 
    # sino qualsevol canvi en df_clean afectaria al DF, sino els 2 apunten a les mateixes dades.
    # Creem una copia independent i que els canvis no alteren el DF original.

    # Eliminem files amb NaN a les columnes de features
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
    return X_scaled, scaler # Per si en el futur ens arriba una nova cançó i volem predir el seu cluster, necessitarem escalar-la exactament amb la mateixa matemàtica que les originals.


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
    # 1. Carreguem el CSV
    df_raw = load_raw_data(path)

    # 2. Neteja bàsica
    df_clean = basic_cleaning(df_raw)

    # 3. Separem info i features
    info_df, features_df = split_info_features(df_clean)

    # 4. Escalem les features
    X_scaled, scaler = scale_features(features_df)

    return info_df, X_scaled, scaler


# -----------------------------
# 7. Prova ràpida
# -----------------------------

if __name__ == "__main__":
    # Això només s'executa si fem: python src/data_prep.py
    info_df, X_scaled, scaler = prepare_dataset()

    print("Mostro les primeres files d'info:")
    print(info_df.head())

    print("\nForma de la matriu de features escalades:")
    print(X_scaled.shape)
