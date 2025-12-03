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

# Ruta al fitxer CSV (utilitzant raw string per evitar problemes d'escapament)
#RAW_DATA_PATH = r"C:\Users\emmah\OneDrive\Escritorio\UNI\TERCER\ML\AC-10\dataset-unificat\data\raw\SpotifyFeatures.csv"
RAW_DATA_PATH = os.path.join("dataset-unificat", "data", "raw", "SpotifyFeatures.csv")

# Ruta per guardar el fitxer processat
#OUTPUT_PATH = r"C:\Users\emmah\OneDrive\Escritorio\UNI\TERCER\ML\AC-10\dataset-unificat\data\processed\spotify_processed.csv"  # Ruta relativa
OUTPUT_PATH = os.path.join("dataset-unificat", "data", "processed", "spotify_processed.csv")


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
# 3. Neteja bàsica modificada
# -----------------------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica alguns passos de neteja molt bàsics.

    De moment:
    - Eliminem files amb valors nuls a les columnes de features.
    - Comprovem valors nuls per columna.
    - Eliminem duplicats si n'hi ha.
    - Substituïm valors nuls en columnes numèriques amb la mitjana i en columnes categòriques amb el mode.
    
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

    # Comprovar valors nuls
    print("Missing values per column:")
    print(df_clean.isnull().sum())  # Mostra el nombre de valors nuls per columna
    
    # Comprovar els tipus de dades
    print("\nData types:")
    print(df_clean.dtypes)  # Mostra els tipus de dades

    # Eliminem duplicats si n'hi ha
    df_clean.drop_duplicates(inplace=True)
    print("\nEliminat duplicats, si n'hi havia.")

    # Substituïm els valors nuls en columnes numèriques amb la mitjana
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    # Substituïm valors nuls en les columnes numèriques amb la mitjana !!!
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Substituïm valors nuls en les columnes categòriques amb el mode !!!
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])


    # Comprovar els valors nuls després de la neteja
    print("\nAfter cleaning, missing values:")
    print(df_clean.isnull().sum())  # Comprovar valors nuls després de la neteja

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
    # .fit_transform() ajusta l'scaler a les dades calculant la mitja i desviació dels features
    # transforma les dades aplicant la normalització dels valors calculats.
    
    return X_scaled, scaler # Per si en el futur ens arriba una nova cançó i volem predir el seu cluster, necessitarem escalar-la exactament amb la mateixa matemàtica que les originals.


# -----------------------------
# 6. Funció "end-to-end" modificada
# -----------------------------

def prepare_dataset(path: str = RAW_DATA_PATH, output_path: str = OUTPUT_PATH) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Pipeline complet bàsic:
    - Carregar CSV
    - Netejar
    - Separar info i features
    - Escalar features
    - Guardar el CSV processat en un fitxer nou (si no existeix) o sobreescriure'l (si ja existeix).

    Parameters
    ----------
    path : str
        Ruta al fitxer CSV original.
    output_path : str
        Ruta per guardar el CSV processat.

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

    # 5. Comprovar si la carpeta de destinació existeix, i si no, crear-la
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Crea la carpeta si no existeix

    # 6. Guardar el CSV processat a la ruta indicada (nou fitxer CSV)
    df_clean.to_csv(output_path, index=False)  # Guarda el CSV processat (sobre escriu si ja existeix)

    print(f"El fitxer processat s'ha guardat a: {output_path}")  # Confirmació

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

    print(os.path.abspath(RAW_DATA_PATH))  # Per veure la ruta absoluta

