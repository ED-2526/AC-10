
import os 
import pandas as pd


### --------------------------------------------------------------------------- ###

def get_column_names(df: pd.DataFrame):
    """
    Aquesta funció retorna el nom de totes les columnes del DataFrame.

    Paràmetres:
    - df: DataFrame amb les dades

    Retorna:
    - Llista de noms de les columnes.
    """
    return df.columns.tolist()



### --------------------------------------------------------------------------- ###

def print_rows_with_nan_and_matches(df: pd.DataFrame):
    """
    Mostra totes les files amb almenys un NaN i també mostra,
    per cada fila trobada, totes les altres files del mateix track_id
    per poder recuperar informació com track_name, genre, etc.

    Retorna el nombre total de files amb NaN.
    """

    mask_nan = df.isna().any(axis=1)
    rows_with_nan = df[mask_nan]

    print("\n========== FILES QUE CONTENEN ALMENYS UN VALOR NaN ==========\n")

    if rows_with_nan.empty:
        print("No s'ha trobat cap fila amb valors NaN. Dataset net! 🎉")
        return 0

    for idx, row in rows_with_nan.iterrows():

        print(f"\n--- Fila {idx} (amb NaN) ---")
        print(row.to_string())

        track_id = row["track_id"]

        print("\n>> Buscant altres files amb el mateix track_id per recuperar informació...")

        # cerquem altres files amb el mateix id
        matches = df[df["track_id"] == track_id]

        # si només n'hi ha una, no hi ha res més per comparar
        if len(matches) <= 1:
            print("   No existeixen altres registres amb aquest track_id.\n")
        else:
            print(f"   {len(matches)-1} registres addicionals trobats:\n")

            for midx, mrow in matches.iterrows():
                if midx == idx:
                    continue  

                print(f"   --- Fila {midx} ---")
                print(mrow.to_string())
                print()

    total = len(rows_with_nan)
    print(f"\n========== TOTAL FILES AMB ALMENYS UN NaN: {total} ==========\n")

    return total



### --------------------------------------------------------------------------- ###

def find_duplicate_tracks(df: pd.DataFrame):
    """
    Aquesta funció busca els duplicats a la columna `track_id` i mostra
    la informació completa de les cançons duplicades.

    Paràmetres:
    - df: DataFrame amb les dades de les cançons

    La funció imprimeix:
    - Coincidència 1: Fila 1 amb `track_id` duplicat i les seves dades
    - Coincidència 2: Fila 2 amb el mateix `track_id` i les seves dades
    - I així successivament per a tots els duplicats.
    """
    # Trobar les files amb id duplicats
    duplicated_ids = df[df.duplicated(subset='track_id', keep=False)]  # Keep False per mantenir totes les ocurrències
    
    # Agrupar les files duplicades pel `track_id`
    grouped = duplicated_ids.groupby('track_id')
    
    for track_id, group in grouped:
        print(f"\nCoincidència per id {track_id}:")
        for idx, row in group.iterrows():
            print(f"\nFila {idx + 1} amb ID {track_id}:")
            print(row[['genre', 'artist_name', 'track_name', 'track_id', 'popularity', 
                       'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
                       'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 
                       'valence']].to_string())
            print("-" * 50)  # Separador visual per millorar llegibilitat



### --------------------------------------------------------------------------- ###

def find_duplicates_with_different_popularity(df: pd.DataFrame):
    """
    Aquesta funció busca coincidències de cançons que tenen el mateix `track_id` i `genre`,
    però amb diferents valors de `popularity`. Mostra les coincidències i les seves dades.
    
    Paràmetres:
    - df: DataFrame amb les dades de les cançons.
    """
    # Trobar les files amb id duplicats i gènere coincident
    duplicated_tracks = df[df.duplicated(subset=['track_id', 'genre'], keep=False)]
    
    # Agrupar les files duplicades per `track_id` i `genre`
    grouped_duplicates = duplicated_tracks.groupby(['track_id', 'genre'])
    
    for (track_id, genre), group in grouped_duplicates:
        # Comprovar si la popularitat és diferent
        if group['popularity'].nunique() > 1:
            print(f"\nCoincidència per ID {track_id} i Gènere {genre}:")
            for idx, row in group.iterrows():
                print(f"Fila {idx + 1} - Popularitat: {row['popularity']}")
                print(f"Detalls de la cançó: {row[['genre', 'artist_name', 'track_name', 'track_id', 'popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']]}")
                print("-" * 50)



### --------------------------------------------------------------------------- ###

def find_perfect_duplicates(df: pd.DataFrame):
    """
    Troba coincidències PERFECTES: files on totes les columnes són idèntiques.
    
    Per cada grup de duplicats:
    - Mostra totes les files que tenen exactament els mateixos valors.
    """
    
    # 1. Detectar duplicats perfectes (totes les columnes iguals)
    perfect_dups = df[df.duplicated(keep=False)]
    
    # 2. Agrupar-los per totes les columnes (grup de files idèntiques)
    grouped = perfect_dups.groupby(list(df.columns))
    
    coincidencia_num = 1
    
    for values, group in grouped:
        # Si només hi ha 1 fila en el grup, no és duplicat
        if len(group) < 2:
            continue
        
        print(f"\n===============================")
        print(f"Coincidència PERFECTA {coincidencia_num}:")
        print(f"(Hi ha {len(group)} files idèntiques)")
        print("===============================\n")

        for idx, row in group.iterrows():
            print(f"Fila {idx}:\n")
            print(row.to_string())
            print("\n" + "-"*60 + "\n")
        
        coincidencia_num += 1



### --------------------------------------------------------------------------- ###

def find_id_pop_same_genere_diff(df: pd.DataFrame):
    """
    Troba coincidències on:
    - el track_id és igual
    - la popularity és igual
    - el genre CANVIA

    Imprimeix totes les coincidències i retorna el total detectat.
    """
    
    coincidencies = 0
    
    # Agrupem per track_id i popularity (valors que han de coincidir)
    grouped = df.groupby(["track_id", "popularity"])
    
    for (track_id, pop), group in grouped:
        # Si hi ha més d’un gènere dins el grup → hi ha discrepància
        if group["genre"].nunique() > 1:
            coincidencies += 1
            
            print("\n===============================================")
            print(f"Coincidència (track_id={track_id}, popularity={pop})")
            print("→ Mateix id i popularitat, PERÒ gènere diferent")
            print("===============================================")
            
            for idx, row in group.iterrows():
                print(f"\nFila {idx}:")
                print(row.to_string())
                print("\n-----------------------------------------------")
    
    print(f"\nTOTAL coincidències trobades: {coincidencies}")
    return coincidencies



### --------------------------------------------------------------------------- ###

def find_id_genere_pop_diff(df: pd.DataFrame):
    """
    Troba coincidències on:
    - track_id és igual
    - genre CANVIA
    - popularity TAMBÉ CANVIA

    Imprimeix totes les coincidències i retorna el total detectat.
    """

    coincidencies = 0

    # Agrupem per track_id perquè és la clau que ha de coincidir
    grouped = df.groupby("track_id")

    for track_id, group in grouped:
        # Necessitem almenys 2 files per comparar
        if len(group) < 2:
            continue
        
        # Comprovar si CANVIA genre i popularity alhora
        if group["genre"].nunique() > 1 and group["popularity"].nunique() > 1:
            coincidencies += 1

            print("\n===================================================")
            print(f"Coincidència per track_id={track_id}")
            print("→ Mateix id, però gènere I popularitat canvien")
            print("===================================================")

            for idx, row in group.iterrows():
                print(f"\nFila {idx}:")
                print(row.to_string())
                print("\n---------------------------------------------------")

    print(f"\nTOTAL coincidències trobades: {coincidencies}")
    return coincidencies






#### MAIN CODE PER PROVAR LES FUNCIONS ####

RAW_DATA_PATH = os.path.join("dataset-unificat", "data", "raw", "SpotifyFeatures.csv")
df = pd.read_csv(RAW_DATA_PATH)


# Cridem la funció per obtenir el nom de les columnes
#column_names = get_column_names(df)
#print(column_names)


# Cridem la funció per imprimir les files amb NaN
print_rows_with_nan_and_matches(df)


# Cridem la funció per trobar i mostrar els duplicats
#find_duplicate_tracks(df)


# Cridem la funció per trobar les coincidències amb popularitat diferent
#find_duplicates_with_different_popularity(df)


# Cridem la funció per trobar les coincidències PERFECTES
#find_perfect_duplicates(df)

# Cridem la funció per trobar coincidències amb mateix id i popularitat però gènere diferent
#find_id_pop_same_genere_diff(df)

# Cridem la funció per trobar coincidències amb mateix id però gènere i popularitat diferents
#find_id_genere_pop_diff(df)
