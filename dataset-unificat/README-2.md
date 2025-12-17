# Music Listener Clustering

Projecte d'Aprenentatge Computacional -- UAB

Aquest document recull **tot el que hem fet fins ara** al projecte, fins
al punt actual del desenvolupament. Encara no inclou models avançats
(GMM, jeràrquic), però sí tot el preprocessament, baseline i primeres
mètriques.

------------------------------------------------------------------------

# 1. Estructura del repositori

    dataset-unificat/
    ├── data/
    │   ├── raw/
    │   │   └── SpotifyFeatures.csv
    │   └── procesada/             # (pendent d’afegir)
    ├── src/
    │   ├── data_prep.py           # Preprocessament complet
    │   └── clustering_baseline.py # Baseline amb KMeans
    ├── figures/                   # Gràfics generats automàticament
    ├── experiments.md             # Registre d'experiments
    ├── guia.md                    # Guia inicial del projecte
    └── requirements.txt

------------------------------------------------------------------------

# 2. Dataset utilitzat

Dataset: **Ultimate Spotify Tracks DB (Kaggle)**\
Fitxer utilitzat: `data/raw/SpotifyFeatures.csv`

Conté: - Informació musical: artista, track, gènere. - Features
numèriques: danceability, valence, energy, loudness, tempo, etc.

Aquestes features són les que fem servir per al clustering.

------------------------------------------------------------------------

# 3. Preprocessament (src/data_prep.py)

El preprocessament fet fins ara inclou:

### 1. Carregar el CSV

Amb comprovació de ruta i lectura segura.

### 2. Neteja bàsica

-   Selecció de columnes necessàries\
-   Eliminació de files amb NaN\
-   Reset de l'índex

### 3. Separar info i features

-   INFO: genre, artist_name, track_name, track_id\
-   FEATURES: totes les columnes numèriques

### 4. Escalat amb StandardScaler

Imprescindible per K-means i models basats en distàncies.

### 5. Funció `prepare_dataset()`

Retorna: - `info_df`\
- `X_scaled`\
- `scaler`

------------------------------------------------------------------------

# 4. Baseline Clustering (src/clustering_baseline.py)

Primer model implementat:

## K-means amb k ∈ \[2..12\]

Per cada valor de k calculem:

-   **SSE / Inertia** → gràfic "Elbow"
-   **Silhouette Score** → qualitat interna del clustering

### Fitxers generats:

-   `figures/elbow_kmeans.png`
-   `figures/silhouette_kmeans.png`

El millor valor de k es tria segons el *silhouette score*.

------------------------------------------------------------------------

# 5. PCA 2D

Reducció dimensional amb PCA:

-   Permet representar els clústers en un espai de 2 dimensions.
-   Ajuda a interpretar visualment la separació entre clusters.

Gràfic generat: - `figures/pca_kmeans.png`

------------------------------------------------------------------------

# 6. Registre d'experiments (experiments.md)

Cada execució guarda una entrada amb:

-   Descripció\
-   Model i configuració\
-   Silhouette score\
-   Observacions

L'script `clustering_baseline.py` actualitza automàticament el document.

Exemple:

    | B1 | baseline | KMeans baseline, k=X | silhouette=... | Millor k segons silhouette |

------------------------------------------------------------------------

# 7. Com executar el projecte

### Instal·lar dependències:

    pip install -r requirements.txt

### Executar baseline:

    python src/clustering_baseline.py

------------------------------------------------------------------------

# 8. Estat actual del projecte

### ✔️ Fet:

-   Preprocessament complet\
-   Escalat\
-   Baseline K-means\
-   Elbow\
-   Silhouette\
-   PCA\
-   Experiments registrats

### En procés:

-   Anàlisi qualitativa de clústers\
-   Interpretació musical

### Encara no fet (properes etapes):

-   Gaussian Mixture Models (GMM)\
-   Clustering jeràrquic (Ward)\
-   Comparació entre models\
-   Conclusions finals\
-   Resultats definitius per a la presentació

------------------------------------------------------------------------

# 9. Objectiu final

Construir un sistema de clustering capaç d'analitzar patrons musicals i
agrupar cançons basant-se en:

-   Energia\
-   Dansabilitat\
-   Valence\
-   Tempo\
-   Loudness

I poder interpretar cada clúster com un estil o comportament musical.

------------------------------------------------------------------------

## 
