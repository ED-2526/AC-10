import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# --- CONFIGURACIÓ ---
# Assegura't que el nom del fitxer coincideix amb el que tens a la carpeta
DATA_FILE = 'SpotifyFeatures.csv' 

def load_and_preprocess_data(filepath):
    """
    Carrega les dades, selecciona les columnes numèriques i les normalitza.
    """
    print(f"📂 Carregant dades de: {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("❌ Error: No s'ha trobat l'arxiu. Comprova la ruta.")
        return None, None, None

    # Seleccionem només les característiques d'àudio rellevants per al clustering
    # Ajusta aquesta llista segons les columnes exactes del teu CSV
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Comprovem que les columnes existeixin
    available_features = [col for col in features if col in df.columns]
    
    # Eliminem files amb valors nuls si n'hi ha
    df_clean = df.dropna(subset=available_features).reset_index(drop=True)
    X = df_clean[available_features]

    # Normalització (StandardScaler) - Molt important per a K-Means i DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Dades processades. Shape: {X_scaled.shape}")
    return df_clean, X_scaled, scaler

def apply_pca(X_scaled, n_components=2):
    """
    Redueix la dimensionalitat a 2 components per poder visualitzar.
    """
    print("📉 Aplicant PCA...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def find_optimal_k_elbow(X_scaled, max_k=10):
    """
    Mostra el gràfic del colze (Elbow Method) per decidir el número de clusters.
    """
    print("🔍 Calculant el mètode del colze...")
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='--')
    plt.title('Mètode del Colze (Elbow Method)')
    plt.xlabel('Nombre de Clusters (K)')
    plt.ylabel('SSE (Inertia)')
    plt.grid(True)
    print("📊 Tanca la finestra del gràfic per continuar...")
    plt.show()

def run_kmeans(df, X_pca, n_clusters=4):
    """
    Executa K-Means i visualitza els resultats.
    """
    print(f"🚀 Executant K-Means amb K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    
    # Entrenem sobre les dades reduïdes per PCA (o sobre X_scaled si preferim)
    # En aquest exemple usem X_pca per coherència visual
    clusters = kmeans.fit_predict(X_pca)
    df['KMeans_Cluster'] = clusters

    # Visualització
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=60, legend='full')
    
    # Dibuixar centroides
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroides')
    
    plt.title(f'K-Means Clustering (K={n_clusters})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
    
    return df

def run_dbscan(df, X_pca, eps=0.5, min_samples=5):
    """
    Executa DBSCAN i visualitza els resultats.
    """
    print(f"🚀 Executant DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_pca)
    df['DBSCAN_Cluster'] = clusters

    # Visualització
    plt.figure(figsize=(10, 6))
    # El cluster -1 és soroll (noise) en DBSCAN
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='plasma', s=60, legend='full')
    plt.title(f'DBSCAN Clustering')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    n_noise = list(clusters).count(-1)
    print(f"ℹ️ DBSCAN ha trobat {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters.")
    print(f"⚠️ Punts considerats soroll (outliers): {n_noise}")
    
    return df

# --- EXECUCIÓ PRINCIPAL ---
if __name__ == "__main__":
    # 1. Càrrega
    df, X_scaled, scaler = load_and_preprocess_data(DATA_FILE)

    if df is not None:
        # 2. PCA
        X_pca = apply_pca(X_scaled)

        # 3. Determinar K (Opcional, pots comentar aquesta línia si ja saps la K)
        find_optimal_k_elbow(X_scaled)

        # 4. Executar K-Means (Canvia n_clusters segons el que vegis al gràfic anterior)
        df = run_kmeans(df, X_pca, n_clusters=4)

        # 5. Executar DBSCAN (Ajusta eps i min_samples segons els resultats)
        df = run_dbscan(df, X_pca, eps=0.5, min_samples=5)
        
        print("\n✅ Procés finalitzat amb èxit!")