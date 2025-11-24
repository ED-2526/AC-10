import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- CONFIGURACIÓ ---
DATA_FILE = 'SpotifyFeatures.csv' # Assegura't que el nom és correcte

def load_data(filepath):
    """Carrega el dataset i mostra informació bàsica."""
    print(f"📂 Carregant dades de: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"   Shape inicial: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Error: No s'ha trobat l'arxiu CSV.")
        return None

def exploratory_data_analysis(df):
    """
    Realitza l'Anàlisi Exploratòria (EDA) típica d'aquest notebook.
    Mostra la matriu de correlació (Important per AE).
    """
    print("📊 Generant matriu de correlació...")
    
    # Seleccionem només columnes numèriques per la correlació
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 10))
    # Calculem la correlació
    corr_matrix = numeric_df.corr()
    
    # Dibuixem el heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Matriu de Correlació de les Característiques d\'Àudio')
    plt.show()

def preprocess_data(df):
    """Neteja i normalitza les dades."""
    print("🧹 Preprocessant les dades...")
    
    # Selecció de features típiques d'aquest notebook
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Filtrar columnes existents
    existing_features = [col for col in features if col in df.columns]
    df_clean = df.dropna(subset=existing_features).reset_index(drop=True)
    X = df_clean[existing_features]
    
    # Escalat (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df_clean, X_scaled

def evaluate_kmeans(X_scaled, k_range=range(2, 11)):
    """
    Avalua K-Means usant el mètode de l'Elbow i el Silhouette Score.
    Això correspon a la secció 'Model Evaluation' del teu notebook.
    """
    print("🔬 Avaluant K-Means (Elbow i Silhouette)...")
    
    sse = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"   K={k}: Silhouette Score = {score:.4f}")

    # Gràfic 1: Elbow
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, sse, marker='o')
    plt.title('Mètode del Colze (Inertia)')
    plt.xlabel('K')
    plt.ylabel('SSE')
    
    # Gràfic 2: Silhouette
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score (Més alt és millor)')
    plt.xlabel('K')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.show()

def run_clustering_models(df, X_scaled):
    """
    Executa els models finals i visualitza amb PCA.
    """
    # Per visualitzar en 2D necessitem PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # --- MODEL 1: K-MEANS (Suposem K=4 o 5 basat en l'anàlisi previ) ---
    k_final = 5
    print(f"🚀 Entrenant K-Means final amb K={k_final}...")
    kmeans = KMeans(n_clusters=k_final, random_state=42, n_init='auto')
    clusters_kmeans = kmeans.fit_predict(X_scaled)
    df['cluster_kmeans'] = clusters_kmeans
    
    # --- MODEL 2: DBSCAN ---
    # Nota: DBSCAN és sensible als paràmetres. 
    eps_val = 0.5
    min_samples_val = 5
    print(f"🚀 Entrenant DBSCAN (eps={eps_val}, min_samples={min_samples_val})...")
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    clusters_dbscan = dbscan.fit_predict(X_scaled)
    df['cluster_dbscan'] = clusters_dbscan
    
    # --- VISUALITZACIÓ ---
    plt.figure(figsize=(16, 6))
    
    # Plot K-Means
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_kmeans, palette='viridis', s=50)
    plt.title(f'Resultat K-Means (K={k_final})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Plot DBSCAN
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_dbscan, palette='plasma', s=50)
    plt.title('Resultat DBSCAN')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Flux principal d'execució
    df = load_data(DATA_FILE)
    
    if df is not None:
        # 1. EDA
        exploratory_data_analysis(df)
        
        # 2. Preprocessament
        df_clean, X_scaled = preprocess_data(df)
        
        # 3. Avaluació de Models (Important per AC)
        evaluate_kmeans(X_scaled, k_range=range(2, 10))
        
        # 4. Execució Final i Visualització
        run_clustering_models(df_clean, X_scaled)
        
        print("\n✅ Anàlisi completa finalitzada.")