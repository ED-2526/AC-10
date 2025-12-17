# clustering_gmm.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # <-- Afegit per t-SNE
from sklearn.preprocessing import StandardScaler

# Importo el pipeline de preprocessament
from data_neteja import preparar_clustering  # Importa des de data_neteja

import pandas as pd

# -------------------------------------------------------------------
# 1. Carregar dataset processat
# -------------------------------------------------------------------

print("Carregant dades...")

# Carregar el dataset netejat
df_clean = pd.read_csv("dataset-millorat/data/processat/05_final_clean_dataset.csv")

# preparar_clustering() fa: carrega CSV -> neteja -> selecciona info/features -> escala
info_df, X_scaled, scaler = preparar_clustering(df_clean)  # Passant df_clean a preparar_clustering()
# info_df = DF de les dades informatives, sense escalar.
# X_scaled = matriu de característiques escalades, transformades amb StandarScaler, entre 0 i 1.
# scaler = objecte StandarScaler, aprèn la mitja i desviació, es guarda per poder reutilitzar-ho a l'hora d'escalar noves dades o prediccions.

# Mostro la forma de la matriu X_scaled
print("Forma de X_scaled:", X_scaled.shape)

# -------------------------------------------------------------------
# 1.1 Definir els noms de les característiques basat en el teu dataset
# -------------------------------------------------------------------

# Llista de columnes que volem utilitzar (només numèriques i les de gènere multilabel)
selected_columns = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 
    'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 
    'tempo', 'valence', 'A Capella', 'Alternative', 'Anime', 'Blues', 
    "Children's Music", 'Children\'s Music', 'Classical', 'Comedy', 'Country', 
    'Dance', 'Electronic', 'Folk', 'Hip-Hop', 'Indie', 'Jazz', 'Movie', 
    'Opera', 'Pop', 'R&B', 'Rap', 'Reggae', 'Reggaeton', 'Rock', 'Ska', 
    'Soul', 'Soundtrack', 'World'
]

# Filtrar les dades per incloure només les columnes seleccionades
df_clean = df_clean[selected_columns]

# ==============================================================================
# AFEGIR AIXÒ AQUÍ (ABANS D'ESCALAR)
# ==============================================================================
print("\n" + "="*50)
print("1. VARIÀNCIA ABANS D'ESCALAR (Dades reals)")
print("="*50)
# Això mostra quines característiques dominarien si no escaléssim (ex: duration_ms)
raw_vars = df_clean.var()
raw_pcts = (raw_vars / raw_vars.sum()) * 100
print(raw_pcts.sort_values(ascending=False).head(10))
print("Note: Si una variable té el 99%, el model ignoraria la resta.\n")
# ==============================================================================

# Escalar les dades
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# ==============================================================================
# AFEGIR AIXÒ AQUÍ (DESPRÉS D'ESCALAR)
# ==============================================================================
print("\n" + "="*50)
print("2. VARIÀNCIA DESPRÉS D'ESCALAR (Dades normalitzades)")
print("="*50)
# Ara comprovem que totes pesen igual
scaled_vars = np.var(X_scaled, axis=0)
scaled_pcts = (scaled_vars / np.sum(scaled_vars)) * 100

# Creem un DataFrame ràpid per visualitzar-ho amb els noms
df_vars_scaled = pd.DataFrame({
    'Feature': selected_columns[:X_scaled.shape[1]], # Assegurem que coincideixin dimensions
    'Variance_%': scaled_pcts
}).sort_values('Variance_%', ascending=False)

print(df_vars_scaled.head(5))
print(f"\nVariància mitjana per feature: {scaled_pcts.mean():.4f}%")
print("Ara totes les variables tenen la mateixa importància pel clustering.")
print("="*50 + "\n")
# ==============================================================================

# Verificar que les columnes coincideixen amb la matriu escalada
if len(selected_columns) != X_scaled.shape[1]:
    print(f"Avís: Nombre de columnes no coincideix! Esperat: {X_scaled.shape[1]}, Obtingut: {len(selected_columns)}")
    if len(selected_columns) > X_scaled.shape[1]:
        selected_columns = selected_columns[:X_scaled.shape[1]]
    else:
        selected_columns.extend([f'feature_{i}' for i in range(len(selected_columns), X_scaled.shape[1])])


print(f"\nNombre de característiques: {len(selected_columns)}")
print(f"Primeres 5 característiques: {selected_columns[:5]}")
print(f"Últimes 5 característiques: {selected_columns[-5:]}")

# -------------------------------------------------------------------
# 2. Definir rang de k i mostra per al silhouette
# -------------------------------------------------------------------

K_RANGE = range(2, 13)

bic_list = []          # valors de BIC (Bayesian Information Criterion) per cada valor de k
silhouette_list = []   # Silhouette (calculat sobre una mostra) --> Com de ben separats estan els clústers

# Mostreig per al silhouette per no morir de temps
# ------------------------------------------------
# Si tenim moltes mostres (230k), calcular silhouette sobre totes és molt costós.
# Per això agafo una mostra aleatòria de max 10.000 punts.
max_sil_samples = 30000

n_samples, n_features = X_scaled.shape
if n_samples > max_sil_samples:
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(n_samples, size=max_sil_samples, replace=False)
    X_sil = X_scaled[sample_idx]
    use_sample = True
    print(f"Faré servir una mostra de {max_sil_samples} punts per al silhouette.")
else:
    X_sil = X_scaled
    sample_idx = np.arange(n_samples)
    use_sample = False
    print("Faré servir totes les mostres per al silhouette.")

# -------------------------------------------------------------------
# 3. Bucle sobre k: ajustar GMM, calcular BIC i silhouette
# -------------------------------------------------------------------

print("\nCalculant GMM per diferents k...\n")

for k in K_RANGE:
    print(f"-> Entrenant GMM amb k={k} components...")

    gmm = GaussianMixture(
        n_components=k,             
        covariance_type="full",     
        random_state=42,            
        n_init=10                    
    )

    gmm.fit(X_scaled)

    bic = gmm.bic(X_scaled)
    bic_list.append(bic)

    labels_full = gmm.predict(X_scaled)

    labels_sil = labels_full[sample_idx]
    sil = silhouette_score(X_sil, labels_sil)
    silhouette_list.append(sil)

    print(f"   BIC={bic:.2f}  Silhouette(sample)={sil:.4f}")

# -------------------------------------------------------------------
# 4. Generar gràfic BIC vs k
# -------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), bic_list, marker="o")
plt.xlabel("Nombre de components (k)")
plt.ylabel("BIC")
plt.title("GMM - BIC per diferents k")
plt.grid(True)

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/gmm_bic.png", dpi=150)
plt.close()

print("\nGràfic BIC guardat a figures/gmm_bic.png")

# -------------------------------------------------------------------
# 5. Generar gràfic Silhouette vs k
# -------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), silhouette_list, marker="o", color="purple")
plt.xlabel("Nombre de components (k)")
plt.ylabel("Silhouette Score (sobre mostra)")
plt.title("GMM - Silhouette Score per diferents k")
plt.grid(True)

plt.savefig("figures/gmm_silhouette.png", dpi=150)
plt.close()

print("Gràfic Silhouette guardat a figures/gmm_silhouette.png")

# -------------------------------------------------------------------
# 6. Triar el millor k (segons BIC)
# -------------------------------------------------------------------

best_k_idx_bic = np.argmin(bic_list)  # BIC menor és millor
best_k = list(K_RANGE)[best_k_idx_bic]
best_bic = bic_list[best_k_idx_bic]
best_sil = silhouette_list[best_k_idx_bic]

print(f"\nMillor k segons BIC (GMM) → {best_k}  (BIC={best_bic:.2f}, silhouette={best_sil:.4f})")

# Re-entreno el GMM amb best_k per poder fer PCA i analitzar
best_gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42,
    n_init=10
)
best_gmm.fit(X_scaled)
best_labels_full = best_gmm.predict(X_scaled)

# -------------------------------------------------------------------
# 7. PCA 2D per visualitzar els clústers de GMM
# -------------------------------------------------------------------

# PCA 2D per visualitzar els clústers de GMM
print("\nCalculant PCA 2D per a GMM...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=best_labels_full,
    palette="tab10",
    s=10,
    alpha=0.7
)
plt.title(f"PCA 2D - GMM (k={best_k}) [Seleccionat per BIC]")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"figures/gmm_pca_k{best_k}.png", dpi=150)
plt.close()

print(f"PCA 2D guardat a figures/gmm_pca_k{best_k}.png")

# -------------------------------------------------------------------
# 7.1 Anàlisi de les característiques més rellevants en el PCA
# -------------------------------------------------------------------

print("\n" + "="*80)
print("ANÀLISI DE CARACTERÍSTIQUES RELLEVANTS EN EL PCA")
print("="*80)

# Obtenir els components del PCA
pca_components = pca.components_

# Obtenir la variància explicada per cada component
explained_variance = pca.explained_variance_ratio_

print(f"\nVariància explicada per cada component:")
for i, var in enumerate(explained_variance):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\nVariància total explicada: {sum(explained_variance):.4f} ({sum(explained_variance)*100:.2f}%)")

# Anàlisi per a cada component principal
for comp_idx in range(pca.n_components_):
    print(f"\n{'='*60}")
    print(f"COMPONENT PRINCIPAL {comp_idx+1} (explica {explained_variance[comp_idx]*100:.2f}% de la variància)")
    print("="*60)
    
    # Obtenir els pesos (loadings) per aquest component
    loadings = pca_components[comp_idx]
    
    # Crear un DataFrame per visualitzar els loadings amb noms reals
    loadings_df = pd.DataFrame({
        'feature': selected_columns,  # Utilitzem els noms reals
        'loading': loadings,
        'abs_loading': np.abs(loadings)
    })
    
    # Ordenar per importància absoluta
    loadings_df = loadings_df.sort_values('abs_loading', ascending=False)
    
    print(f"\nTop 10 característiques més rellevants per a PC{comp_idx+1}:")
    print("-"*80)
    for idx, row in loadings_df.head(10).iterrows():
        sign = "+" if row['loading'] >= 0 else "-"
        print(f"  {row['feature']}: {sign}{abs(row['loading']):.4f}")
    
    print(f"\nBottom 5 característiques menys rellevants per a PC{comp_idx+1}:")
    print("-"*80)
    for idx, row in loadings_df.tail(5).iterrows():
        sign = "+" if row['loading'] >= 0 else "-"
        print(f"  {row['feature']}: {sign}{abs(row['loading']):.4f}")

# Anàlisi combinada per als dos components
print(f"\n{'='*80}")
print("ANÀLISI COMBINADA PER ALS DOS COMPONENTS PRINCIPALS")
print("="*80)

# Crear un DataFrame amb els loadings dels dos components amb noms reals
combined_loadings = pd.DataFrame({
    'feature': selected_columns,  # Utilitzem els noms reals
    'PC1_loading': pca_components[0],
    'PC2_loading': pca_components[1],
    'PC1_abs': np.abs(pca_components[0]),
    'PC2_abs': np.abs(pca_components[1])
})

# Calcular una puntuació combinada (ponderant per la variància explicada)
combined_loadings['combined_score'] = (
    combined_loadings['PC1_abs'] * explained_variance[0] + 
    combined_loadings['PC2_abs'] * explained_variance[1]
)

# Ordenar per la puntuació combinada
combined_loadings = combined_loadings.sort_values('combined_score', ascending=False)

print(f"\nTop 15 característiques més rellevants per a la formació de clusters (combinat):")
print("-"*120)
print(f"{'Característica':<40} | {'PC1 Loading':<12} | {'PC2 Loading':<12} | {'Score Combi':<10} |")
print("-"*120)
for idx, row in combined_loadings.head(15).iterrows():
    pc1_sign = "+" if row['PC1_loading'] >= 0 else "-"
    pc2_sign = "+" if row['PC2_loading'] >= 0 else "-"
    pc1_abs = f"{abs(row['PC1_loading']):.4f}"
    pc2_abs = f"{abs(row['PC2_loading']):.4f}"
    combined_score = f"{row['combined_score']:.4f}"
    
    # Truncar noms llargs si cal
    feature_name = row['feature']
    if len(feature_name) > 40:
        feature_name = feature_name[:37] + "..."
    
    print(f"{feature_name:<40} | {pc1_sign}{pc1_abs:<11} | {pc2_sign}{pc2_abs:<11} | {combined_score:<10} |")

# Anàlisi detallada per a les 5 característiques més importants
print(f"\n{'='*80}")
print("ANÀLISI DETALLADA DE LES 5 CARACTERÍSTIQUES MÉS IMPORTANTS")
print("="*80)

top_5 = combined_loadings.head(5)
for idx, row in top_5.iterrows():
    pc1_direction = "positiva" if row['PC1_loading'] >= 0 else "negativa"
    pc2_direction = "positiva" if row['PC2_loading'] >= 0 else "negativa"
    print(f"\n{row['feature']}:")
    print(f"  - Contribució a PC1: {abs(row['PC1_loading']):.4f} ({pc1_direction})")
    print(f"  - Contribució a PC2: {abs(row['PC2_loading']):.4f} ({pc2_direction})")
    print(f"  - Puntuació combinada: {row['combined_score']:.4f}")

# Interpretació de les característiques més importants
print(f"\n{'='*80}")
print("INTERPRETACIÓ DE LES CARACTERÍSTIQUES MÉS IMPORTANTS PER A CLUSTERS")
print("="*80)

print("\nLes característiques que més influeixen en la formació de clústers són:")
for i, (idx, row) in enumerate(top_5.iterrows()):
    print(f"\n{i+1}. {row['feature']}:")
    
    # Interpretació basada en el signe i magnitud
    pc1_mag = abs(row['PC1_loading'])
    pc2_mag = abs(row['PC2_loading'])
    
    if pc1_mag > pc2_mag:
        print(f"   - Principalment influeix en la separació horitzontal (PC1)")
        if row['PC1_loading'] > 0:
            print(f"   - Valors alts d'aquesta característica mouen els punts cap a la dreta")
        else:
            print(f"   - Valors alts d'aquesta característica mouen els punts cap a l'esquerra")
    else:
        print(f"   - Principalment influeix en la separació vertical (PC2)")
        if row['PC2_loading'] > 0:
            print(f"   - Valors alts d'aquesta característica mouen els punts cap amunt")
        else:
            print(f"   - Valors alts d'aquesta característica mouen els punts cap avall")

# Guardar els resultats en un fitxer CSV
os.makedirs("results", exist_ok=True)
csv_path = f"results/pca_feature_importance_k{best_k}.csv"
combined_loadings.to_csv(csv_path, index=False)
print(f"\nResultats de l'anàlisi de característiques guardats a: {csv_path}")

# -------------------------------------------------------------------
# 8. t-SNE PER VISUALITZACIÓ AVANÇADA
# -------------------------------------------------------------------

print("\n" + "="*80)
print("CALCULANT t-SNE PER A VISUALITZACIÓ AVANÇADA")
print("="*80)

# Configuració per a t-SNE (optimitzat per a grans datasets)
tsne_sample_size = 10000  # Mostra per t-SNE (més ràpid que utilitzar totes les dades)
tsne_perplexity = 30     # Bo per a 10k mostres

if n_samples > tsne_sample_size:
    print(f"Mostrejant {tsne_sample_size} punts per a t-SNE (això pot trigar uns minuts)...")
    rng = np.random.default_rng(seed=123)
    tsne_idx = rng.choice(n_samples, size=tsne_sample_size, replace=False)
    X_tsne_sample = X_scaled[tsne_idx]
    labels_tsne_sample = best_labels_full[tsne_idx]
    use_tsne_sample = True
else:
    X_tsne_sample = X_scaled
    labels_tsne_sample = best_labels_full
    use_tsne_sample = False

# t-SNE amb hiperparàmetres optimitzats
print("\nAplicant t-SNE (això pot trigar uns minuts)...")
tsne = TSNE(
    n_components=2,
    perplexity=tsne_perplexity,
    learning_rate='auto',    # Ajust automàtic basat en la mida de mostra
    n_iter=1000,
    random_state=42,
    init='random',          # Inicialització aleatòria per a millors resultats
    verbose=1,               # Mostra progrés
    n_jobs=-1                # Utilitza tots els cores
)

X_tsne = tsne.fit_transform(X_tsne_sample)
print(f"t-SNE complet! KL divergence: {tsne.kl_divergence_:.4f}")

# -------------------------------------------------------------------
# 9. VISUALITZACIÓ COMPARATIVA PCA vs t-SNE
# -------------------------------------------------------------------

print("\n" + "="*80)
print("GENERANT VISUALITZACIÓ COMPARATIVA PCA vs t-SNE")
print("="*80)

# Crear figura amb PCA i t-SNE costat a costat
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gràfic PCA (esquerra)
scatter1 = axes[0].scatter(
    X_pca[:, 0], 
    X_pca[:, 1], 
    c=best_labels_full, 
    cmap='tab10', 
    s=10, 
    alpha=0.7
)
axes[0].set_title(f'PCA 2D - GMM amb {best_k} clústers')
axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variància)')
axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variància)')
axes[0].grid(True, alpha=0.3)

# Gràfic t-SNE (dreta)
scatter2 = axes[1].scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1], 
    c=labels_tsne_sample, 
    cmap='tab10', 
    s=10, 
    alpha=0.7
)
axes[1].set_title(f't-SNE 2D - GMM amb {best_k} clústers' + 
                  (f' (mostra: {tsne_sample_size})' if use_tsne_sample else ''))
axes[1].set_xlabel('t-SNE Component 1')
axes[1].set_ylabel('t-SNE Component 2')
axes[1].grid(True, alpha=0.3)

# Afegir barres de colors
plt.colorbar(scatter1, ax=axes[0], label='Cluster')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig(f"figures/gmm_pca_vs_tsne_k{best_k}.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"Comparació PCA-t-SNE guardada a figures/gmm_pca_vs_tsne_k{best_k}.png")

# -------------------------------------------------------------------
# 10. t-SNE DETALLAT PER A ANÀLISI
# -------------------------------------------------------------------

print("\n" + "="*80)
print("CALCULANT t-SNE DETALLAT PER A ANÀLISI PROFUNDA")
print("="*80)

# Un t-SNE més detallat amb menys punts per veure millor l'estructura
tsne_detail_size = 5000
rng = np.random.default_rng(seed=456)
detail_idx = rng.choice(n_samples, size=tsne_detail_size, replace=False)
X_tsne_detail = X_scaled[detail_idx]
labels_tsne_detail = best_labels_full[detail_idx]

# t-SNE amb hiperparàmetres per a visualització detallada
tsne_detail = TSNE(
    n_components=2,
    perplexity=50,           # Una mica més alt per a visualització detallada
    learning_rate=200,       # Valor fix per a millor visualització
    n_iter=1500,            # Més iteracions
    random_state=42,
    init='random',
    verbose=1,
    n_jobs=-1
)

X_tsne_detailed = tsne_detail.fit_transform(X_tsne_detail)

# Gràfic t-SNE detallat
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_tsne_detailed[:, 0], 
    X_tsne_detailed[:, 1], 
    c=labels_tsne_detail, 
    cmap='tab20',           # Més colors per a més clústers
    s=15, 
    alpha=0.8,
    edgecolors='w',
    linewidth=0.3
)
plt.title(f't-SNE Detallat - GMM amb {best_k} clústers\n(5.000 punts mostrejats, KL divergence: {tsne_detail.kl_divergence_:.4f})')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.2)
plt.colorbar(scatter, label='Cluster', ticks=range(best_k))

# Afegir anotacions per als clústers principals (opcional)
if best_k <= 15:  # Només si no hi ha massa clústers
    # Trobar els centroides aproximats de cada clúster
    for cluster_id in range(best_k):
        cluster_points = X_tsne_detailed[labels_tsne_detail == cluster_id]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            plt.annotate(f'C{cluster_id}', xy=centroid, xytext=(5, 5),
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f"figures/gmm_tsne_detailed_k{best_k}.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"t-SNE detallat guardat a figures/gmm_tsne_detailed_k{best_k}.png")

# Anàlisi de la distribució de clústers en t-SNE
print(f"\n{'='*80}")
print("ANÀLISI DE DISTRIBUCIÓ DE CLÚSTERS EN t-SNE")
print("="*80)

cluster_counts = np.bincount(labels_tsne_detail)
for cluster_id in range(best_k):
    count = cluster_counts[cluster_id]
    percentage = (count / tsne_detail_size) * 100
    print(f"Cluster {cluster_id}: {count} punts ({percentage:.1f}%)")

# Identificar clústers ben separats vs solapats
print(f"\n{'='*80}")
print("QUALITAT DE SEPARACIÓ DE CLÚSTERS (anàlisi qualitativa)")
print("="*80)

# Anàlisi qualitativa basada en la dispersió de t-SNE
print("\nObservacions basades en la visualització t-SNE:")
print("1. Clústers ben separats: Grups compactes clarament distingibles")
print("2. Clústers solapats: Àrees on múltiples clústers es superposen")
print("3. Outliers: Punts aïllats lluny de qualsevol grup principal")

# -------------------------------------------------------------------
# 11. VISUALITZACIÓ 3D PCA (OPCIONAL)
# -------------------------------------------------------------------

# Opcional: PCA 3D si tens poques dimensions de clústers
if best_k <= 10:  # Només si no hi ha massa clústers
    print(f"\n{'='*80}")
    print("GENERANT PCA 3D (opcional)")
    print("="*80)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    explained_variance_3d = pca_3d.explained_variance_ratio_
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        X_pca_3d[:, 0], 
        X_pca_3d[:, 1], 
        X_pca_3d[:, 2], 
        c=best_labels_full, 
        cmap='tab10', 
        s=10, 
        alpha=0.7
    )
    
    ax.set_title(f'PCA 3D - GMM amb {best_k} clústers')
    ax.set_xlabel(f'PC1 ({explained_variance_3d[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance_3d[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({explained_variance_3d[2]*100:.1f}%)')
    
    plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.6)
    plt.tight_layout()
    plt.savefig(f"figures/gmm_pca_3d_k{best_k}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"PCA 3D guardat a figures/gmm_pca_3d_k{best_k}.png")

# -------------------------------------------------------------------
# 12. ESCRIURE RESULTATS AL FITXER EXPERIMENTS.MD
# -------------------------------------------------------------------

EXPERIMENTS_PATH = "experiments.md"

# Incloure informació de les característiques més rellevants
top_features_summary = ", ".join([row['feature'] for idx, row in combined_loadings.head(5).iterrows()])

# Informació addicional sobre t-SNE
tsne_info = f"t-SNE sample: {tsne_sample_size}" if use_tsne_sample else "t-SNE full data"

entry = f"""
| ID  | Data       | Descripció                               | Model / Config                     | Resultat (BIC) | Característiques rellevants                        | Comentarios                         |
|-----|------------|-------------------------------------------|------------------------------------|----------------|---------------------------------------------------|------------------------------------|
| G1  | baseline   | GMM amb gèneres, k={best_k}               | GMM (covariance_type='full')       | {best_bic:.2f} | {top_features_summary}                            | Millor k segons BIC. {tsne_info} |
"""

# Si experiments.md ja existeix, l'afegeixo al final
if os.path.exists(EXPERIMENTS_PATH):
    with open(EXPERIMENTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + entry)
else:
    # Si no existeix, el creo amb la capçalera
    with open(EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Registre d'experiments\n\n")
        f.write(entry)

print("\nResultats de GMM amb t-SNE afegits a experiments.md")

# -------------------------------------------------------------------
print("\n" + "="*80)
print("ANÀLISI GMM COMPLETAT AMB t-SNE!")
print(f"Millor k: {best_k} (seleccionat per BIC)")
print(f"Silhouette: {best_sil:.4f}")
print(f"BIC: {best_bic:.2f}")
print(f"\nGràfics generats:")
print(f"  - figures/gmm_bic.png")
print(f"  - figures/gmm_silhouette.png")
print(f"  - figures/gmm_pca_k{best_k}.png")
print(f"  - figures/gmm_pca_vs_tsne_k{best_k}.png")
print(f"  - figures/gmm_tsne_detailed_k{best_k}.png")
if best_k <= 10:
    print(f"  - figures/gmm_pca_3d_k{best_k}.png")
print(f"  - results/pca_feature_importance_k{best_k}.csv")
print("="*80)
