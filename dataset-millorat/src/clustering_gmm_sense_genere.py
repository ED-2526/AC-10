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
from data_neteja_2 import preparar_clustering  # Importa des de data_neteja_2

import pandas as pd

# -------------------------------------------------------------------
# 1. Carregar dataset processat
# -------------------------------------------------------------------

print("Carregant dades...")

# Carregar el dataset netejat
df_clean = pd.read_csv("dataset-millorat/data/processat2/06_dataset_prepared_for_clustering.csv")

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

# Llista de columnes que volem utilitzar (només numèriques)
selected_columns = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 
    'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 
    'tempo', 'valence'
]

# Filtrar les dades per incloure només les columnes seleccionades
df_clean = df_clean[selected_columns]

# Escalar les dades
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

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

# Per BIC, busquem el valor més baix (minimitzem BIC)
best_k_idx = np.argmin(bic_list)  # Canviat de argmax(silhouette_list) a argmin(bic_list)
best_k = list(K_RANGE)[best_k_idx]
best_bic = bic_list[best_k_idx]

print(f"\nMillor k segons BIC (GMM) → {best_k}  (BIC={best_bic:.2f})")

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
plt.title(f"PCA 2D - GMM (k={best_k}) sense gènere")
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

# -------------------------------------------------------------------
# 8. t-SNE 2D per visualització alternativa
# -------------------------------------------------------------------

print("\n" + "="*80)
print("CALCULANT t-SNE PER A VISUALITZACIÓ (això pot trigar uns minuts...)")
print("="*80)

# t-SNE és molt lent per a grans datasets, així que utilitzem una mostra
tsne_sample_size = 10000  # Mostra per t-SNE (més petit que per silhouette)

if n_samples > tsne_sample_size:
    print(f"Mostrejant {tsne_sample_size} punts per a t-SNE (t-senar)...")
    rng = np.random.default_rng(seed=42)
    tsne_idx = rng.choice(n_samples, size=tsne_sample_size, replace=False)
    X_tsne_sample = X_scaled[tsne_idx]
    labels_tsne_sample = best_labels_full[tsne_idx]
    use_tsne_sample = True
else:
    X_tsne_sample = X_scaled
    labels_tsne_sample = best_labels_full
    use_tsne_sample = False

# Configuració de t-SNE optimitzada per a grans datasets
print("Aplicant t-SNE (això pot trigar uns minuts...)")
tsne = TSNE(
    n_components=2,
    perplexity=30,           # Bo per a 10k mostres
    learning_rate='auto',    # Automàtic basat en la mida de mostra
    n_iter=1000,
    random_state=42,
    verbose=1,               # Mostra progrés
    n_jobs=-1                # Utilitza tots els cores
)

X_tsne = tsne.fit_transform(X_tsne_sample)

print(f"\nt-SNE complet! KL divergence: {tsne.kl_divergence_:.4f}")

# Crear figura amb PCA i t-SNE costat a costat
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gràfic PCA (esquerra)
axes[0].scatter(
    X_pca[:, 0], 
    X_pca[:, 1], 
    c=best_labels_full, 
    cmap='tab10', 
    s=10, 
    alpha=0.7
)
axes[0].set_title(f'PCA 2D - GMM (k={best_k})')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].grid(True, alpha=0.3)

# Gràfic t-SNE (dreta)
scatter = axes[1].scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1], 
    c=labels_tsne_sample, 
    cmap='tab10', 
    s=10, 
    alpha=0.7
)
axes[1].set_title(f't-SNE 2D - GMM (k={best_k})' + 
                  (f' (mostra: {tsne_sample_size})' if use_tsne_sample else ''))
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)

# Afegir barra de colors
plt.colorbar(scatter, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig(f"figures/gmm_pca_tsne_k{best_k}.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nComparació PCA-t-SNE guardada a figures/gmm_pca_tsne_k{best_k}.png")

# -------------------------------------------------------------------
# 9. t-SNE detallat amb menys mostres per a anàlisi
# -------------------------------------------------------------------

# Un t-SNE més detallat amb menys punts per veure millor l'estructura
print("\n" + "="*80)
print("CALCULANT t-SNE DETALLAT PER A ANÀLISI (5.000 punts)")
print("="*80)

tsne_detail_size = 5000
rng = np.random.default_rng(seed=123)
detail_idx = rng.choice(n_samples, size=tsne_detail_size, replace=False)
X_tsne_detail = X_scaled[detail_idx]
labels_tsne_detail = best_labels_full[detail_idx]

# t-SNE amb hiperparàmetres per a visualització detallada
tsne_detail = TSNE(
    n_components=2,
    perplexity=40,           # Una mica més alt per a visualització
    learning_rate=200,       # Fix per a millor visualització
    n_iter=1500,            # Més iteracions
    random_state=42,
    init='random',          # Inicialització aleatòria per a millors resultats
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
plt.title(f't-SNE Detallat - GMM amb k={best_k} clusters\n(5.000 punts mostrejats)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.2)
plt.colorbar(scatter, label='Cluster')
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

# -------------------------------------------------------------------
# 10. Escriure resultats al fitxer experiments.md amb info de t-SNE
# -------------------------------------------------------------------

EXPERIMENTS_PATH = "experiments.md"

# Incloure informació de les característiques més rellevants
top_features_summary = ", ".join([row['feature'] for idx, row in combined_loadings.head(5).iterrows()])

entry = f"""
| ID  | Data       | Descripció                               | Model / Config                     | Resultat (BIC)               | Característiques rellevants                        | Comentaris                         |
|-----|------------|-------------------------------------------|------------------------------------|------------------------------|---------------------------------------------------|------------------------------------|
| G1  | baseline   | GMM baseline, k={best_k}                  | GMM (covariance_type='full')       | {best_bic:.2f}               | {top_features_summary}                            | Millor k segons BIC + t-SNE inclòs |
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
print("ANÀLISI GMM COMPLETAT!")
print(f"Millor k: {best_k}")
print(f"Gràfics generats:")
print(f"  - figures/gmm_bic.png")
print(f"  - figures/gmm_silhouette.png")
print(f"  - figures/gmm_pca_k{best_k}.png")
print(f"  - figures/gmm_pca_tsne_k{best_k}.png")
print(f"  - figures/gmm_tsne_detailed_k{best_k}.png")
print("="*80)
