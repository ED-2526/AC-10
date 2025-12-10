import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importo el pipeline de preprocessament
from data_neteja_2 import preparar_clustering  # Importa des de data_neteja_2

import pandas as pd

# -------------------------------------------------------------------
# 1. Carregar dataset processat
# -------------------------------------------------------------------

print("Carregant dades...")

# Carregar el dataset netejat
df_clean = pd.read_csv("dataset-millorat/data/processat/06_dataset_prepared_for_clustering.csv")

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
max_sil_samples = 10000

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
        n_init=1                    
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
# 6. Triar el millor k (segons silhouette)
# -------------------------------------------------------------------

best_k_idx = np.argmax(silhouette_list)
best_k = list(K_RANGE)[best_k_idx]
best_sil = silhouette_list[best_k_idx]

print(f"\nMillor k segons silhouette (GMM) → {best_k}  (silhouette={best_sil:.4f})")

# Re-entreno el GMM amb best_k per poder fer PCA i analitzar
best_gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42,
    n_init=1
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
plt.title(f"PCA 2D - GMM (k={best_k})")
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

# Guardar els resultats en un fitxer CSV
os.makedirs("results", exist_ok=True)
csv_path = f"results/pca_feature_importance_k{best_k}.csv"
combined_loadings.to_csv(csv_path, index=False)
print(f"\nResultats de l'anàlisi de característiques guardats a: {csv_path}")

# -------------------------------------------------------------------
# 8. Escriure resultats al fitxer experiments.md
# -------------------------------------------------------------------

EXPERIMENTS_PATH = "experiments.md"

# Incloure informació de les característiques més rellevants
top_features_summary = ", ".join([row['feature'] for idx, row in combined_loadings.head(5).iterrows()])

entry = f"""
| ID  | Data       | Descripció                               | Model / Config                     | Resultat (silhouette mostra) | Característiques rellevants                        | Comentaris                         |
|-----|------------|-------------------------------------------|------------------------------------|------------------------------|---------------------------------------------------|------------------------------------|
| G1  | baseline   | GMM baseline, k={best_k}                  | GMM (covariance_type='full')       | {best_sil:.4f}               | {top_features_summary}                            | Millor k segons silhouette (mostra) |
"""

# Si experiments.md ja existeix, l’afegeixo al final
if os.path.exists(EXPERIMENTS_PATH):
    with open(EXPERIMENTS_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + entry)
else:
    # Si no existeix, el creo amb la capçalera
    with open(EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Registre d'experiments\n\n")
        f.write(entry)

print("\nResultats de GMM afegits a experiments.md")

# -------------------------------------------------------------------
print("\nGMM complet! Gràfics generats i experiment registrat.\n")
