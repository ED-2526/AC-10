"""
clustering_gmm.py

Script per provar Gaussian Mixture Models (GMM) com a model de clustering:

- Carrega i processa dades amb data_prep.py
- Prova valors de k (2..12) amb GMM
- Calcula BIC per cada k (criteri de model)
- Calcula silhouette score (sobre una mostra per accelerar)
- PCA 2D i guarda figura
- Escriu resultats al fitxer experiments.md

Projecte: Music Listener Clustering
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Importo el pipeline de preprocessament
from data_prep import prepare_dataset


# -------------------------------------------------------------------
# 1. Carregar dataset processat
# -------------------------------------------------------------------

print("Carregant dades...")
info_df, X_scaled, scaler = prepare_dataset()
n_samples, n_features = X_scaled.shape
print(f"Forma de X_scaled: {X_scaled.shape}")

# -------------------------------------------------------------------
# 2. Definir rang de k i mostra per al silhouette
# -------------------------------------------------------------------

# Provaré k entre 2 i 12 (mateix rang que a K-Means)
K_RANGE = range(2, 13)

# Aquí guardaré mètriques
bic_list = []          # valors de BIC (Bayesian Information Criterion) per cada valor de k
silhouette_list = []   # Silhouette (calculat sobre una mostra) --> Com de ben separats estan els clústers

# Mostreig per al silhouette per no morir de temps
# ------------------------------------------------
# Si tenim moltes mostres (230k), calcular silhouette sobre totes és molt costós.
# Per això agafo una mostra aleatòria de max 10.000 punts.
max_sil_samples = 10000

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
    print(" faré servir totes les mostres per al silhouette.")


# -------------------------------------------------------------------
# 3. Bucle sobre k: ajustar GMM, calcular BIC i silhouette
# -------------------------------------------------------------------

print("\nCalculant GMM per diferents k...\n")

for k in K_RANGE:
    print(f"-> Entrenant GMM amb k={k} components...")

    # Definim el model GMM --> model probabilístic que assumeix que les dades provenen de distribucions normals (gaussianes).
    gmm = GaussianMixture(
        n_components=k,             # Cada component serà una distribució gaussiana que modelarà una part del conjunt de dades
        covariance_type="full",     # Cada component gaussià pot tenir una covariància independent (formes ellíptiques diferents)
        random_state=42,            # Per reproducibilitat
        n_init=1                    # es podria pujar si volem més estabilitat
    )

    # Ajusto el GMM sobre TOTES les dades escalades
    gmm.fit(X_scaled)

    # BIC (Bayesian Information Criterion): més baix = millor
    bic = gmm.bic(X_scaled)
    bic_list.append(bic)

    # Predicció de labels per a totes les mostres
    labels_full = gmm.predict(X_scaled)  # conté les etiquetes (clusters) assignades a cada cançó

    # Per al silhouette, faig servir només la mostra X_sil
    # agafo els labels corresponents a sample_idx
    labels_sil = labels_full[sample_idx]
    sil = silhouette_score(X_sil, labels_sil)
    silhouette_list.append(sil)

    print(f"   BIC={bic:.2f}  Silhouette(sample)={sil:.4f}")


# -------------------------------------------------------------------
# 4. Gràfic BIC vs k (equivalent a un 'elbow' per GMM)
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
# 5. Gràfic Silhouette vs k
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
# 6. Triar el millor k (segons silhouette) i re-entrenar GMM
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
plt.savefig("figures/gmm_pca_k{}.png".format(best_k), dpi=150)
plt.close()

print(f"PCA 2D guardat a figures/gmm_pca_k{best_k}.png")


# -------------------------------------------------------------------
# 8. Escriure resultats al fitxer experiments.md
# -------------------------------------------------------------------

EXPERIMENTS_PATH = "experiments.md"

entry = f"""
| ID  | Data       | Descripció                               | Model / Config                     | Resultat (silhouette mostra) | Comentaris                         |
|-----|------------|-------------------------------------------|------------------------------------|------------------------------|------------------------------------|
| G1  | baseline   | GMM baseline, k={best_k}                  | GMM (covariance_type='full')       | {best_sil:.4f}               | Millor k segons silhouette (mostra) |
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