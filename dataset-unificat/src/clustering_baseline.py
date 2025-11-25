"""
clustering_baseline.py

Script per generar el baseline de clustering:
- Carrega i processa dades amb data_prep.py
- Prova valors de k (2..12) amb K-means
- Calcula elbow (SSE)
- Calcula silhouette score
- PCA 2D i guarda figura
- Escriu resultats al fitxer experiments.md

Autor: Nino
Projecte: Music Listener Clustering
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Importo el meu data_prep.py
from data_prep import prepare_dataset

# -------------------------------------------------------------------
# 1. Carregar dataset processat
# -------------------------------------------------------------------

print("Carregant dades...")

# prepare_dataset() fa: carrega CSV -> neteja - > selecciona info/features - > escala
info_df, X_scaled, scaler = prepare_dataset()

# Mostro la forma de la matriu X_scaled
# Files = canços, Columnes = 11 features musicals
print("Forma de X_scaled:", X_scaled.shape)


# -------------------------------------------------------------------
# 2. Provar K-means amb diferents valors de k (elbow + silhouette)
# -------------------------------------------------------------------

# Provem k de 2 a 12
K_RANGE = range(2, 13)      # k = 2..12

# Aquí guardem els resultats
sse_list = []   # Sum of Squared Errors per a elbow
silhouette_list = [] # Silhouette scores (qualitat dels clustering)

print("\nCalculant elbow i silhouette per diferents k...\n")

# Bucle principal per provar diferents valors de k
for k in K_RANGE:
    # Inicialitzem K-Means amb k clústers
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    
    # Entrenem el model i obtenim les etiquetes (quin cluster rep cada cançó)
    labels = km.fit_predict(X_scaled)

    # Cqlculem SSE = suma de distàncies fins al seu centre
    sse = km.inertia_
    sse_list.append(sse)

    # Calcume Silhouette Score = mesura entre -1 i 1 (com de separats estan els clusters)
    sil = silhouette_score(X_scaled, labels)
    silhouette_list.append(sil)

    # Mostrem els resultats per aquest k
    print(f"k={k}  SSE={sse:.2f}  Silhouette={sil:.4f}")


# -------------------------------------------------------------------
# 3. Generar gràfic Elbow
# -------------------------------------------------------------------

# Gràfic Elbow: k vs SSE
plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), sse_list, marker="o")
plt.xlabel("Nombre de clústers (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method - KMeans")
plt.grid(True)

# Creem directori figures si no existeix
os.makedirs("figures", exist_ok=True)

# Guardem la figura
plt.savefig("figures/elbow_kmeans.png", dpi=150)
plt.close()

print("\nGràfic Elbow guardat a figures/elbow_kmeans.png")


# -------------------------------------------------------------------
# 4. Gràfic Silhouette
# -------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), silhouette_list, marker="o", color="green")
plt.xlabel("Nombre de clústers (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score per diferents k")
plt.grid(True)
plt.savefig("figures/silhouette_kmeans.png", dpi=150)
plt.close()

print("Gràfic Silhouette guardat a figures/silhouette_kmeans.png")


# -------------------------------------------------------------------
# 5. Triar el millor k segons silhouette (baseline)
# -------------------------------------------------------------------

best_k = K_RANGE[np.argmax(silhouette_list)]
print(f"\nMillor k segons silhouette → {best_k}")

best_km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
best_labels = best_km.fit_predict(X_scaled)


# -------------------------------------------------------------------
# 6. PCA 2D de visualització
# -------------------------------------------------------------------

print("\nCalculant PCA 2D...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=best_labels, palette="tab10", s=10, alpha=0.7
)
plt.title(f"PCA 2D - KMeans (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("figures/pca_kmeans.png", dpi=150)
plt.close()

print("PCA 2D guardat a figures/pca_kmeans.png")


# -------------------------------------------------------------------
# 7. Escriure resultats al fitxer experiments.md
# -------------------------------------------------------------------

EXPERIMENTS_PATH = "experiments.md"

entry = f"""
| ID | Data       | Descripció                          | Model / Config       | Resultat (silhouette) | Comentaris |
|----|------------|--------------------------------------|-----------------------|------------------------|-------------|
| B1 | baseline   | KMeans baseline, k={best_k}          | k-means (auto)        | {max(silhouette_list):.4f} | Millor k segons silhouette |
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

print("\nResultats afegits a experiments.md")


# -------------------------------------------------------------------
print("\n✔️ Baseline complet! Gràfics generats i experiments registrats.\n")
