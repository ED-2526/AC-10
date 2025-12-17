import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregar les dades netejades
df_clean = pd.read_csv("dataset-millorat/data/processat/05_final_clean_dataset.csv")

# Filtrar només les columnes numèriques per calcular la correlació
df_numeric = df_clean.select_dtypes(include=[np.number])

# Calcular la correlació entre les característiques numèriques
correlation_matrix = df_numeric.corr()

# Visualitzar la matriu de correlació amb un heatmap
plt.figure(figsize=(15, 10))  # Ajustem la mida de la figura per a millor visibilitat
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
plt.title("Matriz de Correlación de Características", fontsize=16)

# Guardar la figura amb alta resolució
plt.savefig("figures/correlation_matrix_high_res.png", dpi=300)

# Mostrar la figura
plt.show()
