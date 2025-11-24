import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "vscode" #no enseña el plot pero si las connexiones

# 1. Cargar datos
data = pd.read_csv("Spotify-2000.csv")
print(data.head())

# 2. Eliminar columna innecesaria
data = data.drop("Index", axis=1)

# 3. Matriz de correlación
numeric_data = data._get_numeric_data()
print(numeric_data.corr())

# 4. Selección de variables numéricas para K-Means
data2 = data[["Beats Per Minute (BPM)", "Loudness (dB)", 
              "Liveness", "Valence", "Acousticness", 
              "Speechiness"]]

# 5. Escalar correctamente los datos
scaler = MinMaxScaler()
data2_scaled = scaler.fit_transform(data2)
    
# 6. K-Means (10 clusters)
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(data2_scaled)

# 7. Añadir clusters al dataframe
data["Music Segments"] = clusters

# Mapeo corregido: KMeans genera clusters 0–9, no 1–10
data["Music Segments"] = data["Music Segments"].map({0: "Cluster 1", 
    1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5",
    5: "Cluster 6", 6: "Cluster 7", 7: "Cluster 8", 8: "Cluster 9",
    9: "Cluster 10"})
print(data.head())

# 8. Gráfico 3D
PLOT = go.Figure()

for cluster_name in data["Music Segments"].unique():
    cluster_data = data[data["Music Segments"] == cluster_name]

    PLOT.add_trace(go.Scatter3d(
        x = cluster_data['Beats Per Minute (BPM)'],
        y = cluster_data['Energy'],
        z = cluster_data['Danceability'],                        
        mode = 'markers',
        marker_size = 6,
        marker_line_width = 1,
        name = cluster_name
    ))

PLOT.update_traces(hovertemplate= 'Beats Per Minute (BPM): %{x} <br>Energy: %{y} <br>Danceability: %{z}')

PLOT.update_layout(
    width = 800,
    height = 800,
    autosize = True,
    showlegend = True,
    scene = dict(
        xaxis=dict(title='Beats Per Minute (BPM)'),
        yaxis=dict(title='Energy'),
        zaxis=dict(title='Danceability')
    ),
    font=dict(family="Gilroy", color='black', size=12)
)

PLOT.show()
