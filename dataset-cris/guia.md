
---

## Pla de treball del projecte
En aquest projecte de *Music Listener Clustering* intentarem seguir aproximadament els passos següents.

---
Així es l'estructura inicial del repositori:

```
dataset-cris/
├── data/
│   ├── raw/
│   │   └── spotify.csv          # CSV baixat de Kaggle
│   └── procesada/               # CSV netejat (pendent)
├── src/
│   └── data_prep.py             # Codi per carregar i preparar/netejar les dades (pendent)
├── guia.md                      # Pla de treball, descripció del projecte
└── requirements.txt             # Llista de paquets Python (si escau, els indispensables)
```

---

### 1. Obtenir i preparar les dades

1. Descargar el dataset de Kaggle (Ultimate Spotify Tracks DB) i guardar-lo temporalment a una carpeta del projecte  fins que confirmem quin dataset farem servir (`datdataset-cris/SpotifyFeatures.csv`).

2. Fer una exploració inicial:

   * Nombre de files i columnes.
   * Tipus de cada variable (numèrica, categòrica, identificadors, etc.).
   * Percentatge de valors nuls per columna.
3. Preparar les dades per al model:

   * Seleccionar les columnes que faran de *features* (danceability, energy, valence, acousticness, etc.).
   * Tractar valors nuls (eliminar o imputar).
   * Escalar / normalitzar les variables numèriques.

Si faig servir PyTorch, crearé també (encara esta per decidir):

* Una classe `SpotifyDataset` que llegeixi el CSV i retorni els vectors de característiques.
* Un `DataLoader` que em proporcioni batches de dades per entrenar i avaluar els models.

---

### 2. Entendre el punt de partida (baseline)

Si es proporciona codi o material de referència:

1. Llegir el codi de partida (notebooks, scripts) i la documentació associada (articles, blogs, etc.).
2. Identificar:

   * Quines variables d’entrada utilitza el baseline.
   * Quin preprocés de dades aplica (drop de columnes, escalat, PCA, etc.).
   * Quin algoritme de clustering o model utilitza (k-means, hierarchical, GMM, …).
3. Ser capaç de descriure el baseline en detall:

   * Entrades i sortides del model.
   * Hiperparàmetres més importants (nombre de clústers, etc.).
   * Mètriques d’avaluació utilitzades (silhouette score, inertia, etc.).
4. Reproduir i entrenar el baseline:

   * Executar el codi tal com està.
   * Comprovar que funciona en el meu entorn.
   * Guardar i documentar els resultats obtinguts (mètriques, nombres de clústers, interpretacions bàsiques).

---

### 3. Millorar el baseline

A partir del baseline, provaré de millorar el model o l’estratègia:
1. Canvis de model / estratègia:

   * Provar diferents algoritmes de clustering (k-means amb diferents valors de *k*, clustering jeràrquic, Gaussian Mixture Models, etc.).
   * Provar PCA + clustering i comparar-ho amb clustering directament en l’espai original.
   * Modificar el conjunt de features (afegir o treure variables) i veure com afecta els resultats.
2. Entrenar amb més dades (quan sigui possible):

   * Utilitzar una part més gran del dataset o el dataset complet.
   * Comprovar si, amb més dades, els clústers són més estables i interpretables.
3. Comparar models:

   * Fer servir mètriques quantitatives (silhouette, inertia, distribució de la mida dels clústers).
   * Valorar quina configuració ofereix un millor compromís entre qualitat de clustering i interpretabilitat.

---

### 4. Crear un baseline propi (si no n’hi ha cap)

Si no es proporciona model de partida, definiré un baseline senzill:
1. Definir un pipeline inicial:

   * Seleccionar un conjunt de features numèriques rellevants.
   * Aplicar escalat (per exemple, `StandardScaler`).
   * Entrenar un k-means amb alguns valors inicials de *k* (p.ex. 4, 6, 8).
   * Avaluar amb silhouette score i mètode del colze (*elbow*).
2. Verificar les meves assumpcions:

   * Comprovar que les distribucions de les variables tenen sentit.
   * Mirar cançons representatives de cada clúster.
   * Ajustar features i hiperparàmetres si els clústers no són interpretables.

Aquest baseline propi serà el punt de referència per a les millores posteriors.

---

### 5. Entendre el model

L’objectiu no és només obtenir clústers, sinó entendre què està fent el model i si té sentit musicalment.

#### 5.1. Tests en dades diferents (out-of-distribution / dades pròpies)

* Provar el model en subconjunts de dades que no he mirat abans (per exemple, cançons d’un artista concret, d’un gènere minoritari o dades externes si en tinc).
* Observar a quins clústers acaben aquestes cançons i si la classificació sembla coherent.

#### 5.2. Anàlisi quantitativa

* Calcular i comparar, per a cada configuració de model:

  * Silhouette score.
  * Inertia / SSE (en el cas de k-means).
  * Distribució del nombre de cançons per clúster.
* Utilitzar aquestes mètriques per justificar l’elecció del nombre de clústers i del model final.

#### 5.3. Anàlisi qualitativa

* Analitzar qualitativament cada clúster:

  * Valors mitjans de les principals features (danceability, energy, valence, etc.).
  * Artistes i gèneres més freqüents.
  * Alguns exemples de cançons conegudes.
* Descriure cada clúster en paraules (“clúster de temes dance/party”, “clúster de temes chill/acoustic”, etc.) per donar-li sentit musical.

#### 5.4. Visualització de les decisions

* Fer servir visualitzacions per entendre i comunicar els resultats:

  * PCA 2D amb punts colorejats per clúster.
  * Boxplots o barplots de features per clúster.
  * (Opcional) Gràfics de radar amb les mitjanes de features per clúster.
* Utilitzar aquests gràfics a la memòria per reforçar la interpretació dels clústers i del comportament del model.

## Registre d’experiments

Podem trobar el detall dels experiments a: [`docs/experiments.md`](docs/experiments.md).