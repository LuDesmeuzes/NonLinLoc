# Pipeline NLL — préparation des fichiers pour créer les TTT

Ce dépôt contient deux scripts qui transforment le modèle de vitesse 3D de Jean Virieux et Nouibat Ahmed (VP/VS en fonction de la longitude, latitude et
profondeur) en fichiers prêts à l'emploi pour **NonLinLoc (NLL)**, une méthode de localisation 3D non linéaire de séisme. Plus exactement, les scripts permettent de préparer les fichiers nécessaires à la création des TTT avec Grid2Time (travel time tables, grilles de temps de trajet en fr), indispensable au fonctionnement de **NonLinLoc**. 

## Workflow

```
<model_file>.asc                          config.yaml
      (modèle global VP/VS)          (segment.zones : découpage en tuiles)
                    \                      /
                     v                    v
              ┌───────────────────────────────┐
              │     segment_from_zones.py     │
              └───────────────────────────────┘
                              │
                              v
                     <tuiles_dir>/<tuile>/
                       <tuile>_standard.csv
                       <tuile>_elevated.csv
                       README.txt
                              │
                              v
              ┌───────────────────────────────┐
              │        nll_pipeline.py        │
              └───────────────────────────────┘
                              │
                              v
                     <tuiles_dir>/<tuile>/model/
                       layer.P.mod.hdr / .buf
                       layer.S.mod.hdr / .buf
                       topo.asc
```

`<model_file>` et `<tuiles_dir>` sont les chemins que vous renseignez
dans `config.yaml` (voir Configuration) — aucun chemin n'est imposé par
le code.

1. **`segment_from_zones.py`** — découpe le modèle global en tuiles
   régionales (une par zone définie dans `config.yaml`, clé
   `segment.zones`), avec extension optionnelle en altitude pour
   couvrir la topographie.
2. **`nll_pipeline.py`** — pour chaque tuile : projection cartographique
   Azimuthal Equidistant, interpolation sur grille régulière, extraction
   du plus grand rectangle complet, écriture des fichiers binaires NLL
   (`layer.P/S.mod.hdr/.buf`) et de la topographie (`topo.asc`).

Le code partagé entre les deux scripts (projection, lecture des
`README.txt`, config) vit dans `nll_common.py`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # seulement la première fois sauf si changement dans le requirements.txt
```

## Configuration

Tous les chemins et paramètres sont centralisés dans un fichier YAML —
aucun chemin n'est codé en dur dans le code.

```bash
cp config.example.yaml config.yaml
```

Puis éditer `config.yaml` pour renseigner :
- `paths.model_file` : le fichier du modèle de vitesse global
- `paths.tuiles_dir` : le dossier où lire/écrire les tuiles
- `segment.zones` : la liste des zones/tuiles à découper (une entrée par
  tuile : `numero`, `nom`, `lon_min`, `lon_max`, `lat_min`, `lat_max`, et
  optionnellement `depth_max_km`) — voir `config.example.yaml` pour un exemple

### Modèle de vitesse 3D de Jean Virieux et Nouibat Ahmed

Pour vous le procurer, vous pouvez écrire à ludovic.desmeuzes@univ-grenoble-alpes.fr ou à mickael.langlais@univ-grenoble-alpes.fr. Ensuite, placer le où vous le souhaitez sur votre machine, mais je vous conseille vivement de suivre les suggestions du modop pour un maximum de clarté, puis renseigner son chemin dans `paths.model_file` (voir ci-dessus).

`config.yaml` est spécifique à chaque machine et n'est **pas** versionné
(voir `.gitignore`) ; `config.example.yaml` documente chaque champ et sert
de modèle.

## Utilisation

```bash
# 1. Découpage du modèle global en tuiles
python segment_from_zones.py

# 2. Génération des fichiers NLL pour chaque tuile
python nll_pipeline.py
```

Un fichier de configuration alternatif peut être passé explicitement :

```bash
python nll_pipeline.py --config config_projet_x.yaml
```

## Tests

```bash
pytest tests/
```

Les tests couvrent la projection Azimuthal Equidistant, le parsing des
`README.txt`, le chargement de la configuration et l'algorithme
d'extraction du plus grand rectangle complet.

## Structure des fichiers

```
NonLinLoc_ToolBox/
├── README.md                 ← ce fichier
├── requirements.txt          ← dépendances Python
├── config.example.yaml       ← modèle de configuration (versionné)
├── config.yaml                ← configuration locale (non versionné)
├── nll_common.py              ← code partagé (projection, config, README parsing)
├── segment_from_zones.py      ← étape 1 : découpage en tuiles
├── nll_pipeline.py            ← étape 2 : génération des fichiers NLL
└── tests/                     ← tests unitaires (pytest)
```
