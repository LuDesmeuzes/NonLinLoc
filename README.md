# Pipeline NLL — préparation de modèles de vitesse sismique

Ce dépôt contient deux scripts qui transforment un modèle de vitesse
sismique 3D global (VP/VS en fonction de la longitude, latitude et
profondeur) en fichiers prêts à l'emploi pour **NonLinLoc (NLL)**, un
logiciel de localisation de séismes.

## Workflow

```
04_modele de vitesse/wet_new.asc          zones_modele.xlsx
      (modèle global VP/VS)          (découpage en zones/tuiles)
                    \                      /
                     v                    v
              ┌───────────────────────────────┐
              │     segment_from_zones.py      │
              └───────────────────────────────┘
                              │
                              v
                     01_tuiles/<tuile>/
                       <tuile>_standard.csv
                       <tuile>_elevated.csv
                       README.txt
                              │
                              v
              ┌───────────────────────────────┐
              │        nll_pipeline.py         │
              └───────────────────────────────┘
                              │
                              v
                     01_tuiles/<tuile>/nll/
                       layer.P.mod.hdr / .buf
                       layer.S.mod.hdr / .buf
                       topo.asc
```

1. **`segment_from_zones.py`** — découpe le modèle global en tuiles
   régionales (une par zone définie dans `zones_modele.xlsx`), avec
   extension optionnelle en altitude pour couvrir la topographie.
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
pip install -r requirements.txt
```

## Configuration

Tous les chemins et paramètres sont centralisés dans un fichier YAML —
aucun chemin n'est codé en dur dans le code.

```bash
cp config.example.yaml config.yaml
```

Puis éditer `config.yaml` pour renseigner :
- `paths.model_file` : le fichier du modèle de vitesse global
- `paths.zones_file` : le fichier de définition des zones (xlsx/csv/txt)
- `paths.tuiles_dir` : le dossier où lire/écrire les tuiles

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
02_scripts/
├── README.md                 ← ce fichier
├── requirements.txt          ← dépendances Python
├── config.example.yaml       ← modèle de configuration (versionné)
├── config.yaml                ← configuration locale (non versionné)
├── nll_common.py              ← code partagé (projection, config, README parsing)
├── segment_from_zones.py      ← étape 1 : découpage en tuiles
├── nll_pipeline.py            ← étape 2 : génération des fichiers NLL
├── zones_modele.xlsx          ← définition des zones/tuiles
└── tests/                     ← tests unitaires (pytest)
```
