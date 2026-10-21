## Documentation

Le dossier `docs/` contient la documentation du workflow NLL complet, au-delà de ce que fait ce dépôt :
- `Mise en place NLL sur Seiscomp.pdf` : tutoriel détaillant tout le pipeline NonLinLoc — concepts clés (`TRANS`, modèle 1D vs 3D, grilles `VGGRID`/`LOCGRID`), récupération du modèle de vitesse et des stations, génération des grilles de temps de trajet (1D et 3D), puis mise en place et utilisation de NLL sur SeisComP/OAR.
- `vel2grid_grid2time.in.example` : fichier de contrôle NLL commenté pour lancer `Vel2Grid` puis `Grid2Time` sur un modèle de vitesse **1D** (cas non couvert par les scripts de ce dépôt, qui ne traitent que le modèle 3D — voir le PDF pour le détail des paramètres).

# Pipeline NLL — préparation des fichiers pour créer les TTT

Le reste de ce dépôt (en dehors de docs/) contient deux scripts qui transforment le modèle de vitesse 3D de Jean Virieux et Nouibat Ahmed (VP/VS en fonction de la longitude, latitude et profondeur) en fichiers prêts à l'emploi pour **NonLinLoc (NLL)**, une méthode de localisation 3D non linéaire de séisme. Plus exactement, les scripts permettent de préparer les fichiers nécessaires à la création des TTT avec Grid2Time (travel time tables, grilles de temps de trajet en fr), indispensable au fonctionnement de **NonLinLoc**. 

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

### Lancer un profil NLL en local sur une période

`nll_run.py` relocalise avec NLL en ligne de commande les événements d'une
période, avec la configuration d'un profil SeisComP, sans passer par le
serveur :

```bash
python nll_run.py --profil 1D_ubaye --debut 2024-11-01 --fin 2024-11-30
python nll_run.py --profil 1D_ubaye --debut 2024-11-01 --fin 2024-11-30 \
    --nom 1D_ubaye_oct15 --locsearch "OCT 15 15 15 0.001 50000 5000 0 1"
```

1. Les pointés de la période sont récupérés par `fetch_picks.py` (ci-dessous).
2. Le fichier de contrôle reprend `<run.profils_dir>/<profil>/NLL.<profil>.conf`,
   copie de la configuration du profil sur la machine SeisComP (`TRANS`,
   `LOCGRID`, `LOCSEARCH`, `LOCMETH`, `LOCTOPO_SURFACE`…), avec les grilles
   de temps de `<profil>/time/`. `${NLLROOT}` est remplacé par
   `run.profils_dir`. `--locsearch` remplace `LOCSEARCH` pour un test.
3. `NLLoc` tourne en parallèle sur `--jobs` morceaux de la période (par
   défaut, nombre de cœurs − 1) : environ 2 minutes pour 3000 événements.
4. Le run est écrit dans `<run.runs_dir>/<nom>/` : `<nom>.hyp` (toutes les
   localisations), `evenements.csv` (origine de référence) et `run.yaml`
   (profil, période, paramètres, comptes). LocQuality_ToolBox lit ce dossier
   directement (`paths.nll_runs`).

Les stations des pointés sans grille de temps dans le profil sont signalées
(NLLoc ignore leurs phases). Pour qu'un run local reproduise SeisComP, la
copie locale `NLL.<profil>.conf` doit être à jour avec celle du serveur.

**`--grilles`** calcule avant la localisation les grilles de temps
manquantes (Grid2Time), pour les seules stations pointées sur la période.
Les lignes `TRANS`, `GTFILES` (modèle et dossier des grilles), `GTMODE` et
`GT_PLFD` sont lues dans `<profil>/nll_<profil>.in`. Les positions des
stations viennent de l'inventaire SeisComP (`fetch.fdsnws_station_url`),
valides sur la période. En 3D, les stations hors de l'emprise du modèle
sont ignorées. Compter environ 2 s par station et par phase pour une grille
3D au pas de 0,5 km sur la tuile Ubaye, et 87 Mo par grille (temps + angles).

#### Créer un nouveau profil

Exemple : `profils/05_ubaye_3D_topo/`, modèle 3D au pas de 0,5 km avec
topographie.

1. `model/layer.P.mod.*` et `layer.S.mod.*` : le modèle de vitesse au format
   NLL (sortie de `nll_pipeline.py`).
2. `topo.asc` (si topographie) : sortie de `nll_pipeline.py`, en-tête au
   format GMT 4 (`topo_header()`).
3. `nll_<profil>.in` : `TRANS`, `GTFILES <modèle> <grilles> P`, `GTMODE`,
   `GT_PLFD` (pas de `GTSRCE` : `--grilles` les prend dans l'inventaire).
   `${NLLROOT}` = `run.profils_dir`. Deux profils peuvent partager les mêmes
   grilles : `GTFILES` pointe alors vers le même dossier.
4. `NLL.<profil>.conf` : `TRANS` (le même que pour les grilles), `LOCGRID`,
   `LOCSEARCH`, `LOCMETH`… et `LOCTOPO_SURFACE ${NLLROOT}/<profil>/topo.asc 0`.
   La grille de recherche n'a pas besoin du pas des grilles de temps :
   l'octree subdivise lui-même. Choisir des cellules initiales d'environ
   3 km de côté dans chaque direction (ex. `LOCGRID 101 101 36 -50 -50 -5 1 1 1`
   et `OCT 30 30 11` : 100/30 et 35/11 km).
5. `python nll_run.py --profil <profil> --debut … --fin … --grilles`.

Sur novembre 2024 avec le profil `1D_ubaye`, 93 % des 3022 événements
communs avec le run SeisComP `1D_ubaye_v2` ont exactement la même position.
Les écarts concernent surtout les événements pointés sur des stations à plus
de 50 km. Les statistiques d'ensemble (RMS, incertitudes, écart au LOCSAT,
classes de qualité) sont identiques à 1 % près.

### Pointés SeisComP pour NLL en ligne de commande

`fetch_picks.py` récupère sur le serveur SeisComP (service web
fdsnws-event, sans connexion à la machine) les événements d'une période et
de `fetch.zone`, avec leur origine préférée (le manuel), ses arrivées et ses
pointés, et les convertit au format `NLLOC_OBS` :

```bash
python fetch_picks.py --debut 2024-11-01 --fin 2024-11-30
```

Une requête par jour ; chaque journée est gardée dans `fetch.cache_dir` et
n'est plus redemandée ensuite (`--refresh` pour retélécharger, par exemple
après une révision des pointés). Le service ne renvoie que l'origine
préférée de chaque événement : les origines NLL du serveur ne sont pas
récupérables par ce biais.

`seiscomp_xml_to_obs.py` fait la même conversion sur des dossiers de XML
déjà téléchargés (`phases.xml_dirs`, un ou plusieurs événements par
fichier) :

```bash
python seiscomp_xml_to_obs.py
```

Seules les arrivées utilisées (`timeUsed`) sont gardées. Les pointés sans
incertitude (presque tous : pointés automatiques et manuels sans
incertitude) reçoivent `phases.default_error_s`, la valeur de
`NonLinLoc.defaultPickError` de SeisComP (0,02 s). Chaque événement est
précédé de `PUBLIC_ID <Event ID>`, que NLLoc recopie dans ses `.hyp`. Les
XML vides ou tronqués sont ignorés et listés à part.

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
d'extraction du plus grand rectangle complet, l'en-tête de `topo.asc`, la
conversion des XML SeisComP en pointés NLL (un ou plusieurs événements par
fichier) et la préparation d'un run local (fichier de contrôle, découpage
des pointés, stations sans grille).

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
├── seiscomp_xml_to_obs.py     ← pointés SeisComP (XML) → NLLOC_OBS
├── fetch_picks.py             ← pointés d'une période depuis le serveur (fdsnws)
├── nll_run.py                 ← run NLL local : profil × période
├── tests/                     ← tests unitaires (pytest)
└── docs/                      ← tutoriel NLL/SeisComP + exemples de fichiers .in
```
