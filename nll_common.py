"""
nll_common.py
=============
Fonctions et classes partagées entre segment_from_zones.py et nll_pipeline.py :
  - lecture de la configuration YAML
  - projection cartographique Azimuthal Equidistant
  - lecture des README.txt générés par segment_from_zones.py
  - petit helper de log indenté

Ce module n'a pas vocation à être exécuté directement.
"""

import re
from pathlib import Path

import numpy as np
import yaml

EARTH_RADIUS_KM = 6371.0
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path):
    """
    Charge un fichier de configuration YAML.

    Si le fichier n'existe pas, lève une erreur explicite invitant
    à copier config.example.yaml → config.yaml et à l'adapter.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        example = config_path.parent / "config.example.yaml"
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_path}\n"
            f"→ Copier {example.name} vers {config_path.name} puis adapter "
            f"les chemins à votre machine."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Fichier de configuration vide ou invalide : {config_path}")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# LOG
# ─────────────────────────────────────────────────────────────────────────────

def log(msg, indent=0):
    print("  " * indent + msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION AZIMUTHAL EQUIDISTANT
# ─────────────────────────────────────────────────────────────────────────────

class AzimuthalEquidistant:
    def __init__(self, lon0, lat0, radius_km=EARTH_RADIUS_KM):
        self.lon0  = lon0
        self.lat0  = lat0
        self.R     = radius_km
        self._sinp = np.sin(lat0 * D2R)
        self._cosp = np.cos(lat0 * D2R)

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        dlon   = (lon - self.lon0 + 180.0) % 360.0 - 180.0
        dlon_r = dlon * D2R
        lat_r  = lat  * D2R
        slat = np.sin(lat_r); clat = np.cos(lat_r); clon = np.cos(dlon_r)
        cc = np.clip(self._sinp * slat + self._cosp * clat * clon, -1.0, 1.0)
        at_center = np.abs(cc) >= 1.0
        c     = np.where(at_center, 0.0, np.arccos(cc))
        sin_c = np.sin(c)
        k = np.where(at_center, self.R,
                     self.R * c / np.where(sin_c == 0.0, 1.0, sin_c))
        x = k * clat * np.sin(dlon_r)
        y = k * (self._cosp * slat - self._sinp * clat * clon)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# LECTURE DES README.txt (générés par segment_from_zones.py)
# ─────────────────────────────────────────────────────────────────────────────

def read_center_from_readme(readme_path):
    """
    Extrait lon0 et lat0 depuis le README.txt généré par segment_from_zones.py.
    Cherche les lignes :
        lon0 = 6.750000°
        lat0 = 44.500000°
    Retourne (lon0, lat0) ou lève ValueError si introuvable.
    """
    text = Path(readme_path).read_text(encoding="utf-8")
    lon_match = re.search(r"lon0\s*=\s*([\-\d.]+)", text)
    lat_match = re.search(r"lat0\s*=\s*([\-\d.]+)", text)
    if not lon_match or not lat_match:
        raise ValueError(
            f"Impossible de lire lon0/lat0 dans {readme_path}\n"
            f"Vérifier que le README a bien été généré par segment_from_zones.py"
        )
    return float(lon_match.group(1)), float(lat_match.group(1))


def read_bounds_from_readme(readme_path):
    """
    Extrait lon_min, lon_max, lat_min, lat_max depuis le README.txt.
    Cherche les lignes :
        Longitude : [6.0000°, 7.5000°]
        Latitude  : [44.0000°, 45.0000°]
    Retourne (lon_min, lon_max, lat_min, lat_max).
    """
    text = Path(readme_path).read_text(encoding="utf-8")
    lon_match = re.search(r"Longitude\s*:\s*\[([\-\d.]+)°,\s*([\-\d.]+)°\]", text)
    lat_match = re.search(r"Latitude\s*:\s*\[([\-\d.]+)°,\s*([\-\d.]+)°\]", text)
    if not lon_match or not lat_match:
        raise ValueError(
            f"Impossible de lire les bornes géographiques dans {readme_path}"
        )
    return (float(lon_match.group(1)), float(lon_match.group(2)),
            float(lat_match.group(1)), float(lat_match.group(2)))


def find_input_csv(tuile_dir, use_elevated):
    """
    Trouve le fichier _elevated.csv ou _standard.csv dans le dossier tuile.
    Retourne le Path du fichier trouvé.
    """
    suffix = "_elevated.csv" if use_elevated else "_standard.csv"
    candidates = list(Path(tuile_dir).glob(f"*{suffix}"))
    if not candidates:
        alt_suffix = "_standard.csv" if use_elevated else "_elevated.csv"
        candidates = list(Path(tuile_dir).glob(f"*{alt_suffix}"))
        if candidates:
            log(f"⚠ Fichier {suffix} absent, utilisation de {alt_suffix}", 2)
        else:
            raise FileNotFoundError(
                f"Aucun fichier CSV de modèle trouvé dans {tuile_dir}"
            )
    return candidates[0]
