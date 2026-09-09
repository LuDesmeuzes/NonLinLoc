"""
segment_from_zones.py
=====================
Découpe un modèle de vitesse 3D global (LON, LAT, DEPTH, VP, VS)
en tuiles dont les dimensions sont lues dans un fichier externe
(Excel .xlsx, CSV ou TXT).

Pour chaque tuile, le script crée un sous-dossier dédié et y place :
  - <nom>_standard.csv   : profondeur  0 → 100 km  (m → km converti)
  - <nom>_elevated.csv   : profondeur -5 → 100 km  (couches d'altitude ajoutées)
  - README.txt           : résumé de la tuile (coordonnées, centre AE, stats)

Architecture de sortie générée automatiquement :
  tuiles_dir/
  ├── Z01_Ubaye/
  │   ├── Z01_Ubaye_standard.csv
  │   ├── Z01_Ubaye_elevated.csv
  │   └── README.txt
  ├── Z02_Belledonne/
  │   └── ...
  └── recap_global.csv      ← tableau récapitulatif de toutes les tuiles

Format du fichier de zones (Excel ou CSV) :
  Colonnes obligatoires : numero | nom | lon_min | lon_max | lat_min | lat_max
  Colonne optionnelle  : depth_max_km  (profondeur max en km, ex: 40)
                         Si absente ou vide → pas de limite de profondeur.
  Le centre de projection AE est calculé automatiquement
  comme le centre géographique de chaque tuile.

Configuration :
  Tous les chemins et paramètres (auparavant codés en dur) sont lus depuis
  un fichier YAML — voir config.example.yaml pour le détail des champs.

Utilisation :
    cp config.example.yaml config.yaml   # une seule fois, puis adapter
    python segment_from_zones.py
    python segment_from_zones.py --config un_autre_config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nll_common import load_config, log


# ── 1. Chargement du fichier de zones ────────────────────────────────────────

def load_zones(filepath):
    """
    Charge le fichier de définition des zones.
    Accepte : .xlsx / .xls  (Excel)
              .csv           (virgule ou point-virgule)
              .txt           (tabulation ou espaces)

    Colonnes attendues (insensible à la casse) :
        numero | nom | lon_min | lon_max | lat_min | lat_max | depth_max_km

    Retourne une liste de dicts.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    print(f"Chargement des zones depuis {filepath} …")

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, dtype=str)
    elif ext == ".csv":
        # Détection automatique du séparateur
        raw = filepath.read_text(encoding="utf-8-sig")
        sep = ";" if raw.count(";") > raw.count(",") else ","
        df  = pd.read_csv(filepath, sep=sep, dtype=str)
    else:
        # .txt ou autre → tabulation puis espaces
        try:
            df = pd.read_csv(filepath, sep="\t", dtype=str)
        except Exception:
            df = pd.read_csv(filepath, sep=r"\s+", dtype=str, engine="python")

    # Normalisation des noms de colonnes
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = ["numero", "nom", "lon_min", "lon_max", "lat_min", "lat_max"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier de zones : {missing}\n"
            f"Colonnes trouvées : {list(df.columns)}\n"
            f"Colonnes attendues : {required}"
        )

    zones = []
    for _, row in df.iterrows():
        try:
            z = {
                "numero"  : str(row["numero"]).strip(),
                "nom"     : str(row["nom"]).strip(),
                "lon_min" : float(row["lon_min"]),
                "lon_max" : float(row["lon_max"]),
                "lat_min" : float(row["lat_min"]),
                "lat_max" : float(row["lat_max"]),
            }
            # Profondeur maximale — optionnelle
            if "depth_max_km" in df.columns:
                val = str(row["depth_max_km"]).strip()
                z["depth_max_km"] = float(val) if val not in ("", "nan", "None") else None
            else:
                z["depth_max_km"] = None
            # Centre géographique = centre de projection AE
            z["lon0"] = round((z["lon_min"] + z["lon_max"]) / 2, 6)
            z["lat0"] = round((z["lat_min"] + z["lat_max"]) / 2, 6)
            zones.append(z)
        except (ValueError, KeyError) as e:
            print(f"  ⚠ Ligne ignorée ({row.to_dict()}) : {e}")

    print(f"  {len(zones)} zones chargées")
    for z in zones:
        depth_str = f"  prof max {z['depth_max_km']:.0f} km" if z["depth_max_km"] else ""
        print(f"    {z['numero']:>5}  {z['nom']:<20}  "
              f"lon [{z['lon_min']:.2f}, {z['lon_max']:.2f}]  "
              f"lat [{z['lat_min']:.2f}, {z['lat_max']:.2f}]  "
              f"centre ({z['lon0']:.3f}, {z['lat0']:.3f}){depth_str}")
    return zones


# ── 2. Chargement du modèle global ───────────────────────────────────────────

def load_global_model(filepath, sep, has_header, depth_in_meters):
    """
    Charge le fichier .asc du modèle global.
    Colonnes : LON  LAT  DEPTH  VP  VS
    Convertit la profondeur m → km si depth_in_meters=True.
    """
    print(f"\nChargement du modèle global {filepath} …")
    if has_header:
        df = pd.read_csv(filepath, sep=sep, engine="python")
        df.columns = [c.strip().lower() for c in df.columns]
        rename = {}
        for col in df.columns:
            cl = col.lower()
            if "lon" in cl:                        rename[col] = "lon"
            elif "lat" in cl:                      rename[col] = "lat"
            elif "dep" in cl:                      rename[col] = "depth"
            elif cl in ("vp", "vitesse onde p", "p"): rename[col] = "Vp"
            elif cl in ("vs", "vitesse onde s", "s"): rename[col] = "Vs"
        df.rename(columns=rename, inplace=True)
    else:
        df = pd.read_csv(filepath, sep=sep, engine="python",
                         header=None, names=["lon", "lat", "depth", "Vp", "Vs"])

    required = ["lon", "lat", "depth", "Vp", "Vs"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le modèle : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    df = df[required].astype(float)

    if depth_in_meters:
        df["depth"] = df["depth"] / 1000.0
        print(f"  Conversion profondeur : m → km")

    print(f"  {len(df):,} points  |  "
          f"lon [{df.lon.min():.2f}, {df.lon.max():.2f}]  "
          f"lat [{df.lat.min():.2f}, {df.lat.max():.2f}]  "
          f"prof [{df.depth.min():.0f}, {df.depth.max():.0f}] km")
    return df


# ── 3. Extension en altitude ──────────────────────────────────────────────────

def add_elevation_layers(df_tile, elevation_km, step_km):
    """
    Duplique la couche de surface (depth ≈ 0 km) pour les altitudes
    -step_km, -2*step_km, …, -elevation_km.
    Retourne le DataFrame trié par (lon, lat, depth).
    """
    surface = df_tile[df_tile.depth.abs() < 0.1].copy()
    if len(surface) == 0:
        print("    ⚠ Aucune couche à 0 km — extension impossible")
        return df_tile

    new_layers = []
    for z in np.arange(step_km, elevation_km + step_km * 0.5, step_km):
        layer = surface.copy()
        layer["depth"] = -round(float(z), 6)
        new_layers.append(layer)

    df_out = pd.concat([pd.concat(new_layers), df_tile], ignore_index=True)
    df_out.sort_values(["lon", "lat", "depth"], inplace=True)
    return df_out.reset_index(drop=True)


# ── 4. Écriture d'un README par tuile ────────────────────────────────────────

def write_readme(folder, zone, n_std, n_elev, depth_range, elevation_km):
    """Écrit un fichier README.txt dans le dossier de la tuile."""
    depth_max_str = (f"{zone['depth_max_km']:.0f} km"
                     if zone["depth_max_km"] else "non limitée")
    txt = f"""Tuile : {zone['numero']} — {zone['nom']}
{"="*50}
Étendue géographique
  Longitude : [{zone['lon_min']:.4f}°, {zone['lon_max']:.4f}°]
  Latitude  : [{zone['lat_min']:.4f}°, {zone['lat_max']:.4f}°]

Centre de projection (Azimuthal Equidistant)
  lon0 = {zone['lon0']:.6f}°
  lat0 = {zone['lat0']:.6f}°

Profondeur maximale : {depth_max_str}

Contenu
  standard.csv : {n_std:,} points  (profondeur {depth_range[0]:.0f} → {depth_range[1]:.0f} km)
  elevated.csv : {n_elev:,} points (profondeur -{elevation_km} → {depth_range[1]:.0f} km)

Workflow suivant
  python nll_pipeline.py
"""
    (folder / "README.txt").write_text(txt, encoding="utf-8")


# ── 5. Traitement de toutes les tuiles ───────────────────────────────────────

def process_all_zones(df_global, zones, root_dir, elevation_km, elev_step_km, tol):
    """
    Pour chaque zone :
      - Crée  tuiles_dir/<numero>_<nom>/
      - Extrait les points du modèle global
      - Écrit _standard.csv, _elevated.csv, README.txt
    Retourne la liste des résumés pour le récap global.
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for i, zone in enumerate(zones, 1):
        tag      = f"{zone['numero']}_{zone['nom']}"
        folder   = root_dir / tag
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*55}")
        print(f"[{i}/{len(zones)}]  {tag}")
        print(f"  lon [{zone['lon_min']:.3f}, {zone['lon_max']:.3f}]  "
              f"lat [{zone['lat_min']:.3f}, {zone['lat_max']:.3f}]")

        # ── Extraction géographique ───────────────────────────────────────
        mask = (
            (df_global.lon   >= zone["lon_min"] - tol) &
            (df_global.lon   <= zone["lon_max"] + tol) &
            (df_global.lat   >= zone["lat_min"] - tol) &
            (df_global.lat   <= zone["lat_max"] + tol)
        )
        df_tile = df_global[mask].copy().reset_index(drop=True)

        # ── Limitation de profondeur (optionnelle) ────────────────────────
        if zone["depth_max_km"] is not None:
            before = len(df_tile)
            df_tile = df_tile[df_tile.depth <= zone["depth_max_km"] + 0.001].copy()
            df_tile.reset_index(drop=True, inplace=True)
            print(f"  Profondeur limitée à {zone['depth_max_km']:.0f} km  "
                  f"({before - len(df_tile):,} points supprimés)")

        n_pts = len(df_tile)

        if n_pts == 0:
            print("  ⚠ Aucun point trouvé — tuile ignorée")
            (folder / "VIDE.txt").write_text(
                "Aucun point du modèle global ne correspond à cette zone.\n"
                "Vérifier les coordonnées dans le fichier de zones.\n"
            )
            continue

        depth_range = (df_tile.depth.min(), df_tile.depth.max())
        print(f"  {n_pts:,} points extraits  "
              f"(prof {depth_range[0]:.0f}–{depth_range[1]:.0f} km)")

        COL_NAMES = ["Longitude", "Latitude", "Depth", "Vitesse onde P", "Vitesse onde S"]

        # ── Fichier standard ──────────────────────────────────────────────
        f_std = folder / f"{tag}_standard.csv"
        df_tile.to_csv(f_std, index=False, header=COL_NAMES)
        print(f"  ✓ {f_std.name}")

        # ── Fichier étendu ────────────────────────────────────────────────
        df_elev = add_elevation_layers(df_tile, elevation_km, elev_step_km)
        n_elev  = len(df_elev)
        f_elev  = folder / f"{tag}_elevated.csv"
        df_elev.to_csv(f_elev, index=False, header=COL_NAMES)
        print(f"  ✓ {f_elev.name}  (+{n_elev - n_pts:,} pts d'altitude)")

        # ── README ────────────────────────────────────────────────────────
        write_readme(folder, zone, n_pts, n_elev, depth_range, elevation_km)
        print(f"  ✓ README.txt")

        summary.append({
            "numero"   : zone["numero"],
            "nom"      : zone["nom"],
            "lon_min"  : zone["lon_min"],
            "lon_max"  : zone["lon_max"],
            "lat_min"  : zone["lat_min"],
            "lat_max"  : zone["lat_max"],
            "lon0_AE"  : zone["lon0"],
            "lat0_AE"  : zone["lat0"],
            "pts_std"  : n_pts,
            "pts_elev" : n_elev,
            "dossier"  : str(folder.resolve()),
        })

    return summary


# ── 6. Récapitulatif global ───────────────────────────────────────────────────

def write_global_recap(summary, root_dir):
    """Écrit tuiles_dir/recap_global.csv avec toutes les infos des tuiles."""
    if not summary:
        return
    df = pd.DataFrame(summary)
    out = Path(root_dir) / "recap_global.csv"
    df.to_csv(out, index=False)
    print(f"\n  Récap global → {out}")

    # Affichage console
    print(f"\n{'='*55}")
    print("RÉCAPITULATIF")
    print(f"{'='*55}")
    print(f"{'N°':>5}  {'Nom':<20}  {'lon0_AE':>8}  {'lat0_AE':>8}  "
          f"{'Pts std':>9}  {'Pts elev':>9}")
    print("-" * 70)
    for r in summary:
        print(f"{r['numero']:>5}  {r['nom']:<20}  {r['lon0_AE']:>8.4f}  "
              f"{r['lat0_AE']:>8.4f}  {r['pts_std']:>9,}  {r['pts_elev']:>9,}")
    print(f"\n{len(summary)} tuile(s) traitée(s)")
    print(f"Dossier de sortie : {Path(root_dir).resolve()}/")


# ── Programme principal ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Découpe un modèle de vitesse global en tuiles selon un fichier de zones."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML (défaut : config.yaml)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)
    paths  = config["paths"]
    seg    = config["segment"]

    print("=" * 55)
    print("Segmentation NLL — pilotée par fichier de zones")
    print("=" * 55)

    zones     = load_zones(paths["zones_file"])
    df_global = load_global_model(
        paths["model_file"], seg["model_sep"], seg["model_header"],
        seg["depth_in_meters"],
    )
    summary   = process_all_zones(
        df_global, zones, paths["tuiles_dir"],
        seg["elevation_km"], seg["elev_step_km"], seg["tol"],
    )
    write_global_recap(summary, paths["tuiles_dir"])

    print("\nTerminé.")
    print("=" * 55)


if __name__ == "__main__":
    main()
