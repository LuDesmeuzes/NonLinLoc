"""
nll_pipeline.py
===============
Pipeline complet NLL pour toutes les tuiles générées par segment_from_zones.py.

Pour chaque tuile trouvée dans tuiles_dir, le script enchaîne automatiquement :

  Étape 1 — Projection AE + interpolation cartésienne
             Lit  : <tuile>/<tuile>_standard.csv   (ou _elevated.csv)
             Écrit: <tuile>/work/modele_projete.txt

  Étape 2 — Extraction du plus grand rectangle complet
             Lit  : <tuile>/work/modele_projete.txt
             Écrit: <tuile>/work/modele_rectangle.txt

  Étape 3 — Génération des fichiers NLL (.hdr + .buf)
             Lit  : <tuile>/work/modele_rectangle.txt
             Écrit: <tuile>/model/layer.P.mod.hdr  /  .buf
                    <tuile>/model/layer.S.mod.hdr  /  .buf

  Étape 4 — Génération du fichier de topographie pour LOCTOPO_SURFACE
             Télécharge le MNT SRTM1 (30 m) via srtm.py
             Écrit: <tuile>/model/topo.asc
             Format : imite exactement la sortie GMT (grdinfo + grd2xyz -Z)
             attendue par NLL. Résolution configurable via topo_step_deg.

Architecture finale dans le Finder :
  tuiles_dir/
  ├── Z01_Ubaye/
  │   ├── Z01_Ubaye_standard.csv
  │   ├── Z01_Ubaye_elevated.csv
  │   ├── README.txt
  │   ├── work/
  │   │   ├── modele_projete.txt
  │   │   └── modele_rectangle.txt
  │   └── model/
  │       ├── layer.P.mod.hdr  ← vitesse P
  │       ├── layer.P.mod.buf
  │       ├── layer.S.mod.hdr  ← vitesse S
  │       ├── layer.S.mod.buf
  │       └── topo.asc         ← topographie (LOCTOPO_SURFACE)
  ├── Z02_Belledonne/
  │   └── ...
  └── recap_pipeline.csv

Le centre de projection AE (lon0, lat0) et l'étendue géographique
sont lus automatiquement depuis le README.txt de chaque tuile.

Configuration :
  Tous les paramètres sont lus depuis un fichier
  YAML — voir config.example.yaml pour le détail des champs.

Utilisation :
    pip install -r requirements.txt
    cp config.example.yaml config.yaml   # une seule fois, puis adapter
    python nll_pipeline.py
    python nll_pipeline.py --config un_autre_config.yaml
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import csv as csv_module

from nll_common import (
    AzimuthalEquidistant,
    find_input_csv,
    load_config,
    log,
    read_bounds_from_readme,
    read_center_from_readme,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU CSV DE TUILE
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_model(filepath):
    """Charge le CSV produit par segment_from_zones.py."""
    data = {"lon": [], "lat": [], "depth": [], "Vp": [], "Vs": []}
    col_map = {
        "Longitude": "lon", "Latitude": "lat", "Depth": "depth",
        "Vitesse onde P": "Vp", "Vitesse onde S": "Vs"
    }
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv_module.DictReader(f)
        missing = [c for c in col_map if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Colonnes manquantes dans {filepath} : {missing}")
        for row in reader:
            for src, dst in col_map.items():
                data[dst].append(float(row[src]))
    return {k: np.array(v) for k, v in data.items()}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — PROJECTION AE + INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def step1_project_and_interpolate(csv_path, lon0, lat0, out_path, step_km, method):
    """Projection AE + interpolation 3D → modele_projete.txt"""
    log("Chargement du CSV …", 2)
    data = load_csv_model(csv_path)
    log(f"{len(data['lon']):,} points  |  "
        f"prof [{data['depth'].min():.1f}, {data['depth'].max():.1f}] km", 3)

    log("Projection Azimuthal Equidistant …", 2)
    proj = AzimuthalEquidistant(lon0, lat0)
    x_km, y_km = proj.forward(data["lon"], data["lat"])
    log(f"x [{x_km.min():.2f}, {x_km.max():.2f}] km  "
        f"y [{y_km.min():.2f}, {y_km.max():.2f}] km", 3)

    log(f"Interpolation (pas={step_km} km, méthode={method}) …", 2)
    z_km = data["depth"]

    def make_axis(arr):
        return np.arange(arr.min(), arr.max() + step_km * 0.5, step_km)

    xa = make_axis(x_km); ya = make_axis(y_km); za = make_axis(z_km)
    log(f"Grille : {len(xa)} × {len(ya)} × {len(za)} = "
        f"{len(xa)*len(ya)*len(za):,} pts", 3)

    pts_src = np.column_stack([x_km, y_km, z_km])
    Xg, Yg, Zg = np.meshgrid(xa, ya, za, indexing="ij")
    pts_tgt = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    rows = []
    for param, vals in [("Vp", data["Vp"]), ("Vs", data["Vs"])]:
        log(f"Interpolation {param} …", 3)
        vi = griddata(pts_src, vals, pts_tgt, method=method)
        rows.append(vi)

    Vp_flat = rows[0]; Vs_flat = rows[1]
    mask = ~(np.isnan(Vp_flat) | np.isnan(Vs_flat))
    out  = np.column_stack([
        Xg.ravel()[mask], Yg.ravel()[mask], Zg.ravel()[mask],
        Vp_flat[mask], Vs_flat[mask]
    ])
    np.savetxt(out_path, out, header="x_km   y_km   z_km   Vp   Vs", fmt="%.4f")
    log(f"✓ modele_projete.txt  ({len(out):,} lignes)", 2)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — EXTRACTION DU PLUS GRAND RECTANGLE
# ─────────────────────────────────────────────────────────────────────────────

def largest_rect_histogram(h):
    """
    Algorithme classique "plus grand rectangle dans un histogramme".
    h : liste/array de hauteurs.
    Retourne (aire, indice_gauche, indice_droit) du meilleur rectangle.
    """
    stack = []; best = (0, 0, 0)
    for j, hj in enumerate(h):
        left = j
        while stack and stack[-1][1] >= hj:
            pj, ph = stack.pop()
            a = ph * (j - pj)
            if a > best[0]: best = (a, pj, j - 1)
            left = pj
        stack.append((left, hj))
    for pj, ph in stack:
        a = ph * (len(h) - pj)
        if a > best[0]: best = (a, pj, len(h) - 1)
    return best


def step2_largest_rectangle(in_path, out_path, step_km):
    """Extrait la plus grande sous-grille rectangulaire complète."""
    tol  = step_km * 0.01
    data = np.loadtxt(in_path, comments="#")
    log(f"{len(data):,} lignes chargées", 2)

    def snap(arr):
        s = np.round(arr / step_km) * step_km
        return np.unique(np.round(s, 6))

    x_axis = snap(data[:, 0]); y_axis = snap(data[:, 1])
    nx, ny = len(x_axis), len(y_axis)
    log(f"Axes : {nx} x, {ny} y", 3)

    x_idx = {round(v, 6): i for i, v in enumerate(x_axis)}
    y_idx = {round(v, 6): i for i, v in enumerate(y_axis)}

    present = np.zeros((nx, ny), dtype=bool)
    for row in data:
        xi = round(round(row[0] / step_km) * step_km, 6)
        yi = round(round(row[1] / step_km) * step_km, 6)
        if xi in x_idx and yi in y_idx:
            present[x_idx[xi], y_idx[yi]] = True

    heights = np.zeros(ny, dtype=int)
    best    = (0, 0, 0, 0, 0)
    for i in range(nx):
        for j in range(ny):
            heights[j] = heights[j] + 1 if present[i, j] else 0
        area, jl, jr = largest_rect_histogram(heights)
        h  = heights[jl:jr+1].min()
        if area > best[0]:
            best = (area, i - h + 1, i, jl, jr)

    _, ix0, ix1, iy0, iy1 = best
    xmin, xmax = x_axis[ix0], x_axis[ix1]
    ymin, ymax = y_axis[iy0], y_axis[iy1]
    log(f"Rectangle : x [{xmin:.2f}, {xmax:.2f}]  y [{ymin:.2f}, {ymax:.2f}] km", 2)

    mask = (
        (data[:, 0] >= xmin - tol) & (data[:, 0] <= xmax + tol) &
        (data[:, 1] >= ymin - tol) & (data[:, 1] <= ymax + tol)
    )
    rect = data[mask]
    np.savetxt(out_path, rect, header="x_km   y_km   z_km   Vp   Vs", fmt="%.4f")
    log(f"✓ modele_rectangle.txt  ({len(rect):,} lignes)", 2)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — ÉCRITURE DES FICHIERS NLL
# ─────────────────────────────────────────────────────────────────────────────

def step3_write_nll(in_path, nll_dir, lon0, lat0, step_km, vel_in_ms):
    """Génère layer.P.mod.hdr/.buf et layer.S.mod.hdr/.buf"""
    data = np.loadtxt(in_path, comments="#")
    x_f, y_f, z_f = data[:, 0], data[:, 1], data[:, 2]
    Vp_f, Vs_f    = data[:, 3], data[:, 4]

    if vel_in_ms:
        Vp_f = Vp_f / 1000.0
        Vs_f = Vs_f / 1000.0
        log("Conversion m/s → km/s", 3)

    def unique_axis(arr):
        s = np.round(arr / step_km) * step_km
        return np.unique(np.round(s, 6))

    xa = unique_axis(x_f); ya = unique_axis(y_f); za = unique_axis(z_f)
    nx, ny, nz = len(xa), len(ya), len(za)
    log(f"Grille NLL : nx={nx}  ny={ny}  nz={nz}", 3)

    xi_map = {round(v, 6): i for i, v in enumerate(xa)}
    yi_map = {round(v, 6): i for i, v in enumerate(ya)}
    zi_map = {round(v, 6): i for i, v in enumerate(za)}

    Vp_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    Vs_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    for x, y, z, vp, vs in zip(x_f, y_f, z_f, Vp_f, Vs_f):
        xi = round(round(x / step_km) * step_km, 6)
        yi = round(round(y / step_km) * step_km, 6)
        zi = round(round(z / step_km) * step_km, 6)
        if xi in xi_map and yi in yi_map and zi in zi_map:
            ii, ji, ki = xi_map[xi], yi_map[yi], zi_map[zi]
            Vp_3d[ii, ji, ki] = vp
            Vs_3d[ii, ji, ki] = vs

    def to_slow_len(v3d):
        v_safe = np.where(v3d <= 0.0, 1e-6, v3d)
        return (1.0 / v_safe * step_km).astype(np.float32)

    SLp = to_slow_len(Vp_3d); SLs = to_slow_len(Vs_3d)
    x_orig, y_orig, z_orig = float(xa[0]), float(ya[0]), float(za[0])

    def write_hdr(path, wave):
        with open(path, "w") as f:
            f.write(f"{nx} {ny} {nz}  "
                    f"{x_orig:.4f} {y_orig:.4f} {z_orig:.4f}  "
                    f"{step_km:.4f} {step_km:.4f} {step_km:.4f}  SLOW_LEN\n")
            f.write("\n")
            f.write(f"TRANSFORM  AZIMUTHAL_EQUIDIST  RefEllipsoid WGS-84  "
                    f"LatOrig {lat0:.6f}  LongOrig {lon0:.6f}  RotCW 0.000000\n")
        log(f"✓ layer.{wave}.mod.hdr", 2)

    def write_buf(path, sl3d, wave):
        sl3d.ravel(order="C").tofile(path)
        expected = nx * ny * nz * 4
        actual   = Path(path).stat().st_size
        ok = "✓" if actual == expected else "⚠ TAILLE INCORRECTE"
        log(f"{ok} layer.{wave}.mod.buf  ({sl3d.nbytes/1e6:.1f} Mo)", 2)

    for wave, sl in [("P", SLp), ("S", SLs)]:
        write_hdr(nll_dir / f"layer.{wave}.mod.hdr", wave)
        write_buf(nll_dir / f"layer.{wave}.mod.buf", sl, wave)

    log(f"SLOW_LEN P [{SLp.min():.5f}, {SLp.max():.5f}] s", 3)
    log(f"SLOW_LEN S [{SLs.min():.5f}, {SLs.max():.5f}] s", 3)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — TOPOGRAPHIE POUR LOCTOPO_SURFACE
# ─────────────────────────────────────────────────────────────────────────────

def step4_generate_topo(nll_dir, lon_min, lon_max, lat_min, lat_max, step_deg):
    """
    Génère le fichier topo.asc au format exact attendu par NLL pour LOCTOPO_SURFACE.

    NLL lit un fichier produit par les commandes GMT :
        grdinfo topo.grd  > topo.grd.asc     ← en-tête de métadonnées
        grd2xyz topo.grd -Z >> topo.grd.asc  ← valeurs Z seules, sans lon/lat

    Structure du fichier produit :
        ── En-tête (imite la sortie de grdinfo) ──────────────────────────────
        topo.grd: Title: Topography
        topo.grd: Command:
        topo.grd: Remark:
        topo.grd: Gridline node registration used
        topo.grd: grd_type = Cartesian
        topo.grd: x_min: <lon_min>  x_max: <lon_max>  x_inc: <step>  name: x  nx: <nx>
        topo.grd: y_min: <lat_min>  y_max: <lat_max>  y_inc: <step>  name: y  ny: <ny>
        topo.grd: z_min: <zmin>  z_max: <zmax>  name: z
        topo.grd: node_offset = 0
        ── Données Z (grd2xyz -Z, ordre TL = nord→sud, ouest→est) ───────────
        <elev_m>
        <elev_m>
        ...

    Coordonnées : lon et lat en degrés décimaux, élévation en mètres.
    Ordre des données : ligne par ligne du nord (y_max) au sud (y_min),
                        de gauche (x_min) à droite (x_max) — ordre GMT TL.

    Paramètres
    ----------
    nll_dir   : Path — dossier de sortie (tuile/model/)
    lon_min/max, lat_min/max : float — étendue géographique de la tuile
    step_deg  : float — résolution en degrés
    """
    try:
        import srtm
    except ImportError:
        log("⚠ Package srtm.py non installé → pip install srtm.py", 2)
        log("  Fichier topo.asc non généré.", 2)
        return

    # Grille lon/lat (axes)
    lons = np.arange(lon_min, lon_max + step_deg * 0.5, step_deg)
    lats = np.arange(lat_min, lat_max + step_deg * 0.5, step_deg)
    nx, ny = len(lons), len(lats)

    log(f"Téléchargement SRTM  [{lon_min:.3f},{lon_max:.3f}] × "
        f"[{lat_min:.3f},{lat_max:.3f}]  "
        f"step={step_deg}°  grille {nx}×{ny} …", 2)

    # Récupération SRTM — ordre nord→sud (GMT TL) pour grd2xyz -Z
    elev_data = srtm.get_data()
    z_values  = []
    n_missing = 0

    for lat in lats[::-1]:          # nord → sud  (y_max en premier)
        for lon in lons:            # ouest → est
            elev = elev_data.get_elevation(lat, lon)
            if elev is None:
                z_values.append(np.nan)
                n_missing += 1
            else:
                z_values.append(float(elev))

    z_arr = np.array(z_values)

    # Interpolation des NaN par plus proche voisin
    if n_missing > 0:
        log(f"  {n_missing} points sans donnée SRTM → interpolation voisin", 3)
        # Coordonnées de tous les points (même ordre nord→sud)
        coords = np.array([
            (lon, lat)
            for lat in lats[::-1]
            for lon in lons
        ])
        valid   = ~np.isnan(z_arr)
        invalid =  np.isnan(z_arr)
        if valid.sum() > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords[valid])
            _, idx = tree.query(coords[invalid])
            z_arr[invalid] = z_arr[valid][idx]
        else:
            z_arr[:] = 0.0
            log("  ⚠ Aucune donnée SRTM — élévation fixée à 0 m", 3)

    z_min = float(np.nanmin(z_arr))
    z_max = float(np.nanmax(z_arr))

    # ── Écriture du fichier au format GMT grdinfo + grd2xyz -Z ────────────
    out_path = nll_dir / "topo.asc"
    with open(out_path, "w") as f:
        # En-tête : imite exactement la sortie de grdinfo
        f.write("topo.grd: Title: Topography\n")
        f.write("topo.grd: Command:\n")
        f.write("topo.grd: Remark:\n")
        f.write("topo.grd: Gridline node registration used\n")
        f.write("topo.grd: grd_type = Cartesian\n")
        f.write(
            f"topo.grd: x_min: {lon_min:.6f}  x_max: {lon_max:.6f}  "
            f"x_inc: {step_deg:.6f}  name: x  nx: {nx}\n"
        )
        f.write(
            f"topo.grd: y_min: {lat_min:.6f}  y_max: {lat_max:.6f}  "
            f"y_inc: {step_deg:.6f}  name: y  ny: {ny}\n"
        )
        f.write(
            f"topo.grd: z_min: {z_min:.4f}  z_max: {z_max:.4f}  name: z\n"
        )
        f.write("topo.grd: node_offset = 0\n")

        # Données Z seules, une valeur par ligne (grd2xyz -Z)
        for z in z_arr:
            f.write(f"{z:.1f}\n")

    log(f"✓ topo.asc  ({nx}×{ny}={nx*ny:,} pts  |  "
        f"élev [{z_min:.0f}, {z_max:.0f}] m)", 2)
    log(f"  → LOCTOPO_SURFACE {out_path.resolve()} 0", 2)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(tuiles_dir, use_elevated, step_km, velocities_in_ms,
                  interp_method, generate_topo, topo_step_deg):
    root = Path(tuiles_dir)
    if not root.exists():
        print(f"ERREUR : dossier introuvable : {root.resolve()}")
        sys.exit(1)

    # Découverte des tuiles = sous-dossiers contenant un README.txt
    tuile_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and (d / "README.txt").exists()
    ])

    if not tuile_dirs:
        print(f"Aucune tuile trouvée dans {root.resolve()}")
        print("Vérifier que segment_from_zones.py a bien été exécuté.")
        sys.exit(1)

    print("=" * 60)
    print(f"Pipeline NLL — {len(tuile_dirs)} tuile(s) détectée(s)")
    print(f"Dossier racine : {root.resolve()}")
    print(f"Fichier source : {'_elevated.csv' if use_elevated else '_standard.csv'}")
    print(f"Pas de grille  : {step_km} km")
    print(f"Topographie    : {'oui (step=' + str(topo_step_deg) + '°)' if generate_topo else 'non'}")
    print("=" * 60)

    summary = []

    for idx, tuile_dir in enumerate(tuile_dirs, 1):
        tag = tuile_dir.name
        print(f"\n{'━'*60}")
        print(f"[{idx}/{len(tuile_dirs)}]  {tag}")
        print(f"{'━'*60}")

        status = "OK"
        try:
            # ── Lecture du centre AE depuis README ────────────────────────
            readme_path = tuile_dir / "README.txt"
            lon0, lat0  = read_center_from_readme(readme_path)
            lon_min, lon_max, lat_min, lat_max = read_bounds_from_readme(readme_path)
            log(f"Centre AE : lon0={lon0:.4f}°  lat0={lat0:.4f}°", 1)
            log(f"Étendue   : lon [{lon_min:.3f},{lon_max:.3f}]  "
                f"lat [{lat_min:.3f},{lat_max:.3f}]", 1)

            # ── Fichier CSV source ────────────────────────────────────────
            csv_path = find_input_csv(tuile_dir, use_elevated)
            log(f"Source    : {csv_path.name}", 1)

            # ── Création des dossiers de travail ──────────────────────────
            work_dir = tuile_dir / "work"
            nll_dir  = tuile_dir / "model"
            work_dir.mkdir(exist_ok=True)
            nll_dir.mkdir(exist_ok=True)

            projete_path   = work_dir / "modele_projete.txt"
            rectangle_path = work_dir / "modele_rectangle.txt"

            # ── Étape 1 ───────────────────────────────────────────────────
            print()
            log("ÉTAPE 1 — Projection AE + interpolation", 1)
            step1_project_and_interpolate(
                csv_path, lon0, lat0, projete_path, step_km, interp_method
            )

            # ── Étape 2 ───────────────────────────────────────────────────
            print()
            log("ÉTAPE 2 — Extraction du plus grand rectangle", 1)
            step2_largest_rectangle(projete_path, rectangle_path, step_km)

            # ── Étape 3 ───────────────────────────────────────────────────
            print()
            log("ÉTAPE 3 — Écriture des fichiers NLL", 1)
            step3_write_nll(
                rectangle_path, nll_dir, lon0, lat0, step_km, velocities_in_ms
            )

            # ── Étape 4 ───────────────────────────────────────────────────
            if generate_topo:
                print()
                log("ÉTAPE 4 — Topographie SRTM (LOCTOPO_SURFACE)", 1)
                step4_generate_topo(
                    nll_dir, lon_min, lon_max, lat_min, lat_max, topo_step_deg
                )

            # ── Vérification finale ───────────────────────────────────────
            nll_files = list(nll_dir.glob("layer.*.mod.*"))
            topo_ok   = (nll_dir / "topo.asc").exists()
            log(f"\n✅ {tag} — {len(nll_files)} fichiers NLL"
                f"{' + topo.asc' if topo_ok else ''} générés dans model/", 1)

        except Exception as e:
            status = f"ERREUR : {e}"
            log(f"\n❌ {tag} — {status}", 1)
            traceback.print_exc()

        summary.append({"tuile": tag, "statut": status})

    # ── Récapitulatif global ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RÉCAPITULATIF DU PIPELINE")
    print(f"{'='*60}")
    ok_count  = sum(1 for r in summary if r["statut"] == "OK")
    err_count = len(summary) - ok_count
    for r in summary:
        icon = "✅" if r["statut"] == "OK" else "❌"
        print(f"  {icon}  {r['tuile']:<30}  {r['statut']}")

    print(f"\n  {ok_count}/{len(summary)} tuile(s) traitée(s) avec succès")
    if err_count:
        print(f"  {err_count} erreur(s) — consulter les messages ci-dessus")

    # Sauvegarde CSV du récapitulatif
    recap_path = root / "recap_pipeline.csv"
    pd.DataFrame(summary).to_csv(recap_path, index=False)
    print(f"\n  Récap → {recap_path}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline NLL complet pour toutes les tuiles générées par segment_from_zones.py."
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
    pipe   = config["pipeline"]

    run_pipeline(
        tuiles_dir=config["paths"]["tuiles_dir"],
        use_elevated=pipe["use_elevated"],
        step_km=pipe["step_km"],
        velocities_in_ms=pipe["velocities_in_ms"],
        interp_method=pipe["interp_method"],
        generate_topo=pipe["generate_topo"],
        topo_step_deg=pipe["topo_step_deg"],
    )


if __name__ == "__main__":
    main()
