"""
nll_run.py
==========
Lance NLL en ligne de commande (NLLoc) avec un profil SeisComP sur une
période, sans passer par le serveur SeisComP.

Étapes :
  1. pointés de la période (fetch_picks.py : téléchargement fdsnws si besoin) ;
  2. fichier de contrôle à partir de <run.profils_dir>/<profil>/NLL.<profil>.conf
     (la configuration du profil sur la machine SeisComP) : mêmes TRANS,
     LOCGRID, LOCSEARCH, LOCMETH… ; ${NLLROOT} est remplacé par
     run.profils_dir ; les grilles de temps sont celles de
     <profil>/time/<préfixe> ;
  3. avec --grilles : grilles de temps manquantes calculées par Grid2Time
     pour les stations pointées sur la période, en P et en S même si une
     seule phase y est pointée (pour scrtdd), à partir des lignes TRANS,
     GTFILES, GTMODE et GT_PLFD de <profil>/nll_<profil>.in, avec les
     positions de l'inventaire SeisComP (fdsnws-station) valides sur la
     période ;
  4. NLLoc en parallèle sur --jobs morceaux de la période ;
  5. regroupement des localisations dans un seul fichier .hyp.

Les stations des pointés sans grille de temps dans le profil sont signalées :
NLLoc ignore leurs phases. Avec une grille 3D, les stations hors de la grille
du modèle sont ignorées.

Sorties (dans <run.runs_dir>/<nom>/) :
  - <nom>.hyp          : toutes les localisations (format NLLoc, PUBLIC_ID =
                         Event ID SeisComP), lues directement par LocQuality
  - evenements.csv     : événements de la période et origine de référence
  - run.yaml           : profil, période, paramètres de recherche, comptes
  - phases.csv         : une ligne par phase des événements localisés (station,
                         phase, résidu, poids, distance, azimut…)
  - nlloc_<k>.in, obs/, loc/ : fichiers de travail de NLLoc

Configuration (sections run, fetch et phases de config.yaml) :
  run.profils_dir : dossier des profils (un sous-dossier par profil)
  run.runs_dir    : dossier où créer les runs
  run.nlloc       : chemin du binaire NLLoc (Grid2Time est cherché à côté)
  fetch.fdsnws_station_url : service fdsnws-station (positions des stations)

Utilisation :
    python nll_run.py --profil 1D_ubaye --debut 2024-11-01 --fin 2024-11-30
    python nll_run.py --profil 05_ubaye_3D_topo --debut 2024-11-01 --fin 2024-11-30 --grilles
    python nll_run.py --profil 1D_ubaye --debut 2024-11-01 --fin 2024-11-30 \\
        --nom 1D_ubaye_local_test --jobs 6 \\
        --locsearch "OCT 15 15 15 0.001 50000 5000 0 1"
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from fetch_picks import fetch_period
from nll_common import AzimuthalEquidistant, load_config, log

# Mots-clés gérés par nll_run.py, jamais recopiés depuis le .conf du profil
MANAGED_KEYWORDS = {"CONTROL", "LOCFILES", "LOCHYPOUT"}


def read_profile_conf(conf_path, profils_dir):
    """
    Lignes de contrôle NLL du profil, sans commentaires ni lignes gérées par
    nll_run.py ; ${NLLROOT} est remplacé par profils_dir.
    """
    lines = []
    for raw in Path(conf_path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.split()[0] in MANAGED_KEYWORDS:
            continue
        lines.append(line.replace("${NLLROOT}", str(profils_dir)))
    return lines


def keyword_values(lines, keyword):
    """Valeurs (liste de chaînes) de la première ligne commençant par keyword, ou None."""
    for line in lines:
        parts = line.split()
        if parts[0] == keyword:
            return parts[1:]
    return None


def read_grid_control(profile_dir, profils_dir):
    """
    Lignes de génération des grilles de <profil>/nll_<profil>.in : dict
    trans, model_root, time_root (de GTFILES), gtmode, gt_plfd ; None si le
    fichier n'existe pas.
    """
    path = Path(profile_dir) / f"nll_{Path(profile_dir).name}.in"
    if not path.exists():
        return None
    lines = read_profile_conf(path, profils_dir)
    gtfiles = keyword_values(lines, "GTFILES")
    if not gtfiles:
        return None
    return {
        "trans": " ".join(keyword_values(lines, "TRANS")),
        "model_root": gtfiles[0], "time_root": gtfiles[1],
        "gtmode": " ".join(keyword_values(lines, "GTMODE") or ["GRID3D", "ANGLES_YES"]),
        "gt_plfd": " ".join(keyword_values(lines, "GT_PLFD") or ["1.0e-4", "1"]),
    }


def time_grid_root(profile_dir, grid_control=None):
    """
    Racine des grilles de temps du profil, ex. <profil>/time/layer : celle de
    GTFILES dans nll_<profil>.in, sinon déduite des grilles présentes.
    """
    if grid_control:
        return Path(grid_control["time_root"])
    hdrs = sorted((Path(profile_dir) / "time").glob("*.P.*.time.hdr"))
    if not hdrs:
        raise FileNotFoundError(f"Aucune grille de temps P dans {Path(profile_dir) / 'time'}")
    prefix = hdrs[0].name.split(".P.")[0]
    return Path(profile_dir) / "time" / prefix


def station_coordinates(url, stations, debut, fin):
    """
    Positions des stations dans l'inventaire SeisComP (fdsnws-station),
    valides sur la période : {code: (lat, lon, altitude_km)}. Une station
    présente sous plusieurs réseaux garde la première ligne.
    """
    params = {"station": ",".join(sorted(stations)), "level": "station", "format": "text",
              "starttime": debut.isoformat(), "endtime": (fin + timedelta(days=1)).isoformat()}
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=120) as resp:
        text = resp.read().decode() if resp.status != 204 else ""
    if not text.strip():
        return {}
    inv = pd.read_csv(io.StringIO(text), sep="|")
    inv.columns = [c.strip().lstrip("#") for c in inv.columns]
    inv = inv.drop_duplicates("Station")
    return {r.Station: (r.Latitude, r.Longitude, r.Elevation / 1000.0) for r in inv.itertuples()}


def inside_model_grid(model_root, trans, lat, lon):
    """Vrai si la station tombe dans l'emprise horizontale du modèle (grilles 3D)."""
    head = Path(f"{model_root}.P.mod.hdr").read_text().split()
    nx, ny = int(head[0]), int(head[1])
    x0, y0, dx, dy = float(head[3]), float(head[4]), float(head[6]), float(head[7])
    t = trans.split()
    x, y = AzimuthalEquidistant(float(t[3]), float(t[2])).forward(lon, lat)
    return x0 <= x <= x0 + (nx - 1) * dx and y0 <= y <= y0 + (ny - 1) * dy


def make_time_grids(grid_control, missing, coords, grid2time, work_dir):
    """
    Lance Grid2Time pour les stations sans grille (missing : {phase: [codes]}),
    une phase après l'autre (Grid2Time recopie le modèle dans le dossier des
    grilles : deux processus en parallèle se marcheraient dessus).
    Renvoie {phase: [stations ignorées (sans position ou hors grille)]}.
    """
    Path(grid_control["time_root"]).parent.mkdir(parents=True, exist_ok=True)
    is_3d = "GRID3D" in grid_control["gtmode"]
    skipped = {}
    for phase, stations in missing.items():
        ok, skipped[phase] = [], []
        for s in stations:
            pos = coords.get(s)
            if pos is None or (is_3d and not inside_model_grid(
                    grid_control["model_root"], grid_control["trans"], pos[0], pos[1])):
                skipped[phase].append(s)
            else:
                ok.append(s)
        if not ok:
            continue
        gtsrce = work_dir / f"gtsrce_{phase}.txt"
        gtsrce.write_text("".join(f"GTSRCE {s} LATLON {coords[s][0]:.6f} {coords[s][1]:.6f} 0 "
                                  f"{coords[s][2]:.3f}\n" for s in ok))
        in_file = work_dir / f"grid2time_{phase}.in"
        in_file.write_text("\n".join([
            "CONTROL 1 159753",
            f"TRANS {grid_control['trans']}",
            f"GTFILES {grid_control['model_root']} {grid_control['time_root']} {phase}",
            f"GTMODE {grid_control['gtmode']}",
            f"INCLUDE {gtsrce}",
            f"GT_PLFD {grid_control['gt_plfd']}",
        ]) + "\n")
        log(f"Grid2Time {phase} : {len(ok)} stations", 1)
        with open(work_dir / f"grid2time_{phase}.log", "w") as out:
            if subprocess.run([str(grid2time), str(in_file)], stdout=out,
                              stderr=subprocess.STDOUT).returncode != 0:
                raise RuntimeError(f"Grid2Time a échoué (voir {work_dir / f'grid2time_{phase}.log'})")
    return skipped


def stations_used(obs_path):
    """Stations des pointés, par type de phase : {'P': {...}, 'S': {...}}."""
    used = {"P": set(), "S": set()}
    for line in Path(obs_path).read_text().splitlines():
        parts = line.split()
        if len(parts) > 5 and not line.startswith(("#", "PUBLIC_ID")):
            used.setdefault(parts[4][:1].upper(), set()).add(parts[0])
    return used


def stations_without_grid(obs_path, grid_root, both_phases=False):
    """
    Stations des pointés sans grille de temps, par type de phase : {'P': [...], 'S': [...]}.

    both_phases=True : chaque station pointée doit avoir ses grilles P et S,
    même si une seule phase y est pointée (scrtdd peut calculer des temps
    théoriques pour l'autre phase, en corrélation croisée).
    """
    used = stations_used(obs_path)
    if both_phases:
        pointed = set().union(*used.values())
        used = {"P": pointed, "S": pointed}
    missing = {}
    for phase, stations in used.items():
        missing[phase] = sorted(s for s in stations
                                if not Path(f"{grid_root}.{phase}.{s}.time.hdr").exists())
    return missing


def drop_stations(obs_path, stations, out_path):
    """Copie obs_path sans les phases des stations données ; renvoie (out_path, phases retirées)."""
    stations, kept, n_dropped = set(stations), [], 0
    for line in Path(obs_path).read_text().splitlines():
        parts = line.split()
        if len(parts) > 5 and not line.startswith(("#", "PUBLIC_ID")) and parts[0] in stations:
            n_dropped += 1
            continue
        kept.append(line)
    Path(out_path).write_text("\n".join(kept) + "\n")
    return Path(out_path), n_dropped


def split_obs(obs_path, n_chunks, out_dir):
    """Découpe un fichier NLLOC_OBS en n_chunks fichiers d'événements contigus."""
    blocks = [b for b in Path(obs_path).read_text().split("\n\n") if b.strip()]
    n_chunks = max(1, min(n_chunks, len(blocks)))
    size = -(-len(blocks) // n_chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n_chunks):
        chunk = blocks[k * size:(k + 1) * size]
        if not chunk:
            break
        path = out_dir / f"obs_{k}.obs"
        path.write_text("\n\n".join(chunk) + "\n\n")
        paths.append(path)
    return paths, len(blocks)


def control_file(profile_lines, obs_path, grid_root, loc_root, seed=159753):
    """Fichier de contrôle NLLoc : profil + LOCFILES / LOCHYPOUT de ce run."""
    return "\n".join([f"CONTROL 1 {seed}"] + profile_lines + [
        f"LOCFILES {obs_path} NLLOC_OBS {grid_root} {loc_root}",
        # Un .hyp par événement (avec ses phases) en plus de la synthèse :
        # extract_phases() en tire phases.csv puis les supprime
        "LOCHYPOUT SAVE_NLLOC_ALL",
    ]) + "\n"


def run_nlloc(nlloc, in_files, run_dir):
    """Lance NLLoc sur chaque fichier de contrôle en parallèle et attend la fin."""
    procs = []
    for f in in_files:
        out = open(run_dir / f"{f.stem}.log", "w")
        procs.append((f, subprocess.Popen([str(nlloc), str(f)], stdout=out,
                                          stderr=subprocess.STDOUT, cwd=run_dir), out))
    failed = []
    for f, p, out in procs:
        if p.wait() != 0:
            failed.append(f.name)
        out.close()
    if failed:
        raise RuntimeError(f"NLLoc a échoué pour {failed} (voir les .log dans {run_dir})")


def merge_hyp(loc_dirs, out_path):
    """Concatène les .hyp de synthèse de chaque morceau ; renvoie (localisés, rejetés)."""
    n_located = n_rejected = 0
    with open(out_path, "w") as out:
        for d in loc_dirs:
            for hyp in sorted(Path(d).glob("*.sum.grid0.loc.hyp")):
                text = hyp.read_text()
                n_located += len(re.findall(r'^NLLOC .*"LOCATED"', text, re.M))
                n_rejected += len(re.findall(r'^NLLOC .*"REJECTED"', text, re.M))
                out.write(text)
    return n_located, n_rejected


# Colonnes de phases.csv : celles d'une ligne du bloc PHASE d'un .hyp NLLoc,
# avant « > » (pointé) puis après (calcul NLLoc)
PHASE_COLUMNS = ["event_id", "station", "phase", "time", "error_s", "tt_pred_s",
                 "residual_s", "weight", "dist_km", "azimuth", "ray_azimuth",
                 "ray_dip", "tcorr_s"]


def parse_phase_line(event_id, line):
    """Une ligne du bloc PHASE → dict (colonnes PHASE_COLUMNS)."""
    left, right = line.split(">", 1)
    l, r = left.split(), right.split()
    # Heure et minutes ajoutées à part : NLLoc écrit « 2400 » (et non 0000 le
    # lendemain) pour une phase arrivée juste après minuit
    hhmm = int(l[7])
    time = datetime.strptime(l[6], "%Y%m%d") + timedelta(
        hours=hhmm // 100, minutes=hhmm % 100, seconds=float(l[8]))
    # Après « > » : TTpred Res Weight StaX StaY StaZ SDist SAzim RAz RDip RQual Tcorr …
    return {"event_id": event_id, "station": l[0], "phase": l[4],
            "time": time.isoformat(), "error_s": float(l[10]),
            "tt_pred_s": float(r[0]), "residual_s": float(r[1]), "weight": float(r[2]),
            "dist_km": float(r[6]), "azimuth": float(r[7]), "ray_azimuth": float(r[8]),
            "ray_dip": float(r[9]), "tcorr_s": float(r[11])}


def extract_phases(loc_dirs, out_csv):
    """
    Lit le bloc PHASE des .hyp par événement (localisés seulement) et écrit
    out_csv (une ligne par phase), puis supprime les fichiers par événement
    (.hyp, .hdr, .scat) et les last.* : la synthèse et phases.csv suffisent.
    Renvoie le nombre de phases.
    """
    rows = []
    for d in loc_dirs:
        for hyp in sorted(Path(d).glob("*.grid0.loc.hyp")):
            if ".sum." in hyp.name:
                continue
            text = hyp.read_text()
            event = re.search(r"^PUBLIC_ID (\S+)", text, re.M)
            if event and re.search(r'^NLLOC .*"LOCATED"', text, re.M):
                block = re.search(r"^PHASE .*?\n(.*?)^END_PHASE", text, re.M | re.S)
                if block:
                    rows += [parse_phase_line(event.group(1), l)
                             for l in block.group(1).splitlines() if ">" in l]
        for f in Path(d).iterdir():
            if (".sum." not in f.name and f.name.endswith((".grid0.loc.hyp", ".grid0.loc.hdr",
                                                           ".grid0.loc.scat"))) \
                    or f.name.startswith("last."):
                f.unlink()
    pd.DataFrame(rows, columns=PHASE_COLUMNS).to_csv(out_csv, index=False)
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lance NLLoc avec un profil SeisComP sur une période."
    )
    parser.add_argument("--profil", required=True, help="Nom du profil (dossier de run.profils_dir)")
    parser.add_argument("--debut", required=True, type=date.fromisoformat,
                        help="Premier jour (AAAA-MM-JJ)")
    parser.add_argument("--fin", required=True, type=date.fromisoformat,
                        help="Dernier jour, inclus (AAAA-MM-JJ)")
    parser.add_argument("--nom", help="Nom du run (défaut : <profil>_<debut>_<fin>)")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Nombre de NLLoc lancés en parallèle (défaut : cœurs − 1)")
    parser.add_argument("--locsearch", help="Remplace le LOCSEARCH du profil, ex. \"OCT 30 30 9 0.001 50000 5000 0 1\"")
    parser.add_argument("--refresh", action="store_true",
                        help="Retélécharger les pointés déjà en cache")
    parser.add_argument("--exclure-hors-modele", metavar="PROFIL",
                        help="Retirer les phases des stations hors de l'emprise du modèle 3D de PROFIL "
                             "(pour comparer des profils sur les mêmes phases)")
    parser.add_argument("--grilles", action="store_true",
                        help="Calculer (Grid2Time) les grilles de temps manquantes du profil")
    parser.add_argument("--force", action="store_true",
                        help="Écraser un run existant du même nom")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML (défaut : config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg = config["run"]
    profils_dir = Path(cfg["profils_dir"])
    profile_dir = profils_dir / args.profil
    conf_path = profile_dir / f"NLL.{args.profil}.conf"
    if not conf_path.exists():
        raise SystemExit(f"Configuration du profil introuvable : {conf_path}")
    name = args.nom or f"{args.profil}_{args.debut:%Y%m%d}_{args.fin:%Y%m%d}"
    run_dir = Path(cfg["runs_dir"]) / name
    if run_dir.exists():
        if not args.force:
            raise SystemExit(f"Le run {run_dir} existe déjà (--force pour l'écraser)")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    log(f"Run {name} : profil {args.profil}, du {args.debut} au {args.fin}")
    log("Pointés")
    obs_path, events_csv = fetch_period(config, args.debut, args.fin, args.refresh)
    shutil.copy(events_csv, run_dir / "evenements.csv")
    excluded, n_dropped = [], 0
    if args.exclure_hors_modele:
        ref_dir = profils_dir / args.exclure_hors_modele
        ref_control = read_grid_control(ref_dir, profils_dir)
        if ref_control is None:
            raise SystemExit(f"--exclure-hors-modele : pas de GTFILES dans {ref_dir}")
        used = set().union(*stations_used(obs_path).values())
        coords = station_coordinates(config["fetch"]["fdsnws_station_url"], used,
                                     args.debut, args.fin)
        excluded = sorted(s for s in used if s not in coords or not inside_model_grid(
            ref_control["model_root"], ref_control["trans"], coords[s][0], coords[s][1]))
        obs_path, n_dropped = drop_stations(obs_path, excluded, run_dir / "pointes_filtres.obs")
        log(f"{len(excluded)} stations hors du modèle de {args.exclure_hors_modele} : "
            f"{n_dropped} phases retirées", 1)

    log("Fichier de contrôle")
    lines = read_profile_conf(conf_path, profils_dir)
    if args.locsearch:
        lines = [f"LOCSEARCH {args.locsearch}" if l.split()[0] == "LOCSEARCH" else l for l in lines]
    topo = keyword_values(lines, "LOCTOPO_SURFACE")
    if topo and not Path(topo[0]).exists():
        raise SystemExit(f"Topographie du profil introuvable : {topo[0]}")
    grid_control = read_grid_control(profile_dir, profils_dir)
    grid_root = time_grid_root(profile_dir, grid_control)
    log(f"LOCSEARCH {' '.join(keyword_values(lines, 'LOCSEARCH'))}", 1)
    log(f"grilles de temps : {grid_root}", 1)
    missing = stations_without_grid(obs_path, grid_root)
    to_make = stations_without_grid(obs_path, grid_root, both_phases=True)

    if args.grilles and any(to_make.values()):
        if grid_control is None:
            raise SystemExit(f"--grilles : pas de GTFILES dans {profile_dir / f'nll_{args.profil}.in'}")
        log("Grilles de temps manquantes")
        all_missing = set().union(*to_make.values())
        coords = station_coordinates(config["fetch"]["fdsnws_station_url"], all_missing,
                                     args.debut, args.fin)
        grid2time = Path(cfg["nlloc"]).with_name("Grid2Time")
        start = datetime.now()
        skipped = make_time_grids(grid_control, to_make, coords, grid2time, run_dir)
        log(f"grilles calculées en {(datetime.now() - start).total_seconds():.0f} s", 1)
        for phase, stations in skipped.items():
            if stations:
                log(f"{phase} : {len(stations)} stations hors de la grille du modèle ou "
                    f"absentes de l'inventaire", 1)
        missing = stations_without_grid(obs_path, grid_root)

    for phase, stations in missing.items():
        if stations:
            log(f"⚠ {len(stations)} stations sans grille {phase} (phases ignorées) : "
                f"{', '.join(stations)}", 1)

    chunks, n_events = split_obs(obs_path, args.jobs, run_dir / "obs")
    in_files, loc_dirs = [], []
    for k, chunk in enumerate(chunks):
        loc_dir = run_dir / "loc" / f"part_{k}"
        loc_dir.mkdir(parents=True)
        in_file = run_dir / f"nlloc_{k}.in"
        in_file.write_text(control_file(lines, chunk, grid_root, loc_dir / "loc"))
        in_files.append(in_file)
        loc_dirs.append(loc_dir)

    log(f"NLLoc : {n_events} événements en {len(chunks)} morceaux parallèles")
    start = datetime.now()
    run_nlloc(cfg["nlloc"], in_files, run_dir)
    elapsed = (datetime.now() - start).total_seconds()
    n_located, n_rejected = merge_hyp(loc_dirs, run_dir / f"{name}.hyp")
    log(f"{n_located} localisés, {n_rejected} rejetés en {elapsed:.0f} s", 1)
    n_phases = extract_phases(loc_dirs, run_dir / "phases.csv")
    log(f"{n_phases} phases (résidus) → phases.csv", 1)

    meta = {
        "nom": name, "profil": args.profil,
        "debut": args.debut.isoformat(), "fin": args.fin.isoformat(),
        "lance_le": start.isoformat(timespec="seconds"),
        "locsearch": " ".join(keyword_values(lines, "LOCSEARCH")),
        "locgrid": " ".join(keyword_values(lines, "LOCGRID")),
        "loctopo_surface": topo[0] if topo else None,
        "grilles_temps": str(grid_root),
        "pointes": str(obs_path),
        "erreur_pointe_defaut_s": config["phases"]["default_error_s"],
        "stations_exclues": {"hors_modele_de": args.exclure_hors_modele,
                             "stations": excluded, "phases_retirees": n_dropped}
                            if args.exclure_hors_modele else None,
        "evenements": n_events, "localises": n_located, "rejetes": n_rejected,
        "phases": n_phases,
        "duree_s": round(elapsed),
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    log(f"→ {run_dir / f'{name}.hyp'}", 1)
    log("Terminé.")


if __name__ == "__main__":
    main()
