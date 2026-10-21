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
  3. NLLoc en parallèle sur --jobs morceaux de la période ;
  4. regroupement des localisations dans un seul fichier .hyp.

Les stations des pointés sans grille de temps dans le profil sont signalées :
NLLoc ignore leurs phases.

Sorties (dans <run.runs_dir>/<nom>/) :
  - <nom>.hyp          : toutes les localisations (format NLLoc, PUBLIC_ID =
                         Event ID SeisComP), lues directement par LocQuality
  - evenements.csv     : événements de la période et origine de référence
  - run.yaml           : profil, période, paramètres de recherche, comptes
  - nlloc_<k>.in, obs/, loc/ : fichiers de travail de NLLoc

Configuration (sections run, fetch et phases de config.yaml) :
  run.profils_dir : dossier des profils (un sous-dossier par profil)
  run.runs_dir    : dossier où créer les runs
  run.nlloc       : chemin du binaire NLLoc

Utilisation :
    python nll_run.py --profil 1D_ubaye --debut 2024-11-01 --fin 2024-11-30
    python nll_run.py --profil 1D_ubaye --debut 2024-11-01 --fin 2024-11-30 \\
        --nom 1D_ubaye_local_test --jobs 6 \\
        --locsearch "OCT 15 15 15 0.001 50000 5000 0 1"
"""

import argparse
import os
import re
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path

import yaml

from fetch_picks import fetch_period
from nll_common import load_config, log

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


def time_grid_root(profile_dir):
    """Racine des grilles de temps du profil, ex. <profil>/time/layer."""
    hdrs = sorted((Path(profile_dir) / "time").glob("*.P.*.time.hdr"))
    if not hdrs:
        raise FileNotFoundError(f"Aucune grille de temps P dans {Path(profile_dir) / 'time'}")
    prefix = hdrs[0].name.split(".P.")[0]
    return Path(profile_dir) / "time" / prefix


def stations_without_grid(obs_path, grid_root):
    """Stations des pointés sans grille de temps, par type de phase : {'P': [...], 'S': [...]}."""
    used = {"P": set(), "S": set()}
    for line in Path(obs_path).read_text().splitlines():
        parts = line.split()
        if len(parts) > 5 and not line.startswith(("#", "PUBLIC_ID")):
            used.setdefault(parts[4][:1].upper(), set()).add(parts[0])
    missing = {}
    for phase, stations in used.items():
        missing[phase] = sorted(s for s in stations
                                if not Path(f"{grid_root}.{phase}.{s}.time.hdr").exists())
    return missing


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
        "LOCHYPOUT SAVE_NLLOC_SUM",
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

    log("Fichier de contrôle")
    lines = read_profile_conf(conf_path, profils_dir)
    if args.locsearch:
        lines = [f"LOCSEARCH {args.locsearch}" if l.split()[0] == "LOCSEARCH" else l for l in lines]
    topo = keyword_values(lines, "LOCTOPO_SURFACE")
    if topo and not Path(topo[0]).exists():
        raise SystemExit(f"Topographie du profil introuvable : {topo[0]}")
    grid_root = time_grid_root(profile_dir)
    log(f"LOCSEARCH {' '.join(keyword_values(lines, 'LOCSEARCH'))}", 1)
    log(f"grilles de temps : {grid_root}", 1)
    for phase, stations in stations_without_grid(obs_path, grid_root).items():
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
        "evenements": n_events, "localises": n_located, "rejetes": n_rejected,
        "duree_s": round(elapsed),
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    log(f"→ {run_dir / f'{name}.hyp'}", 1)
    log("Terminé.")


if __name__ == "__main__":
    main()
