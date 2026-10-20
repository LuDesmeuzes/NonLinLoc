"""
seiscomp_xml_to_obs.py
======================
Convertit les exports XML SeisComP (un fichier par événement, contenant
l'origine passée à NLL, ses arrivées et ses pointés) en un fichier de
pointés au format NLLOC_OBS, pour relancer NLL en ligne de commande
(standalone) sur exactement les mêmes phases que SeisComP.

Pour chaque fichier XML :
  - on prend l'origine (en pratique l'origine d'entrée, LOCSAT) et ses
    arrivées dont timeUsed est vrai ;
  - chaque arrivée renvoie à un pointé (pickID) : station, phase, temps et
    incertitude ;
  - l'incertitude est la moyenne de lowerUncertainty / upperUncertainty du
    pointé si elle existe, sinon la valeur par défaut de la phase
    (phases.default_error_s), à aligner sur NonLinLoc.defaultPickError de
    SeisComP.

Sortie : un fichier .obs par dossier de XML, événements séparés par une
ligne vide, et un CSV récapitulatif (un événement par ligne : fichier,
temps de l'origine, nombre de P et de S, gap, nombre de phases utilisées).

Configuration (section phases de config.yaml) :
  xml_dirs        : liste des dossiers de XML à convertir
  output_dir      : dossier où écrire les .obs
  default_error_s : incertitude par défaut (s) par type de phase, ex. {P: 0.1, S: 0.2}

Utilisation :
    python seiscomp_xml_to_obs.py
    python seiscomp_xml_to_obs.py --config un_autre_config.yaml
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from nll_common import load_config, log


def _strip_ns(root):
    """Retire l'espace de noms SeisComP des balises (version du schéma variable)."""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el, path, default=None):
    found = el.find(path)
    return found.text if found is not None and found.text is not None else default


def parse_event_xml(path):
    """
    Lit un XML SeisComP et renvoie (infos origine, liste de phases).
    Chaque phase : dict station, phase, time (datetime), error_s (float ou None).
    """
    root = _strip_ns(ET.parse(path).getroot())
    picks = {p.get("publicID"): p for p in root.iter("pick")}
    origin = next(root.iter("origin"), None)
    if origin is None:
        return None, []

    info = {
        "origin_id": origin.get("publicID"),
        "origin_time": _text(origin, "time/value"),
        "method": _text(origin, "methodID"),
        "gap": _text(origin, "quality/azimuthalGap"),
        "used_phases": _text(origin, "quality/usedPhaseCount"),
    }

    phases = []
    for arr in origin.iter("arrival"):
        if _text(arr, "timeUsed", "true").lower() != "true":
            continue
        pick = picks.get(_text(arr, "pickID"))
        if pick is None:
            continue
        wf = pick.find("waveformID")
        lo = _text(pick, "time/lowerUncertainty")
        up = _text(pick, "time/upperUncertainty")
        unc = [float(v) for v in (lo, up) if v is not None]
        phases.append({
            "station": wf.get("stationCode"),
            "phase": _text(arr, "phase") or _text(pick, "phaseHint", "?"),
            "time": datetime.fromisoformat(_text(pick, "time/value").replace("Z", "+00:00")),
            "error_s": sum(unc) / len(unc) if unc else None,
        })
    return info, sorted(phases, key=lambda p: p["time"])


def obs_line(ph, default_error):
    """Une ligne NLLOC_OBS (même mise en forme que nll_phases.txt)."""
    t = ph["time"]
    err = ph["error_s"] if ph["error_s"] is not None else default_error
    sec = t.second + t.microsecond / 1e6
    return (f"{ph['station']:<8s} ?    ?    ? {ph['phase']:<6s} ? "
            f"{t:%Y%m%d} {t:%H%M} {sec:8.4f} GAU {err:6.3f}      "
            f"-1.00e+00 -1.00e+00 -1.00e+00")


def convert_dir(xml_dir, out_dir, default_errors):
    xml_dir = Path(xml_dir)
    files = sorted(xml_dir.glob("*.xml"))
    if not files:
        log(f"⚠ Aucun XML dans {xml_dir}", 1)
        return
    out_obs = Path(out_dir) / f"{xml_dir.name}.obs"
    out_csv = Path(out_dir) / f"{xml_dir.name}_evenements.csv"
    n_ev = n_ph = n_default = 0
    invalid, empty = [], 0

    with open(out_obs, "w") as fobs, open(out_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["fichier", "origin_id", "origin_time", "method", "n_P", "n_S",
                    "gap", "used_phases"])
        for f in files:
            try:
                info, phases = parse_event_xml(f)
            except ET.ParseError:
                # Fichier vide ou tronqué (téléchargement wget raté)
                invalid.append(f.name)
                continue
            if not phases:
                empty += 1
                continue
            fobs.write(f"#event {f.stem} {len(phases)} picks\n")
            for ph in phases:
                ptype = ph["phase"][:1].upper()
                if ph["error_s"] is None:
                    n_default += 1
                fobs.write(obs_line(ph, default_errors.get(ptype, default_errors["P"])) + "\n")
            fobs.write("\n")
            n_ev += 1
            n_ph += len(phases)
            w.writerow([f.name, info["origin_id"], info["origin_time"], info["method"],
                        sum(p["phase"][:1].upper() == "P" for p in phases),
                        sum(p["phase"][:1].upper() == "S" for p in phases),
                        info["gap"], info["used_phases"]])

    log(f"{xml_dir.name} : {n_ev} événements, {n_ph} phases "
        f"({n_default} sans incertitude → valeur par défaut)", 1)
    if invalid or empty:
        log(f"ignorés : {len(invalid)} XML illisibles (vides ou tronqués), "
            f"{empty} sans phase utilisée", 2)
    if invalid:
        out_bad = Path(out_dir) / f"{xml_dir.name}_xml_illisibles.txt"
        out_bad.write_text("\n".join(invalid) + "\n")
        log(f"liste des XML illisibles → {out_bad}", 2)
    log(f"→ {out_obs}", 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convertit des XML SeisComP en pointés NLLOC_OBS."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML (défaut : config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg = config["phases"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    default_errors = {k.upper(): float(v) for k, v in cfg["default_error_s"].items()}

    log("Conversion XML SeisComP → NLLOC_OBS")
    for d in cfg["xml_dirs"]:
        convert_dir(d, out_dir, default_errors)
    log("Terminé.")


if __name__ == "__main__":
    main()
