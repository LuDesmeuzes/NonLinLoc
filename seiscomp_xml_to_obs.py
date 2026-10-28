"""
seiscomp_xml_to_obs.py
======================
Convertit les exports XML SeisComP (sc3ml) en un fichier de pointés au
format NLLOC_OBS, pour relancer NLL en ligne de commande (standalone) sur
exactement les mêmes phases que SeisComP.

Un XML peut contenir un seul événement (export par eventid) ou plusieurs
(export d'une journée par fetch_picks.py). Pour chaque événement :
  - on prend son origine préférée (en pratique la localisation manuelle
    LOCSAT, celle qui est passée à NLL) et ses arrivées dont timeUsed est
    vrai ; sans élément <event>, la première origine du fichier ;
  - chaque arrivée renvoie à un pointé (pickID) : station, phase, temps et
    incertitude ;
  - l'incertitude est la moyenne de lowerUncertainty / upperUncertainty du
    pointé si elle existe, sinon la valeur par défaut de la phase
    (phases.default_error_s), alignée sur NonLinLoc.defaultPickError de
    SeisComP.

Sortie : un fichier .obs par dossier de XML (un bloc par événement, précédé
de « PUBLIC_ID <Event ID> », que NLLoc recopie dans ses .hyp), et un CSV
récapitulatif (un événement par ligne : Event ID, origine de référence avec
position, incertitudes et RMS, nombre de P et de S, gap).

Configuration (section phases de config.yaml) :
  xml_dirs        : liste des dossiers de XML à convertir
  output_dir      : dossier où écrire les .obs
  default_error_s : incertitude par défaut (s) par type de phase, ex. {P: 0.02, S: 0.02}

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

# Colonnes du CSV récapitulatif (une ligne par événement)
SUMMARY_COLUMNS = ["event_id", "event_type", "origin_id", "origin_time", "method",
                   "latitude", "longitude", "depth", "lat_unc_km", "lon_unc_km",
                   "depth_unc_km", "rms", "n_P", "n_S", "gap", "used_phases", "source"]


def _strip_ns(root):
    """Retire l'espace de noms SeisComP des balises (version du schéma variable)."""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _text(el, path, default=None):
    found = el.find(path)
    return found.text if found is not None and found.text is not None else default


def _float(el, path):
    value = _text(el, path)
    return float(value) if value is not None else None


def _phases(origin, picks):
    """
    Phases utilisées par une origine : dict station, phase, time, error_s,
    et pour scrtdd network, location, channel, lower_s, upper_s.
    """
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
        if not unc and _text(pick, "time/uncertainty") is not None:
            unc = [float(_text(pick, "time/uncertainty"))]
        phases.append({
            "station": wf.get("stationCode"),
            "phase": _text(arr, "phase") or _text(pick, "phaseHint", "?"),
            "time": datetime.fromisoformat(_text(pick, "time/value").replace("Z", "+00:00")),
            "error_s": sum(unc) / len(unc) if unc else None,
            "network": wf.get("networkCode"),
            "location": wf.get("locationCode") or "",
            "channel": wf.get("channelCode") or "",
            "lower_s": float(lo) if lo is not None else None,
            "upper_s": float(up) if up is not None else None,
        })
    return sorted(phases, key=lambda p: p["time"])


def _origin_info(origin):
    return {
        "origin_id": origin.get("publicID"),
        "origin_time": _text(origin, "time/value"),
        "method": _text(origin, "methodID"),
        "latitude": _float(origin, "latitude/value"),
        "longitude": _float(origin, "longitude/value"),
        "depth": _float(origin, "depth/value"),
        "lat_unc_km": _float(origin, "latitude/uncertainty"),
        "lon_unc_km": _float(origin, "longitude/uncertainty"),
        "depth_unc_km": _float(origin, "depth/uncertainty"),
        "rms": _float(origin, "quality/standardError"),
        "gap": _text(origin, "quality/azimuthalGap"),
        "used_phases": _text(origin, "quality/usedPhaseCount"),
    }


def parse_events_xml(path):
    """
    Lit un XML SeisComP et renvoie une liste d'événements : dict avec
    event_id (None si le XML n'a pas d'élément <event>), event_type,
    magnitude (préférée, None si absente), les infos de l'origine utilisée
    (_origin_info) et phases (liste de dicts, voir _phases).
    """
    root = _strip_ns(ET.parse(path).getroot())
    picks = {p.get("publicID"): p for p in root.iter("pick")}
    origins = {o.get("publicID"): o for o in root.iter("origin")}
    if not origins:
        return []

    xml_events = list(root.iter("event"))
    if not xml_events:
        # Export sans élément <event> : première origine du fichier
        origin = next(iter(origins.values()))
        return [{"event_id": None, "event_type": None,
                 **_origin_info(origin), "phases": _phases(origin, picks)}]

    # Magnitudes : l'export fdsnws n'en contient en général que quelques-unes
    magnitudes = {m.get("publicID"): _float(m, "magnitude/value") for m in root.iter("magnitude")}
    events = []
    for ev in xml_events:
        origin = origins.get(_text(ev, "preferredOriginID"))
        if origin is None:
            continue
        events.append({"event_id": ev.get("publicID"), "event_type": _text(ev, "type"),
                       "magnitude": magnitudes.get(_text(ev, "preferredMagnitudeID")),
                       **_origin_info(origin), "phases": _phases(origin, picks)})
    return events


def obs_line(ph, default_error):
    """Une ligne NLLOC_OBS (même mise en forme que nll_phases.txt)."""
    t = ph["time"]
    err = ph["error_s"] if ph["error_s"] is not None else default_error
    sec = t.second + t.microsecond / 1e6
    return (f"{ph['station']:<8s} ?    ?    ? {ph['phase']:<6s} ? "
            f"{t:%Y%m%d} {t:%H%M} {sec:8.4f} GAU {err:6.3f}      "
            f"-1.00e+00 -1.00e+00 -1.00e+00")


def write_obs(events, out_obs, out_csv, default_errors):
    """
    Écrit les événements (sortie de parse_events_xml, event_id renseigné)
    dans un fichier NLLOC_OBS et un CSV récapitulatif. Renvoie
    (nombre d'événements, nombre de phases, phases sans incertitude).
    """
    n_ev = n_ph = n_default = 0
    with open(out_obs, "w") as fobs, open(out_csv, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for ev in events:
            phases = ev["phases"]
            fobs.write(f"#event {ev['event_id']} {len(phases)} picks\n")
            fobs.write(f"PUBLIC_ID {ev['event_id']}\n")
            for ph in phases:
                ptype = ph["phase"][:1].upper()
                if ph["error_s"] is None:
                    n_default += 1
                fobs.write(obs_line(ph, default_errors.get(ptype, default_errors["P"])) + "\n")
            fobs.write("\n")
            n_ev += 1
            n_ph += len(phases)
            w.writerow({**ev,
                        "n_P": sum(p["phase"][:1].upper() == "P" for p in phases),
                        "n_S": sum(p["phase"][:1].upper() == "S" for p in phases)})
    return n_ev, n_ph, n_default


def read_xml_files(files):
    """
    Lit une liste de XML et renvoie (événements avec phases, XML illisibles,
    nombre d'événements sans phase). Un événement présent dans plusieurs
    fichiers n'est gardé qu'une fois. Sans Event ID, le nom du fichier sert
    d'identifiant.
    """
    events, invalid, empty, seen = [], [], 0, set()
    for f in files:
        try:
            parsed = parse_events_xml(f)
        except ET.ParseError:
            # Fichier vide ou tronqué (téléchargement raté)
            invalid.append(Path(f).name)
            continue
        for ev in parsed:
            ev["event_id"] = ev["event_id"] or Path(f).stem
            ev["source"] = Path(f).name
            if ev["event_id"] in seen:
                continue
            seen.add(ev["event_id"])
            if not ev["phases"]:
                empty += 1
                continue
            events.append(ev)
    events.sort(key=lambda e: e["origin_time"] or "")
    return events, invalid, empty


def convert_files(files, out_obs, default_errors):
    """Convertit une liste de XML en out_obs (+ <out_obs>_evenements.csv)."""
    out_obs = Path(out_obs)
    out_csv = out_obs.with_name(f"{out_obs.stem}_evenements.csv")
    events, invalid, empty = read_xml_files(files)
    n_ev, n_ph, n_default = write_obs(events, out_obs, out_csv, default_errors)
    log(f"{out_obs.stem} : {n_ev} événements, {n_ph} phases "
        f"({n_default} sans incertitude → valeur par défaut)", 1)
    if invalid or empty:
        log(f"ignorés : {len(invalid)} XML illisibles (vides ou tronqués), "
            f"{empty} événements sans phase utilisée", 2)
    if invalid:
        out_bad = out_obs.with_name(f"{out_obs.stem}_xml_illisibles.txt")
        out_bad.write_text("\n".join(invalid) + "\n")
        log(f"liste des XML illisibles → {out_bad}", 2)
    log(f"→ {out_obs}", 2)
    return out_obs, out_csv


def convert_dir(xml_dir, out_dir, default_errors):
    xml_dir = Path(xml_dir)
    files = sorted(xml_dir.glob("*.xml"))
    if not files:
        log(f"⚠ Aucun XML dans {xml_dir}", 1)
        return
    convert_files(files, Path(out_dir) / f"{xml_dir.name}.obs", default_errors)


def default_errors_from_config(config):
    return {k.upper(): float(v) for k, v in config["phases"]["default_error_s"].items()}


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
    default_errors = default_errors_from_config(config)

    log("Conversion XML SeisComP → NLLOC_OBS")
    for d in cfg["xml_dirs"]:
        convert_dir(d, out_dir, default_errors)
    log("Terminé.")


if __name__ == "__main__":
    main()
