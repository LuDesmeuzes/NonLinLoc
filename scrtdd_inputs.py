"""
scrtdd_inputs.py
================
Prépare les fichiers d'entrée de scrtdd (relocalisation double différence)
à partir d'un run NLL local (nll_run.py) : les positions de départ sont les
localisations du run, les pointés ceux des XML SeisComP téléchargés pour le
run (cache de fetch_picks.py).

scrtdd lit ces trois CSV à la place d'une liste d'origines de la base
(eventlist de sclistorg) : on peut ainsi relocaliser sur le serveur, avec le
profil scrtdd du serveur, des localisations qui n'existent qu'en local.

Événements gardés : localisés par NLLoc (LOCATED), avec un PUBLIC_ID, et
sur la période --debut / --fin si elle est donnée. Phases gardées : celles
utilisées par l'origine de référence (comme pour le run NLL), sans les
stations exclues du run (--exclure-hors-modele de nll_run.py).

Avec --origines reference, les positions de départ sont celles de l'origine
de référence (evenements.csv du run, en général le LOCSAT manuel) au lieu
des localisations NLL : mêmes événements, mêmes id, mêmes phases et mêmes
stations, pour comparer les deux points de départ de la double différence.

Sorties (dans <run.runs_dir>/<run>/scrtdd/ ou scrtdd_reference/, ou --sortie) :
  - station.csv   : latitude,longitude,elevation (m),networkCode,stationCode,locationCode
  - event.csv     : id,isotime,latitude,longitude,depth (km),magnitude
                    (id = 1, 2, … ; magnitude 0 si absente de l'export fdsnws)
  - phase.csv     : eventId,isotime,lowerUncertainty,upperUncertainty,type,
                    networkCode,stationCode,locationCode,channelCode
                    (incertitudes « nan » si le pointé n'en a pas)
  - eventlist_<run>.csv : une ligne par événement, dans l'ordre des id
                    (origin = origine de référence dont viennent les pointés,
                    event = Event ID SeisComP) : LocQuality (ScrtddRun) s'en
                    sert pour relier les id scrtdd aux Event ID.

Les positions des stations viennent de l'inventaire SeisComP
(fdsnws-station), valides sur la période du run.

Sur le serveur, copier le dossier scrtdd/ puis lancer (à vérifier avec
scrtdd --help, selon la version) :
    scrtdd --reloc-catalog station.csv,event.csv,phase.csv --profile <profil> \\
        --console=1 -v
Les sorties de scrtdd (reloc-event.csv…) sont à rapatrier dans le même
dossier, à côté de l'eventlist, pour les analyses dd_* de LocQuality.

Utilisation :
    python scrtdd_inputs.py --run 05_ubaye_1D_topo_20230514_20250630
    python scrtdd_inputs.py --run 05_ubaye_1D_topo_20230514_20250630 \\
        --debut 2024-12-01 --fin 2025-02-28 --sortie /chemin/vers/dossier
    python scrtdd_inputs.py --run 05_ubaye_1D_topo_20230514_20250630 --origines reference
"""

import argparse
import csv
import io
import math
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from fetch_picks import days_between
from nll_common import load_config, log, read_hyp
from seiscomp_xml_to_obs import read_xml_files

STATION_COLUMNS = ["latitude", "longitude", "elevation", "networkCode", "stationCode",
                   "locationCode"]
EVENT_COLUMNS = ["id", "isotime", "latitude", "longitude", "depth", "magnitude"]
PHASE_COLUMNS = ["eventId", "isotime", "lowerUncertainty", "upperUncertainty", "type",
                 "networkCode", "stationCode", "locationCode", "channelCode"]


def isotime(t):
    """datetime (UTC, naïf ou non) → 2024-05-14T12:34:15.229646Z"""
    return t.replace(tzinfo=None).isoformat(timespec="microseconds") + "Z"


def unc(value):
    return "nan" if value is None else f"{value:g}"


def station_inventory(url, stations, debut, fin):
    """
    Positions des stations (fdsnws-station, niveau station) valides sur la
    période : {(réseau, station): (lat, lon, altitude en m)}.
    """
    params = {"station": ",".join(sorted({s for _, s in stations})), "level": "station",
              "format": "text", "starttime": debut.isoformat(),
              "endtime": (fin + timedelta(days=1)).isoformat()}
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=120) as resp:
        text = resp.read().decode() if resp.status != 204 else ""
    if not text.strip():
        return {}
    inv = pd.read_csv(io.StringIO(text), sep="|")
    inv.columns = [c.strip().lstrip("#") for c in inv.columns]
    inv = inv.drop_duplicates(["Network", "Station"])
    return {(r.Network, r.Station): (r.Latitude, r.Longitude, r.Elevation)
            for r in inv.itertuples()}


def select_events(hyp_events, debut=None, fin=None):
    """Événements localisés, avec PUBLIC_ID, dans la période, triés par temps."""
    keep = [e for e in hyp_events if e["status"] == "LOCATED" and e["event_id"]]
    if debut:
        keep = [e for e in keep if e["time"].date() >= debut]
    if fin:
        keep = [e for e in keep if e["time"].date() <= fin]
    return sorted(keep, key=lambda e: e["time"])


def reference_positions(events, evenements_csv):
    """
    Mêmes événements (et même ordre) que events, avec la position et le temps
    de l'origine de référence lus dans evenements.csv. Les événements sans
    origine de référence complète sont ignorés.
    """
    ref = pd.read_csv(evenements_csv).dropna(subset=["origin_time", "latitude", "longitude", "depth"])
    ref = ref.drop_duplicates("event_id").set_index("event_id")
    out = []
    for ev in events:
        if ev["event_id"] not in ref.index:
            continue
        r = ref.loc[ev["event_id"]]
        out.append({**ev, "time": datetime.fromisoformat(r["origin_time"].replace("Z", "+00:00")),
                    "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
                    "depth": float(r["depth"])})
    return out


def build_tables(events, xml_events, excluded):
    """
    Lignes de event.csv, phase.csv et de l'eventlist (dicts), et l'ensemble
    des (réseau, station, location) pointés. Les événements sans pointé dans
    les XML sont ignorés.
    """
    ev_rows, ph_rows, el_rows, stations = [], [], [], set()
    for ev in events:
        xml = xml_events.get(ev["event_id"])
        phases = [p for p in (xml or {}).get("phases", []) if p["station"] not in excluded]
        if not phases:
            continue
        k = len(ev_rows) + 1
        mag = xml.get("magnitude")
        ev_rows.append({"id": k, "isotime": isotime(ev["time"]),
                        "latitude": f"{ev['latitude']:.6f}", "longitude": f"{ev['longitude']:.6f}",
                        "depth": f"{ev['depth']:.4f}",
                        "magnitude": f"{mag:.2f}" if mag is not None else "0"})
        el_rows.append({"origin": xml["origin_id"], "event": ev["event_id"]})
        for p in phases:
            ph_rows.append({"eventId": k, "isotime": isotime(p["time"]),
                            "lowerUncertainty": unc(p["lower_s"]),
                            "upperUncertainty": unc(p["upper_s"]),
                            "type": p["phase"], "networkCode": p["network"],
                            "stationCode": p["station"], "locationCode": p["location"],
                            "channelCode": p["channel"]})
            stations.add((p["network"], p["station"], p["location"]))
    return ev_rows, ph_rows, el_rows, stations


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prépare station.csv, event.csv et phase.csv pour scrtdd à partir d'un run NLL local."
    )
    parser.add_argument("--run", required=True, help="Nom du run (dossier de run.runs_dir)")
    parser.add_argument("--debut", type=date.fromisoformat,
                        help="Premier jour gardé (AAAA-MM-JJ, défaut : début du run)")
    parser.add_argument("--fin", type=date.fromisoformat,
                        help="Dernier jour gardé, inclus (AAAA-MM-JJ, défaut : fin du run)")
    parser.add_argument("--origines", choices=["nll", "reference"], default="nll",
                        help="Positions de départ : localisations NLL du run (défaut) ou "
                             "origines de référence de evenements.csv (LOCSAT)")
    parser.add_argument("--sortie", help="Dossier de sortie (défaut : <run>/scrtdd, "
                                         "ou <run>/scrtdd_reference avec --origines reference)")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML (défaut : config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_dir = Path(config["run"]["runs_dir"]) / args.run
    meta_path = run_dir / "run.yaml"
    if not meta_path.exists():
        raise SystemExit(f"Run introuvable : {meta_path}")
    meta = yaml.safe_load(meta_path.read_text())
    debut = args.debut or date.fromisoformat(meta["debut"])
    fin = args.fin or date.fromisoformat(meta["fin"])
    default_dir = "scrtdd" if args.origines == "nll" else "scrtdd_reference"
    out_dir = Path(args.sortie) if args.sortie else run_dir / default_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Run {args.run} : du {debut} au {fin}")
    events = select_events(read_hyp(run_dir / f"{args.run}.hyp"), debut, fin)
    log(f"{len(events)} événements localisés", 1)
    if args.origines == "reference":
        events = reference_positions(events, run_dir / "evenements.csv")
        log(f"positions de départ : origines de référence ({len(events)} événements)", 1)

    cache_dir = Path(config["fetch"]["cache_dir"])
    files = [cache_dir / f"{d.isoformat()}.xml" for d in days_between(debut, fin)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} journées absentes du cache {cache_dir} "
                         f"(ex. {missing[0]}) : relancer fetch_picks.py sur la période")
    xml_events, invalid, _ = read_xml_files(files)
    if invalid:
        log(f"⚠ {len(invalid)} XML illisibles ignorés", 1)
    xml_by_id = {e["event_id"]: e for e in xml_events}

    excluded = set((meta.get("stations_exclues") or {}).get("stations") or [])
    ev_rows, ph_rows, el_rows, stations = build_tables(events, xml_by_id, excluded)
    log(f"{len(ev_rows)} événements avec pointés, {len(ph_rows)} phases"
        + (f" ({len(excluded)} stations exclues du run)" if excluded else ""), 1)

    log("Inventaire des stations")
    inv = station_inventory(config["fetch"]["fdsnws_station_url"],
                            {(n, s) for n, s, _ in stations}, debut, fin)
    # Pointés dont le réseau diffère de l'inventaire (ex. XX.GREVE pour 9W.GREVE) :
    # rattachés au réseau de l'inventaire si le code de station n'y est pas ambigu
    networks = {}
    for net, sta in inv:
        networks.setdefault(sta, []).append(net)
    renamed = {(net, sta): networks[sta][0] for net, sta, _ in stations
               if (net, sta) not in inv and len(networks.get(sta, [])) == 1}
    if renamed:
        n_renamed = 0
        for row in ph_rows:
            new = renamed.get((row["networkCode"], row["stationCode"]))
            if new:
                row["networkCode"] = new
                n_renamed += 1
        stations = {(renamed.get((n, s), n), s, l) for n, s, l in stations}
        log(f"réseau corrigé d'après l'inventaire pour {n_renamed} phases : "
            + ", ".join(f"{n}.{s} → {new}.{s}" for (n, s), new in sorted(renamed.items())), 1)
    st_rows, unknown = [], []
    for net, sta, loc in sorted(stations):
        if (net, sta) not in inv:
            unknown.append(f"{net}.{sta}")
            continue
        lat, lon, elev = inv[(net, sta)]
        st_rows.append({"latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
                        "elevation": f"{elev:.1f}" if not math.isnan(elev) else "0",
                        "networkCode": net, "stationCode": sta, "locationCode": loc})
    log(f"{len(st_rows)} stations", 1)
    if unknown:
        # scrtdd refuse tout le catalogue si une phase cite une station absente
        # de station.csv (« unknown station ») : ces phases sont retirées
        n_before = len(ph_rows)
        ph_rows = [r for r in ph_rows if (r["networkCode"], r["stationCode"]) in inv]
        log(f"⚠ absentes de l'inventaire : {', '.join(unknown)} → "
            f"{n_before - len(ph_rows)} phases retirées", 1)

    write_csv(out_dir / "station.csv", STATION_COLUMNS, st_rows)
    write_csv(out_dir / "event.csv", EVENT_COLUMNS, ev_rows)
    write_csv(out_dir / "phase.csv", PHASE_COLUMNS, ph_rows)
    write_csv(out_dir / f"eventlist_{args.run}.csv", ["origin", "event"], el_rows)
    log(f"→ {out_dir}", 1)


if __name__ == "__main__":
    main()
