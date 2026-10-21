"""
fetch_picks.py
==============
Récupère sur le serveur SeisComP (service web fdsnws-event) les événements
d'une période et d'une zone, avec leur origine préférée, ses arrivées et
ses pointés, puis les convertit en pointés NLLOC_OBS pour NLL en ligne de
commande.

Une requête par jour, au format sc3ml. Chaque journée est gardée dans
fetch.cache_dir (AAAA-MM-JJ.xml) : une journée déjà téléchargée n'est pas
redemandée (sauf --refresh), et un téléchargement interrompu reprend où il
s'est arrêté. Les pointés révisés sur le serveur depuis le téléchargement
ne sont donc pris en compte qu'avec --refresh.

Sorties (dans fetch.obs_dir) :
  - periode_<debut>_<fin>.obs              : pointés NLLOC_OBS, un bloc par
                                             événement (PUBLIC_ID = Event ID)
  - periode_<debut>_<fin>_evenements.csv   : un événement par ligne, avec
                                             l'origine de référence (LOCSAT)

Configuration (sections fetch et phases de config.yaml) :
  fetch.fdsnws_url       : URL du service fdsnws-event (…/fdsnws/event/1/query)
  fetch.zone             : {lat_min, lat_max, lon_min, lon_max}
  fetch.event_types      : types d'événements gardés (null → tous)
  fetch.cache_dir        : dossier des XML journaliers
  fetch.obs_dir          : dossier des fichiers .obs produits
  phases.default_error_s : incertitude des pointés qui n'en ont pas

Utilisation :
    python fetch_picks.py --debut 2024-11-01 --fin 2024-11-30
    python fetch_picks.py --debut 2024-11-01 --fin 2024-11-30 --refresh
"""

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

from nll_common import load_config, log
from seiscomp_xml_to_obs import default_errors_from_config, read_xml_files, write_obs

# XML écrit pour une journée sans événement (réponse 204 du serveur)
EMPTY_DAY_XML = "<seiscomp><EventParameters/></seiscomp>\n"


def day_url(cfg, day):
    """URL de la requête fdsnws-event d'une journée."""
    zone = cfg["zone"]
    params = {
        "starttime": day.isoformat(),
        "endtime": (day + timedelta(days=1)).isoformat(),
        "minlatitude": zone["lat_min"], "maxlatitude": zone["lat_max"],
        "minlongitude": zone["lon_min"], "maxlongitude": zone["lon_max"],
        "includearrivals": "true", "includepicks": "true",
        "format": "sc3ml",
    }
    return f"{cfg['fdsnws_url']}?{urllib.parse.urlencode(params)}"


def download_day(cfg, day, cache_dir, refresh=False, retries=3):
    """Télécharge une journée dans cache_dir/AAAA-MM-JJ.xml (si absente) et renvoie son chemin."""
    path = Path(cache_dir) / f"{day.isoformat()}.xml"
    if path.exists() and not refresh:
        return path
    url = day_url(cfg, day)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                content = resp.read() if resp.status != 204 else EMPTY_DAY_XML.encode()
            ET.fromstring(content)  # refuse un XML tronqué
            tmp = path.with_suffix(".part")
            tmp.write_bytes(content)
            tmp.replace(path)
            return path
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as err:
            if attempt == retries:
                raise RuntimeError(f"Échec du téléchargement du {day} : {err}\n{url}") from err
            time.sleep(5 * attempt)


def days_between(debut, fin):
    """Jours de debut à fin inclus."""
    return [debut + timedelta(days=i) for i in range((fin - debut).days + 1)]


def fetch_period(config, debut, fin, refresh=False):
    """
    Télécharge (si besoin) la période et écrit les pointés NLLOC_OBS.
    Renvoie (chemin du .obs, chemin du CSV récapitulatif).
    """
    cfg = config["fetch"]
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = Path(cfg["obs_dir"])
    obs_dir.mkdir(parents=True, exist_ok=True)

    days = days_between(debut, fin)
    missing = [d for d in days if refresh or not (cache_dir / f"{d.isoformat()}.xml").exists()]
    log(f"{len(days)} jours, {len(missing)} à télécharger depuis {cfg['fdsnws_url']}", 1)
    files = []
    for i, day in enumerate(days, 1):
        files.append(download_day(cfg, day, cache_dir, refresh))
        if day in missing and (i % 10 == 0 or i == len(days)):
            log(f"{i}/{len(days)} jours", 2)

    events, invalid, empty = read_xml_files(files)
    types = cfg.get("event_types")
    if types:
        events = [e for e in events if e["event_type"] in types]
    stem = f"periode_{debut.isoformat()}_{fin.isoformat()}"
    out_obs = obs_dir / f"{stem}.obs"
    out_csv = obs_dir / f"{stem}_evenements.csv"
    n_ev, n_ph, n_default = write_obs(events, out_obs, out_csv, default_errors_from_config(config))
    log(f"{n_ev} événements, {n_ph} phases ({n_default} sans incertitude → valeur par défaut)", 1)
    if invalid or empty:
        log(f"ignorés : {len(invalid)} XML illisibles, {empty} événements sans phase utilisée", 2)
    log(f"→ {out_obs}", 2)
    return out_obs, out_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Récupère les pointés SeisComP d'une période (fdsnws) au format NLLOC_OBS."
    )
    parser.add_argument("--debut", required=True, type=date.fromisoformat,
                        help="Premier jour (AAAA-MM-JJ)")
    parser.add_argument("--fin", required=True, type=date.fromisoformat,
                        help="Dernier jour, inclus (AAAA-MM-JJ)")
    parser.add_argument("--refresh", action="store_true",
                        help="Retélécharger les journées déjà en cache")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML (défaut : config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fin < args.debut:
        raise SystemExit("--fin doit être postérieure ou égale à --debut")
    config = load_config(args.config)
    log(f"Pointés SeisComP du {args.debut} au {args.fin}")
    fetch_period(config, args.debut, args.fin, args.refresh)
    log("Terminé.")


if __name__ == "__main__":
    main()
