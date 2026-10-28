import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nll_pipeline import largest_rect_histogram, step2_largest_rectangle


def test_largest_rect_histogram_simple():
    # Histogramme plat de hauteur 3 sur 4 colonnes → aire max = 12
    area, left, right = largest_rect_histogram([3, 3, 3, 3])
    assert area == 12
    assert (left, right) == (0, 3)


def test_largest_rect_histogram_with_dip():
    # Un creux au milieu limite le plus grand rectangle plein-hauteur
    # aux colonnes autour du creux ; ici la meilleure aire vient des 2
    # premières colonnes de hauteur 4 (aire 8) plutôt que du bloc entier.
    area, left, right = largest_rect_histogram([4, 4, 1, 4, 4])
    assert area == 8
    assert (left, right) == (0, 1)


def test_step2_largest_rectangle_extracts_full_subgrid(tmp_path):
    """
    Grille 3x3 en x/y avec un point manquant en (1,1) : le plus grand
    rectangle complet doit être une bande de 3x1 ou 1x3 (aire 3),
    jamais la grille entière (qui contient un trou).
    """
    step_km = 1.0
    rows = []
    for x in (0.0, 1.0, 2.0):
        for y in (0.0, 1.0, 2.0):
            if (x, y) == (1.0, 1.0):
                continue  # point manquant
            rows.append([x, y, 0.0, 5.0, 3.0])

    in_path = tmp_path / "modele_projete.txt"
    out_path = tmp_path / "modele_rectangle.txt"
    np.savetxt(in_path, np.array(rows), header="x_km y_km z_km Vp Vs", fmt="%.4f")

    step2_largest_rectangle(in_path, out_path, step_km)
    result = np.loadtxt(out_path, comments="#")

    # Le point manquant ne doit jamais apparaître, et le rectangle extrait
    # doit être complet (pas de trou) et non-trivial (plus d'un point).
    assert len(result) >= 3
    extracted_xy = {(round(r[0], 4), round(r[1], 4)) for r in result}
    assert (1.0, 1.0) not in extracted_xy


def test_topo_header_matches_nll_reader():
    """
    NLL (velmod.c, read_grd) lit l'en-tête par position : ligne 4 doit dire
    "Normal" (enregistrement aux nœuds) et ligne 9 donner scale_factor, sinon
    les altitudes sont multipliées par un facteur non initialisé.
    """
    from nll_pipeline import topo_header

    lines = topo_header(6.11, 7.39, 44.06, 44.96, 0.001, 1281, 901, 269.0, 3943.0).splitlines()
    assert len(lines) == 9
    assert lines[3].split()[1] == "Normal"
    assert lines[5].split()[-1] == "1281" and "x_inc: 0.001000" in lines[5]
    assert lines[6].split()[-1] == "901"
    assert lines[8] == "topo.grd: scale_factor: 1 add_offset: 0"


def test_seiscomp_xml_to_obs(tmp_path):
    """Une arrivée utilisée → une ligne NLLOC_OBS ; l'arrivée non utilisée est ignorée."""
    from seiscomp_xml_to_obs import obs_line, parse_events_xml

    xml = tmp_path / "ev.xml"
    xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<seiscomp xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.13"><EventParameters>'
        '<pick publicID="p1"><time><value>2023-05-14T06:25:18.5603Z</value></time>'
        '<waveformID networkCode="FR" stationCode="CREF"/><phaseHint>Pg</phaseHint></pick>'
        '<pick publicID="p2"><time><value>2023-05-14T06:25:23.7164Z</value>'
        '<lowerUncertainty>0.1</lowerUncertainty><upperUncertainty>0.3</upperUncertainty></time>'
        '<waveformID networkCode="FR" stationCode="JAUF"/><phaseHint>Sg</phaseHint></pick>'
        '<pick publicID="p3"><time><value>2023-05-14T06:25:30.0Z</value></time>'
        '<waveformID networkCode="FR" stationCode="SURF"/></pick>'
        '<origin publicID="o1"><time><value>2023-05-14T06:25:16.7Z</value></time><methodID>LOCSAT</methodID>'
        '<arrival><pickID>p1</pickID><phase>Pg</phase><timeUsed>true</timeUsed></arrival>'
        '<arrival><pickID>p2</pickID><phase>Sg</phase><timeUsed>true</timeUsed></arrival>'
        '<arrival><pickID>p3</pickID><phase>Pg</phase><timeUsed>false</timeUsed></arrival>'
        '</origin></EventParameters></seiscomp>'
    )
    [event] = parse_events_xml(xml)
    assert event["event_id"] is None
    assert event["method"] == "LOCSAT"
    phases = event["phases"]
    assert [p["station"] for p in phases] == ["CREF", "JAUF"]
    assert phases[0]["error_s"] is None
    assert phases[1]["error_s"] == pytest.approx(0.2)

    line = obs_line(phases[0], 0.1).split()
    assert line[0] == "CREF" and line[4] == "Pg"
    assert line[6:9] == ["20230514", "0625", "18.5603"]
    assert line[9:11] == ["GAU", "0.100"]


def test_parse_events_xml_journee(tmp_path):
    """XML d'une journée : un événement par <event>, avec son origine préférée."""
    from seiscomp_xml_to_obs import parse_events_xml, write_obs

    xml = tmp_path / "2024-11-01.xml"
    xml.write_text(
        '<seiscomp xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.13"><EventParameters>'
        '<pick publicID="p1"><time><value>2024-11-01T23:55:56.38Z</value></time>'
        '<waveformID networkCode="FR" stationCode="SURF"/></pick>'
        '<pick publicID="p2"><time><value>2024-11-01T22:39:53.1Z</value></time>'
        '<waveformID networkCode="FR" stationCode="JAUF"/></pick>'
        '<origin publicID="o1"><time><value>2024-11-01T23:55:55.36Z</value></time>'
        '<latitude><value>44.49</value><uncertainty>0.5</uncertainty></latitude>'
        '<longitude><value>6.88</value><uncertainty>0.4</uncertainty></longitude>'
        '<depth><value>1.57</value></depth><methodID>LOCSAT</methodID>'
        '<quality><standardError>0.12</standardError></quality>'
        '<arrival><pickID>p1</pickID><phase>P</phase><timeUsed>true</timeUsed></arrival></origin>'
        '<origin publicID="o2"><time><value>2024-11-01T22:39:51.9Z</value></time><methodID>LOCSAT</methodID>'
        '<arrival><pickID>p2</pickID><phase>S</phase><timeUsed>true</timeUsed></arrival></origin>'
        '<event publicID="ev1"><preferredOriginID>o1</preferredOriginID><type>earthquake</type></event>'
        '<event publicID="ev2"><preferredOriginID>o2</preferredOriginID><type>earthquake</type></event>'
        '</EventParameters></seiscomp>'
    )
    events = parse_events_xml(xml)
    assert [e["event_id"] for e in events] == ["ev1", "ev2"]
    assert events[0]["origin_id"] == "o1" and events[0]["latitude"] == pytest.approx(44.49)
    assert events[0]["lat_unc_km"] == pytest.approx(0.5) and events[0]["rms"] == pytest.approx(0.12)
    assert [p["station"] for p in events[1]["phases"]] == ["JAUF"]

    obs, csv_path = tmp_path / "p.obs", tmp_path / "p.csv"
    n_ev, n_ph, n_default = write_obs(events, obs, csv_path, {"P": 0.02, "S": 0.02})
    assert (n_ev, n_ph, n_default) == (2, 2, 2)
    text = obs.read_text()
    assert "PUBLIC_ID ev1" in text and "PUBLIC_ID ev2" in text
    assert " 0.020 " in text


def test_read_profile_conf(tmp_path):
    """Commentaires et lignes gérées par nll_run.py retirés, ${NLLROOT} remplacé."""
    from nll_run import control_file, keyword_values, read_profile_conf

    conf = tmp_path / "NLL.p.conf"
    conf.write_text(
        "CONTROL 9 159753\n"
        "# commentaire\n"
        "LOCSEARCH OCT 30 30 9 0.001 50000 5000 0 1    # depuis v2\n"
        "LOCFILES a b c d\n"
        "LOCTOPO_SURFACE ${NLLROOT}/p/topo.asc\n"
    )
    lines = read_profile_conf(conf, "/profils")
    assert lines == ["LOCSEARCH OCT 30 30 9 0.001 50000 5000 0 1",
                     "LOCTOPO_SURFACE /profils/p/topo.asc"]
    assert keyword_values(lines, "LOCSEARCH")[:4] == ["OCT", "30", "30", "9"]
    text = control_file(lines, "obs.obs", "time/layer", "loc/loc")
    assert text.startswith("CONTROL 1 ")
    assert "LOCFILES obs.obs NLLOC_OBS time/layer loc/loc" in text


def test_read_grid_control_and_model_bounds(tmp_path):
    """Lignes Grid2Time lues dans nll_<profil>.in ; emprise du modèle 3D."""
    from nll_run import inside_model_grid, read_grid_control, time_grid_root

    profile = tmp_path / "profils" / "p3d"
    (profile / "model").mkdir(parents=True)
    (profile / "nll_p3d.in").write_text(
        "CONTROL 1 159753\n"
        "TRANS AZIMUTHAL_EQUIDIST WGS-84 44.505000 6.745000 0\n"
        "GTFILES ${NLLROOT}/p3d/model/layer ${NLLROOT}/p3d/time/layer P   # commentaire\n"
        "GTMODE GRID3D ANGLES_YES\n"
    )
    gc = read_grid_control(profile, tmp_path / "profils")
    assert gc["model_root"] == str(tmp_path / "profils" / "p3d" / "model" / "layer")
    assert gc["gtmode"] == "GRID3D ANGLES_YES" and gc["gt_plfd"] == "1.0e-4 1"
    assert time_grid_root(profile, gc) == tmp_path / "profils" / "p3d" / "time" / "layer"

    # Modèle de 20 x 20 km centré sur l'origine de TRANS
    (profile / "model" / "layer.P.mod.hdr").write_text(
        "41 41 10  -10.0 -10.0 -5.0  0.5 0.5 0.5  SLOW_LEN\n")
    assert inside_model_grid(gc["model_root"], gc["trans"], 44.505, 6.745)
    assert not inside_model_grid(gc["model_root"], gc["trans"], 45.5585, 2.9121)


def test_build_nll_model_on_grid_reproduces_linear_model(tmp_path):
    """
    Modèle source régulier en (lon, lat, prof) et linéaire : l'interpolation
    dans la grille d'origine doit le restituer exactement aux nœuds NLL
    (ramenés en lon/lat par la projection AE inverse).
    """
    from nll_common import AzimuthalEquidistant
    from nll_pipeline import build_nll_model_on_grid

    lon0, lat0 = 6.745, 44.505
    lons = np.round(np.arange(6.5, 7.0001, 0.05), 4)
    lats = np.round(np.arange(44.3, 44.7001, 0.05), 4)
    deps = np.arange(-1.0, 4.0001, 1.0)
    vp = lambda lo, la, z: 5000 + 100 * (lo - 6.5) + 200 * (la - 44.3) + 50 * z
    rows = ["Longitude,Latitude,Depth,Vitesse onde P,Vitesse onde S"]
    for lo in lons:
        for la in lats:
            for z in deps:
                rows.append(f"{lo},{la},{z},{vp(lo, la, z)},{vp(lo, la, z) / 1.73}")
    csv = tmp_path / "tuile_elevated.csv"
    csv.write_text("\n".join(rows) + "\n")

    assert build_nll_model_on_grid(csv, tmp_path, lon0, lat0, 0.5, True)
    head = (tmp_path / "layer.P.mod.hdr").read_text().split()
    nx, ny, nz = int(head[0]), int(head[1]), int(head[2])
    x0, y0, z0 = float(head[3]), float(head[4]), float(head[5])
    assert x0 % 0.5 == 0 and y0 % 0.5 == 0 and z0 == -1.0   # nœuds aux multiples exacts du pas
    v = 0.5 / np.fromfile(tmp_path / "layer.P.mod.buf", dtype=np.float32).reshape(nx, ny, nz)
    i, j, k = nx // 2, ny // 3, 2
    lon, lat = AzimuthalEquidistant(lon0, lat0).inverse(x0 + 0.5 * i, y0 + 0.5 * j)
    assert v[i, j, k] == pytest.approx(vp(lon, lat, z0 + 0.5 * k) / 1000, rel=1e-5)


def test_azimuthal_equidistant_inverse_roundtrip():
    from nll_common import AzimuthalEquidistant

    ae = AzimuthalEquidistant(6.745, 44.505)
    lon = np.array([5.5, 6.745, 8.0]); lat = np.array([43.7, 44.505, 45.4])
    back_lon, back_lat = ae.inverse(*ae.forward(lon, lat))
    assert back_lon == pytest.approx(lon, abs=1e-9)
    assert back_lat == pytest.approx(lat, abs=1e-9)


def test_fix_vpvs(tmp_path):
    """Seuls les nœuds hors bornes changent ; Vp n'est jamais modifié."""
    from nll_pipeline import fix_vpvs

    vp = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    vs = np.array([2.9, 6.0, 3.6, 2.0], dtype=np.float32)   # Vp/Vs 1,72 · 0,83 · 1,39 · 2,5
    (0.5 / vp).astype(np.float32).tofile(tmp_path / "layer.P.mod.buf")
    (0.5 / vs).astype(np.float32).tofile(tmp_path / "layer.S.mod.buf")
    n, total = fix_vpvs(tmp_path, 1.5, 2.0, 1.73)
    assert (n, total) == (3, 4)
    new_vs = 0.5 / np.fromfile(tmp_path / "layer.S.mod.buf", dtype=np.float32)
    assert new_vs == pytest.approx([2.9, 5 / 1.73, 5 / 1.73, 5 / 1.73], rel=1e-5)
    assert 0.5 / np.fromfile(tmp_path / "layer.P.mod.buf", dtype=np.float32) == pytest.approx(vp)


def test_parse_phase_line():
    """Ligne réelle du bloc PHASE d'un .hyp NLLoc 7."""
    from nll_run import parse_phase_line

    line = ("JAUF   ?    ?    ? S      ? 20240601 0338   10.7272 GAU  5.00e-02 -1.00e+00 "
            "-1.00e+00 -1.00e+00 >    4.3940   -0.1787    0.8278   -0.0954   -9.0169   "
            "-1.6410   12.2572 200.54 200.5 112.5  9     0.0000")
    ph = parse_phase_line("ev1", line)
    assert (ph["station"], ph["phase"], ph["event_id"]) == ("JAUF", "S", "ev1")
    assert ph["time"] == "2024-06-01T03:38:10.727200"
    assert ph["error_s"] == pytest.approx(0.05)
    assert (ph["tt_pred_s"], ph["residual_s"], ph["weight"]) == (4.394, -0.1787, 0.8278)
    assert (ph["dist_km"], ph["azimuth"], ph["ray_dip"]) == (12.2572, 200.54, 112.5)

    # Phase arrivée juste après minuit : NLLoc écrit 2400 la veille
    line = ("SGSF   ?    ?    ? S      ? 20261021 2400    1.6900 GAU  2.00e-02 -1.00e+00 "
            "-1.00e+00 -1.00e+00 >   17.8128   -0.0378    1.5689   -3.2579  -59.8303   "
            "-0.9700   59.2993 183.55 183.5  99.3  9     0.0000")
    assert parse_phase_line("ev2", line)["time"] == "2026-10-22T00:00:01.690000"


def test_drop_stations(tmp_path):
    from nll_run import drop_stations, stations_used

    line = "{} ?    ?    ? {}      ? 20241101 2355  56.3800 GAU  0.020      -1.00e+00 -1.00e+00 -1.00e+00"
    obs = tmp_path / "p.obs"
    obs.write_text("#event ev1 3 picks\nPUBLIC_ID ev1\n" + "\n".join(
        [line.format("SURF", "P"), line.format("LOIN", "P"), line.format("LOIN", "S")]) + "\n\n")
    out, n = drop_stations(obs, ["LOIN"], tmp_path / "f.obs")
    assert n == 2
    assert stations_used(out) == {"P": {"SURF"}, "S": set()}
    assert "PUBLIC_ID ev1" in out.read_text()


def test_split_obs_and_missing_grids(tmp_path):
    from nll_run import split_obs, stations_without_grid

    obs = tmp_path / "p.obs"
    line = "{} ?    ?    ? {}      ? 20241101 2355  56.3800 GAU  0.020      -1.00e+00 -1.00e+00 -1.00e+00"
    blocks = [f"#event ev{i} 2 picks\nPUBLIC_ID ev{i}\n" + line.format("SURF", "P") + "\n"
              + line.format("ABSE", "S") for i in range(5)]
    obs.write_text("\n\n".join(blocks) + "\n\n")
    paths, n = split_obs(obs, 2, tmp_path / "parts")
    assert n == 5 and len(paths) == 2
    assert sum(p.read_text().count("PUBLIC_ID") for p in paths) == 5

    (tmp_path / "time").mkdir()
    (tmp_path / "time" / "layer.P.SURF.time.hdr").write_text("")
    missing = stations_without_grid(obs, tmp_path / "time" / "layer")
    assert missing == {"P": [], "S": ["ABSE"]}


def test_read_hyp(tmp_path):
    from nll_common import read_hyp

    hyp = tmp_path / "run.hyp"
    hyp.write_text(
        'NLLOC "loc.a" "LOCATED" "Location completed."\n'
        "PUBLIC_ID ev1\n"
        "GEOGRAPHIC  OT 2023 05 14  06 25 16.700850  Lat 44.618439 Long 6.677626 Depth -2.581676\n"
        "QUALITY  Pmax 1.6e+10 MFmin 10.3 MFmax 12.5 RMS 0.00237999 Nphs 5 Gap 199.6 Dist 8.8\n"
        "END_NLLOC\n\n"
        'NLLOC "loc.b" "REJECTED" "too few phases"\n'
        "GEOGRAPHIC  OT 2023 05 15  19 03 38.29  Lat 44.5 Long 6.8 Depth 2.2\n"
        "END_NLLOC\n"
    )
    events = read_hyp(hyp)
    assert [(e["event_id"], e["status"]) for e in events] == [("ev1", "LOCATED"), (None, "REJECTED")]
    assert events[0]["time"].isoformat() == "2023-05-14T06:25:16.700850"
    assert events[0]["depth"] == pytest.approx(-2.581676)
    assert events[0]["rms"] == pytest.approx(0.00237999)


def test_scrtdd_build_tables():
    from datetime import date, datetime

    from scrtdd_inputs import build_tables, select_events

    hyp = [
        {"event_id": "ev2", "status": "LOCATED", "time": datetime(2024, 12, 2, 1, 0, 0),
         "latitude": 44.5, "longitude": 6.8, "depth": 3.0},
        {"event_id": "ev1", "status": "LOCATED", "time": datetime(2024, 12, 1, 0, 45, 59, 153000),
         "latitude": 44.50265, "longitude": 6.84224, "depth": -0.5},
        {"event_id": None, "status": "LOCATED", "time": datetime(2024, 12, 1, 3, 0, 0),
         "latitude": 44.5, "longitude": 6.8, "depth": 3.0},
        {"event_id": "ev3", "status": "REJECTED", "time": datetime(2024, 12, 1, 4, 0, 0),
         "latitude": 44.5, "longitude": 6.8, "depth": 3.0},
        {"event_id": "ev4", "status": "LOCATED", "time": datetime(2025, 1, 1, 0, 0, 0),
         "latitude": 44.5, "longitude": 6.8, "depth": 3.0},
    ]
    events = select_events(hyp, date(2024, 12, 1), date(2024, 12, 31))
    assert [e["event_id"] for e in events] == ["ev1", "ev2"]

    def phase(sta, ph, lo=None, up=None):
        return {"station": sta, "phase": ph, "time": datetime(2024, 12, 1, 0, 46, 1, 500000),
                "network": "FR", "location": "00", "channel": "HHZ", "lower_s": lo, "upper_s": up}

    xml = {"ev1": {"origin_id": "o1", "magnitude": 0.42,
                   "phases": [phase("SURF", "Pg", 0.1, 0.1), phase("BHB", "Sg")]},
           "ev2": {"origin_id": "o2", "magnitude": None, "phases": [phase("BHB", "P")]}}
    ev_rows, ph_rows, el_rows, stations = build_tables(events, xml, excluded={"BHB"})
    # ev2 n'a plus de phase une fois BHB exclue : ignoré, les id restent contigus
    assert [r["id"] for r in ev_rows] == [1]
    assert ev_rows[0]["isotime"] == "2024-12-01T00:45:59.153000Z"
    assert ev_rows[0]["depth"] == "-0.5000" and ev_rows[0]["magnitude"] == "0.42"
    assert el_rows == [{"origin": "o1", "event": "ev1"}]
    assert [(r["eventId"], r["stationCode"], r["type"], r["lowerUncertainty"]) for r in ph_rows] \
        == [(1, "SURF", "Pg", "0.1")]
    assert stations == {("FR", "SURF", "00")}


def test_scrtdd_reference_positions(tmp_path):
    from datetime import datetime

    from scrtdd_inputs import reference_positions

    csv_path = tmp_path / "evenements.csv"
    csv_path.write_text(
        "event_id,origin_id,origin_time,method,latitude,longitude,depth\n"
        "ev1,o1,2026-10-22T05:28:01.5Z,LOCSAT,45.5858,3.6704,13.6\n"
        "ev2,o2,,LOCSAT,,,\n"
    )
    events = [
        {"event_id": "ev1", "status": "LOCATED", "time": datetime(2026, 10, 22, 5, 28, 1),
         "latitude": 45.59, "longitude": 3.66, "depth": 11.0},
        {"event_id": "ev2", "status": "LOCATED", "time": datetime(2026, 10, 22, 6, 0, 0),
         "latitude": 45.59, "longitude": 3.66, "depth": 11.0},
    ]
    out = reference_positions(events, csv_path)
    # ev2 n'a pas d'origine de référence complète : ignoré
    assert [e["event_id"] for e in out] == ["ev1"]
    assert (out[0]["latitude"], out[0]["longitude"], out[0]["depth"]) == (45.5858, 3.6704, 13.6)
    assert out[0]["time"].replace(tzinfo=None) == datetime(2026, 10, 22, 5, 28, 1, 500000)
