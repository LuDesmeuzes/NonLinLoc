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
    from seiscomp_xml_to_obs import obs_line, parse_event_xml

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
    info, phases = parse_event_xml(xml)
    assert info["method"] == "LOCSAT"
    assert [p["station"] for p in phases] == ["CREF", "JAUF"]
    assert phases[0]["error_s"] is None
    assert phases[1]["error_s"] == pytest.approx(0.2)

    line = obs_line(phases[0], 0.1).split()
    assert line[0] == "CREF" and line[4] == "Pg"
    assert line[6:9] == ["20230514", "0625", "18.5603"]
    assert line[9:11] == ["GAU", "0.100"]
