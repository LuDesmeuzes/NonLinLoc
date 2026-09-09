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
