import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nll_common import (
    AzimuthalEquidistant,
    load_config,
    read_bounds_from_readme,
    read_center_from_readme,
)


def test_azimuthal_equidistant_center_is_origin():
    proj = AzimuthalEquidistant(lon0=6.75, lat0=44.5)
    x, y = proj.forward(6.75, 44.5)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_azimuthal_equidistant_one_degree_north():
    proj = AzimuthalEquidistant(lon0=0.0, lat0=0.0)
    x, y = proj.forward(0.0, 1.0)
    # 1° de latitude ≈ 111.19 km le long d'un méridien
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(111.19, abs=0.5)


def test_azimuthal_equidistant_is_symmetric_east_west():
    proj = AzimuthalEquidistant(lon0=5.0, lat0=45.0)
    x_east, y_east = proj.forward(6.0, 45.0)
    x_west, y_west = proj.forward(4.0, 45.0)
    assert x_east == pytest.approx(-x_west, abs=1e-6)
    assert y_east == pytest.approx(y_west, abs=1e-6)


def test_read_center_from_readme(tmp_path):
    readme = tmp_path / "README.txt"
    readme.write_text(
        "Centre de projection (Azimuthal Equidistant)\n"
        "  lon0 = 6.750000°\n"
        "  lat0 = 44.500000°\n",
        encoding="utf-8",
    )
    lon0, lat0 = read_center_from_readme(readme)
    assert lon0 == pytest.approx(6.75)
    assert lat0 == pytest.approx(44.5)


def test_read_center_from_readme_missing_raises(tmp_path):
    readme = tmp_path / "README.txt"
    readme.write_text("Rien à voir ici.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_center_from_readme(readme)


def test_read_bounds_from_readme(tmp_path):
    readme = tmp_path / "README.txt"
    readme.write_text(
        "Étendue géographique\n"
        "  Longitude : [6.0000°, 7.5000°]\n"
        "  Latitude  : [44.0000°, 45.0000°]\n",
        encoding="utf-8",
    )
    lon_min, lon_max, lat_min, lat_max = read_bounds_from_readme(readme)
    assert (lon_min, lon_max, lat_min, lat_max) == pytest.approx((6.0, 7.5, 44.0, 45.0))


def test_load_config_missing_file_gives_helpful_error(tmp_path):
    missing = tmp_path / "config.yaml"
    with pytest.raises(FileNotFoundError, match="config.example.yaml"):
        load_config(missing)


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths:\n  tuiles_dir: /tmp/tuiles\n", encoding="utf-8")
    config = load_config(config_path)
    assert config["paths"]["tuiles_dir"] == "/tmp/tuiles"
