from pathlib import Path
import numpy as np
from src.pathsnap.io import read_trip, validate_trip


def test_read_trip_and_validate(tmp_path: Path):
    tid = "00000001"
    d = tmp_path / tid
    d.mkdir()
    (d / f"{tid}.track").write_text("10.0 20.0 0\n10.1 20.1 1\n")
    (d / f"{tid}.nodes").write_text("10.0 20.0\n10.1 20.1\n")
    (d / f"{tid}.arcs").write_text("0 1\n")
    (d / f"{tid}.route").write_text("0\n")
    trip = read_trip(d, require_route=True)
    stats = validate_trip(trip)
    assert len(trip.track) == 2
    assert np.array_equal(trip.route, np.array([0]))
    assert stats["invalid_arc_endpoints"] == 0
    assert stats["missing_route_ids"] == 0
