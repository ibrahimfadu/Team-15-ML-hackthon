"""Create a reproducible inventory for train/ and student_test/ trip folders."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pathsnap.io import read_trip, validate_trip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Folder containing train/ and student_test/ folders")
    parser.add_argument("--output", default="reports/eda/trip_inventory.csv")
    args = parser.parse_args()
    data_root = Path(args.dataset_root)
    if not data_root.exists():
        raise SystemExit(f"Dataset root does not exist: {data_root}")
    records = []
    for partition in ("train", "student_test"):
        part_dir = data_root / partition
        if not part_dir.exists():
            print(f"WARNING: partition folder missing: {part_dir}")
            continue
        for trip_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            try:
                trip = read_trip(trip_dir, require_route=(partition != "evaluator_test"))
                row = validate_trip(trip)
                row["partition"] = partition
                records.append(row)
            except Exception as exc:
                print(f"ERROR {partition}/{trip_dir.name}: {exc}")
    if not records:
        raise SystemExit("No valid trips found. Check the dataset root and folder names.")
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    cols = ["partition"] + [c for c in df.columns if c != "partition"]
    df[cols].to_csv(out, index=False)
    print(df.groupby("partition").agg(trips=("trip_id", "count"), gps_observations=("gps_observations", "sum"),
                                      nodes=("nodes", "sum"), arcs=("arcs", "sum")).to_string())
    print(f"\nSaved inventory: {out}")


if __name__ == "__main__":
    main()
