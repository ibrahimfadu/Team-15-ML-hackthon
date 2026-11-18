import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TRAIN_DIR, TEST_DIR, RESULTS_DIR, CANDIDATE_RADIUS_METERS
from src.data_loader import list_trip_ids, load_trip_data
from src.projection import get_projection_scale, project_coords
from src.alignment import align_track_to_route
from src.candidate_generator import CandidateGenerator
from src.feature_extractor import extract_features_for_trip, FEATURE_NAMES

def process_partition(partition_dir, trip_ids, is_train=True, max_trips=None):
    all_X = []
    all_y = []
    all_trip_tags = []
    all_point_indices = []
    all_arc_ids = []

    total_gps_points = 0
    total_missed_candidates = 0

    target_trips = trip_ids[:max_trips] if max_trips else trip_ids
    print(f"Processing {len(target_trips)} trips from {partition_dir.name}...")

    t0 = time.time()
    for idx, tid in enumerate(target_trips):
        track, nodes, arcs, route = load_trip_data(partition_dir, tid)
        ref_lon, ref_lat = nodes[0, 0], nodes[0, 1]
        kx, ky = get_projection_scale(ref_lat)
        nodes_m = project_coords(nodes, ref_lon, ref_lat, kx, ky)

        # Ground truth alignment
        matched_arcs, _ = align_track_to_route(track, nodes, arcs, route)

        # Candidate generation
        generator = CandidateGenerator(nodes, arcs, search_radius=CANDIDATE_RADIUS_METERS)
        candidates_per_point, gps_m = generator.get_candidates_for_track(track)

        # Measure Candidate Miss Rate
        if matched_arcs is not None:
            for pt_i in range(len(track)):
                total_gps_points += 1
                cands = candidates_per_point[pt_i]
                if matched_arcs[pt_i] not in cands:
                    total_missed_candidates += 1

        # Feature extraction
        X, y, meta = extract_features_for_trip(
            track, nodes_m, arcs, candidates_per_point, gps_m, matched_arcs=matched_arcs
        )

        all_X.append(X)
        if y is not None:
            all_y.append(y)
        for pt_idx, arc_id in meta:
            all_trip_tags.append(tid)
            all_point_indices.append(pt_idx)
            all_arc_ids.append(arc_id)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(target_trips):
            print(f"  Processed {idx + 1}/{len(target_trips)} trips ({time.time() - t0:.1f}s)")

    X_mat = np.vstack(all_X) if len(all_X) > 0 else np.empty((0, len(FEATURE_NAMES)))
    y_mat = np.concatenate(all_y) if len(all_y) > 0 else None

    miss_rate = (total_missed_candidates / total_gps_points * 100.0) if total_gps_points > 0 else 0.0
    print(f"-> Extraction complete! Total candidate rows: {len(X_mat):,}")
    print(f"-> Candidate Miss Rate: {miss_rate:.2f}% (Found in candidates: {100.0 - miss_rate:.2f}%)")

    meta_df = pd.DataFrame({
        "trip_id": all_trip_tags,
        "point_idx": all_point_indices,
        "arc_id": all_arc_ids
    })

    return X_mat, y_mat, meta_df, miss_rate

def main():
    print("=" * 60)
    print("PathSnap: Spatial Candidate Generation & Feature Extraction")
    print("=" * 60)

    train_ids = list_trip_ids(TRAIN_DIR)
    test_ids = list_trip_ids(TEST_DIR)

    # We extract 15 representative training trips (~80k candidate rows for ultra-fast multi-model training & CV)
    X_train, y_train, meta_train, train_miss = process_partition(
        TRAIN_DIR, train_ids, is_train=True, max_trips=15
    )

    # Process all 10 student-test trips
    X_test, y_test, meta_test, test_miss = process_partition(
        TEST_DIR, test_ids, is_train=False, max_trips=10
    )

    np.save(RESULTS_DIR / "X_train.npy", X_train)
    np.save(RESULTS_DIR / "y_train.npy", y_train)
    meta_train.to_pickle(RESULTS_DIR / "meta_train.pkl")

    np.save(RESULTS_DIR / "X_test.npy", X_test)
    np.save(RESULTS_DIR / "y_test.npy", y_test)
    meta_test.to_pickle(RESULTS_DIR / "meta_test.pkl")

    with open(RESULTS_DIR / "miss_rate.txt", "w") as f:
        f.write(f"Train Candidate Miss Rate: {train_miss:.2f}%\n")
        f.write(f"Student Test Candidate Miss Rate: {test_miss:.2f}%\n")

    print(f"\n[OK] Feature matrices saved successfully to {RESULTS_DIR}!")

if __name__ == "__main__":
    main()
