"""PathSnap API with server-side per-trip route evaluation."""
from __future__ import annotations

import os
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.pipeline import predict_one_trip

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = Path(
    os.getenv("PATHSNAP_EVALUATOR_DIR", PROJECT_ROOT / "data" / "evaluator_test")
)

app = FastAPI(title="PathSnap Team 15 API", version="0.2.0")


def _available_trip_dirs() -> dict[str, Path]:
    if not EVALUATOR_DIR.exists():
        return {}
    found: dict[str, Path] = {}
    for folder in EVALUATOR_DIR.iterdir():
        if folder.is_dir():
            tid = folder.name
            if all((folder / f"{tid}{ext}").exists()
                   for ext in (".track", ".nodes", ".arcs")):
                found[tid] = folder
    return found


def _read_reference_route(trip_dir: Path) -> list[int]:
    """Read labels only on the server, after inference; never return the IDs."""
    route_path = trip_dir / f"{trip_dir.name}.route"
    if not route_path.exists():
        raise FileNotFoundError(f"Reference route missing for {trip_dir.name}")
    values = route_path.read_text(encoding="utf-8").split()
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"Invalid arc ID in {route_path.name}") from exc


def _score_route(predicted: list[int], reference: list[int]) -> dict:
    """Order-aware route similarity plus multiset segment precision/recall/F1."""
    if not predicted or not reference:
        raise ValueError("Predicted route or reference route is empty.")

    predicted = [int(x) for x in predicted]
    reference = [int(x) for x in reference]

    pred_counts = Counter(predicted)
    ref_counts = Counter(reference)
    matched = sum((pred_counts & ref_counts).values())

    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    sequence_similarity = SequenceMatcher(
        None, predicted, reference, autojunk=False
    ).ratio()

    return {
        "route_accuracy_percent": round(sequence_similarity * 100, 2),
        "segment_precision_percent": round(precision * 100, 2),
        "segment_recall_percent": round(recall * 100, 2),
        "segment_f1_percent": round(f1 * 100, 2),
        "predicted_segment_count": len(predicted),
        "reference_segment_count": len(reference),
        "matched_segment_count": int(matched),
        "metric_definition": (
            "Route accuracy is order-aware sequence similarity; "
            "precision/recall/F1 use multiset overlap of arc IDs."
        ),
    }


@app.get("/health")
def health():
    trips = _available_trip_dirs()
    return {
        "status": "ok",
        "service": "pathsnap-api",
        "evaluator_data_dir": str(EVALUATOR_DIR),
        "evaluator_trip_count": len(trips),
        "prediction_pipeline_connected": True,
        "per_trip_scoring": True,
    }


@app.get("/evaluation/trips")
def evaluation_trips():
    """List IDs only. Never expose route labels or reference files to the UI."""
    trips = sorted(_available_trip_dirs().keys())
    if not trips:
        raise HTTPException(
            status_code=503,
            detail=f"No evaluator trips found under {EVALUATOR_DIR}.",
        )
    return {"trip_ids": trips, "count": len(trips)}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    trip_dirs = _available_trip_dirs()
    if trip_id not in trip_dirs:
        raise HTTPException(status_code=404, detail="Unknown evaluator trip ID")

    trip_dir = trip_dirs[trip_id]
    try:
        # Inference runs first. The prediction pipeline does not read .route.
        result = predict_one_trip(trip_dir)

        # Score only after inference, on the server. Reference IDs never leave here.
        predicted_route = result.get("reconstructed_route_arc_ids")
        if not predicted_route:
            predicted_route = result.get("predicted_arc_ids")
        if not predicted_route:
            raise ValueError("Prediction response contains no predicted arc IDs.")

        reference_route = _read_reference_route(trip_dir)
        result["per_trip_evaluation"] = _score_route(
            predicted_route, reference_route
        )
        result["per_trip_evaluation"]["trip_id"] = trip_id
        result["ground_truth_used"] = False
        result["evaluation_note"] = (
            "Reference route was used only for server-side scoring after inference; "
            "reference arc IDs are not included in this response."
        )

        forbidden = {
            "ground_truth", "reference_route", "true_arc_ids",
            "route_labels", "reference_arc_ids",
        }
        if forbidden.intersection(result):
            raise ValueError("Prediction response attempted to expose ground-truth IDs.")
        return result

    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction/evaluation failed: {exc}"
        ) from exc
