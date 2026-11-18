"""
scripts/train_eval_models.py
Trains and evaluates at least 5 distinct machine learning models from different algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. K-Nearest Neighbors (Instance-based)
3. Decision Tree (Tree-based)
4. Random Forest (Ensemble)
5. HistGradientBoosting (Boosting)

Performs GroupKFold cross-validation on training trips, evaluates on student-test trips,
computes Observation-level Top-1 Accuracy, Macro F1, Log Loss, and Inference Time,
and exports comparison tables and trained weights.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import sys
from pathlib import Path
import time
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from src.data_loader import load_single_trip
from src.feature_engineering import extract_trip_features


FEATURE_COLS = [
    "dist_to_segment",
    "projection_ratio",
    "heading_diff",
    "segment_length",
    "dist_to_start",
    "dist_to_end",
    "vehicle_speed",
    "candidate_rank"
]


def load_partition_features(partition_dir: Path, max_trips: int = 12, k_candidates: int = 10, is_training: bool = True) -> pd.DataFrame:
    trip_folders = sorted([f for f in partition_dir.iterdir() if f.is_dir() and not f.name.startswith(".")])
    if max_trips is not None and max_trips > 0:
        trip_folders = trip_folders[:max_trips]

    dfs = []
    print(f"Extracting features from {len(trip_folders)} trips in {partition_dir.name}...", flush=True)
    for folder in trip_folders:
        trip = load_single_trip(folder)
        df_trip = extract_trip_features(trip, k_candidates=k_candidates, is_training=is_training)
        dfs.append(df_trip)

    return pd.concat(dfs, ignore_index=True)


def compute_observation_top1(df_eval: pd.DataFrame, y_probs: np.ndarray) -> float:
    """
    Computes observation-level Top-1 accuracy:
    For each GPS observation (trip_id, obs_id), select candidate with max predicted probability.
    Check if that candidate is the true target arc (is_match == 1).
    """
    df_temp = df_eval[["trip_id", "obs_id", "is_match"]].copy()
    df_temp["score"] = y_probs

    idx_top1 = df_temp.groupby(["trip_id", "obs_id"])["score"].idxmax()
    top1_matches = df_temp.loc[idx_top1, "is_match"].values
    return float(np.mean(top1_matches))


def main():
    train_dir = Path("Student folder/train")
    test_dir = Path("Student folder/student_test")
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    results_dir = artifacts_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Training Data
    train_df = load_partition_features(train_dir, max_trips=12, k_candidates=10, is_training=True)
    test_df = load_partition_features(test_dir, max_trips=10, k_candidates=10, is_training=True)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_match"].values
    groups = train_df["trip_id"].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_match"].values

    print(f"\nTraining set: {len(train_df)} candidate pairs ({len(np.unique(groups))} trips)", flush=True)
    print(f"Testing set: {len(test_df)} candidate pairs ({len(np.unique(test_df['trip_id']))} trips)", flush=True)

    # 2. Define 5 Distinct Models from different Algorithm Families
    models = {
        "Logistic Regression": {
            "family": "Linear and probabilistic",
            "config": "C=1.0, solver='lbfgs', max_iter=500",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
            ])
        },
        "Decision Tree": {
            "family": "Tree-based",
            "config": "max_depth=10, min_samples_split=20",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42))
            ])
        },
        "Random Forest": {
            "family": "Ensemble",
            "config": "n_estimators=30, max_depth=10, n_jobs=1",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1))
            ])
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "config": "max_iter=80, learning_rate=0.1, max_depth=6",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_depth=6, random_state=42))
            ])
        },
        "K-Nearest Neighbors": {
            "family": "Instance-based",
            "config": "n_neighbors=7, algorithm='kd_tree'",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, algorithm="kd_tree", n_jobs=1))
            ])
        }
    }

    # 3. Group-aware Cross-Validation on Training Trips (3-Fold for quick, rigorous CV)
    gkf = GroupKFold(n_splits=3)
    cv_scores = {}

    print("\n--- Running Group Cross-Validation on Training Trips ---", flush=True)
    # For KNN in CV, subsample for fast validation
    for name, m_info in models.items():
        pipe = m_info["pipeline"]
        fold_f1s = []

        sub_rate = 5 if name == "K-Nearest Neighbors" else 1
        sub_idx = np.arange(0, len(X_train), sub_rate)
        X_sub, y_sub, g_sub = X_train[sub_idx], y_train[sub_idx], groups[sub_idx]

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_sub, y_sub, groups=g_sub)):
            pipe.fit(X_sub[train_idx], y_sub[train_idx])
            val_preds = pipe.predict(X_sub[val_idx])
            fold_f1s.append(f1_score(y_sub[val_idx], val_preds, average="macro"))

        mean_cv_f1 = float(np.mean(fold_f1s))
        cv_scores[name] = mean_cv_f1
        print(f"[{name}] Group CV Macro F1: {mean_cv_f1:.4f}", flush=True)

    # 4. Train on Full Training Set, Evaluate on student_test, and Record Metrics
    comparison_results = []
    print("\n--- Training on Full Train Data and Evaluating on student_test ---", flush=True)

    for name, m_info in models.items():
        pipe = m_info["pipeline"]

        # Train & time
        t0 = time.time()
        # For KNN final fit, train on subsampled data for instant inference
        if name == "K-Nearest Neighbors":
            sub_idx = np.arange(0, len(X_train), 4)
            pipe.fit(X_train[sub_idx], y_train[sub_idx])
        else:
            pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        # Inference & time
        t1 = time.time()
        if hasattr(pipe, "predict_proba"):
            y_probs = pipe.predict_proba(X_test)[:, 1]
        else:
            decision = pipe.decision_function(X_test)
            y_probs = 1 / (1 + np.exp(-decision))
        inf_time = time.time() - t1

        y_preds = (y_probs >= 0.5).astype(int)

        # Metrics
        test_macro_f1 = f1_score(y_test, y_preds, average="macro")
        test_precision = precision_score(y_test, y_preds, zero_division=0)
        test_recall = recall_score(y_test, y_preds, zero_division=0)
        loss = log_loss(y_test, np.clip(y_probs, 1e-6, 1 - 1e-6))
        top1_acc = compute_observation_top1(test_df, y_probs)

        # Save model artifact
        model_save_path = models_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipe, model_save_path)

        res = {
            "Model": name,
            "Algorithm Family": m_info["family"],
            "Key Configuration": m_info["config"],
            "CV Macro F1": round(cv_scores[name], 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Student-Test Macro F1": round(test_macro_f1, 4),
            "Precision": round(test_precision, 4),
            "Recall": round(test_recall, 4),
            "Log Loss": round(loss, 4),
            "Inference Time (s)": round(inf_time, 4),
            "Training Time (s)": round(train_time, 4)
        }
        comparison_results.append(res)
        print(f"[{name}] Top-1 Acc: {top1_acc*100:.2f}% | Test Macro F1: {test_macro_f1:.4f} | Log Loss: {loss:.4f} | Inf Time: {inf_time:.3f}s", flush=True)

    # 5. Export Results
    results_df = pd.DataFrame(comparison_results)
    csv_path = results_dir / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll models saved to {models_dir}", flush=True)
    print(f"Comparison table exported to {csv_path}", flush=True)

    # Print Table
    print("\n=== FINAL MODEL COMPARISON TABLE FOR EVALUATION 1 ===", flush=True)
    cols_to_show = ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1", "Top-1 Accuracy", "Student-Test Macro F1", "Log Loss", "Inference Time (s)"]
    print(results_df[cols_to_show].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
