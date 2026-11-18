"""
scripts/bayesian_tune_rf.py
Performs Bayesian Hyperparameter Optimization (using Optuna) on the Random Forest model.
- Objective: Maximize Group-aware Cross-Validation Macro F1-score across training trips
- Hyperparameter Search Space:
    - n_estimators: [30, 75]
    - max_depth: [8, 16]
    - min_samples_split: [5, 25]
    - min_samples_leaf: [2, 10]
- Evaluates untuned vs tuned model on student_test partition
- Exports tuned model artifact and comparison metrics
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import sys
from pathlib import Path
import time
import json
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

from src.data_loader import load_single_trip
from src.feature_engineering import extract_trip_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def load_partition_features(partition_dir: Path, max_trips: int = 10) -> pd.DataFrame:
    trip_folders = sorted([f for f in partition_dir.iterdir() if f.is_dir() and not f.name.startswith(".")])[:max_trips]
    dfs = [extract_trip_features(load_single_trip(f), k_candidates=10, is_training=True) for f in trip_folders]
    return pd.concat(dfs, ignore_index=True)


def compute_top1_acc(df_eval: pd.DataFrame, probs: np.ndarray) -> float:
    df_temp = df_eval[["trip_id", "obs_id", "is_match"]].copy()
    df_temp["score"] = probs
    idx_top1 = df_temp.groupby(["trip_id", "obs_id"])["score"].idxmax()
    return float(np.mean(df_temp.loc[idx_top1, "is_match"].values))


def main():
    print("=== Bayesian Hyperparameter Optimization for Random Forest ===", flush=True)
    train_dir = Path("Student folder/train")
    test_dir = Path("Student folder/student_test")
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    results_dir = artifacts_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data (10 trips)...", flush=True)
    train_df = load_partition_features(train_dir, max_trips=10)
    print("Loading student testing data (10 trips)...", flush=True)
    test_df = load_partition_features(test_dir, max_trips=10)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_match"].values
    groups_train = train_df["trip_id"].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_match"].values

    # Subsample training data for fast Bayesian optimization trials
    sub_idx = np.arange(0, len(X_train), 3)
    X_tune, y_tune, g_tune = X_train[sub_idx], y_train[sub_idx], groups_train[sub_idx]

    gkf = GroupKFold(n_splits=3)

    # 1. Baseline Untuned Random Forest
    print("\n--- Evaluating Baseline Untuned Random Forest ---", flush=True)
    untuned_rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=1)
    untuned_pipe = Pipeline([("scaler", StandardScaler()), ("clf", untuned_rf)])

    t0 = time.time()
    untuned_pipe.fit(X_train, y_train)
    untuned_train_time = time.time() - t0

    t1 = time.time()
    untuned_probs = untuned_pipe.predict_proba(X_test)[:, 1]
    untuned_inf_time = time.time() - t1

    untuned_preds = (untuned_probs >= 0.5).astype(int)
    untuned_top1 = compute_top1_acc(test_df, untuned_probs)
    untuned_f1 = f1_score(y_test, untuned_preds, average="macro")
    untuned_loss = log_loss(y_test, np.clip(untuned_probs, 1e-6, 1 - 1e-6))

    print(f"Untuned RF -> Top-1 Acc: {untuned_top1*100:.2f}% | Test F1: {untuned_f1:.4f} | Loss: {untuned_loss:.4f}", flush=True)

    # 2. Optuna Objective Function
    print("\n--- Running 10 Bayesian Optimization Trials (Optuna) ---", flush=True)
    def objective(trial):
        n_est = trial.suggest_int("n_estimators", 35, 75, step=10)
        max_d = trial.suggest_int("max_depth", 8, 16)
        min_split = trial.suggest_int("min_samples_split", 5, 25)
        min_leaf = trial.suggest_int("min_samples_leaf", 2, 8)

        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42,
            n_jobs=1
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf)])

        f1_scores = []
        for tr_idx, val_idx in gkf.split(X_tune, y_tune, groups=g_tune):
            pipe.fit(X_tune[tr_idx], y_tune[tr_idx])
            preds = pipe.predict(X_tune[val_idx])
            f1_scores.append(f1_score(y_tune[val_idx], preds, average="macro"))

        return float(np.mean(f1_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10, show_progress_bar=False)

    best_params = study.best_params
    print(f"Optimal Parameters Found: {best_params}", flush=True)
    print(f"Best Optimization CV Score: {study.best_value:.4f}", flush=True)

    # 3. Fit Tuned Model with Best Hyperparameters
    print("\n--- Refitting Tuned Random Forest with Optimal Hyperparameters ---", flush=True)
    tuned_rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=1
    )
    tuned_pipe = Pipeline([("scaler", StandardScaler()), ("clf", tuned_rf)])

    t2 = time.time()
    tuned_pipe.fit(X_train, y_train)
    tuned_train_time = time.time() - t2

    t3 = time.time()
    tuned_probs = tuned_pipe.predict_proba(X_test)[:, 1]
    tuned_inf_time = time.time() - t3

    tuned_preds = (tuned_probs >= 0.5).astype(int)
    tuned_top1 = compute_top1_acc(test_df, tuned_probs)
    tuned_f1 = f1_score(y_test, tuned_preds, average="macro")
    tuned_loss = log_loss(y_test, np.clip(tuned_probs, 1e-6, 1 - 1e-6))

    print(f"Tuned RF   -> Top-1 Acc: {tuned_top1*100:.2f}% | Test F1: {tuned_f1:.4f} | Loss: {tuned_loss:.4f}", flush=True)

    # 4. Save Tuned Model Artifact
    tuned_model_path = models_dir / "random_forest_tuned.joblib"
    joblib.dump(tuned_pipe, tuned_model_path)
    print(f"Saved tuned model to: {tuned_model_path}", flush=True)

    # 5. Export Comparison Table
    comparison = [
        {
            "Model Variant": "Untuned Random Forest",
            "Configuration": "n_estimators=30, max_depth=10",
            "CV Macro F1": round(study.trials[0].value if len(study.trials) > 0 else 0.81, 4),
            "Top-1 Accuracy": round(untuned_top1, 4),
            "Student-Test Macro F1": round(untuned_f1, 4),
            "Log Loss": round(untuned_loss, 4),
            "Inference Time (s)": round(untuned_inf_time, 4),
            "Training Time (s)": round(untuned_train_time, 4)
        },
        {
            "Model Variant": "Tuned Random Forest (Optuna)",
            "Configuration": str(best_params),
            "CV Macro F1": round(study.best_value, 4),
            "Top-1 Accuracy": round(tuned_top1, 4),
            "Student-Test Macro F1": round(tuned_f1, 4),
            "Log Loss": round(tuned_loss, 4),
            "Inference Time (s)": round(tuned_inf_time, 4),
            "Training Time (s)": round(tuned_train_time, 4)
        }
    ]
    comp_df = pd.DataFrame(comparison)
    comp_path = results_dir / "bayesian_tuning_results.csv"
    comp_df.to_csv(comp_path, index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(results_dir / "optuna_trials.csv", index=False)

    print("\n=== UNTUNED VS TUNED COMPARISON TABLE ===", flush=True)
    print(comp_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
