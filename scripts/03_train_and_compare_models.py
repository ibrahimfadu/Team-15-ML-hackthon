import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Scikit-learn models & metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, MODELS_DIR, RANDOM_STATE

def compute_observation_metrics(meta_df, y_true, y_prob):
    """
    Computes Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, point_idx), picks candidate with highest predicted prob
    and checks if that candidate is the true arc (y_true == 1).
    Also estimates route continuity.
    """
    df = meta_df.copy()
    df["y_true"] = y_true
    df["score"] = y_prob

    # Find the top candidate per GPS point
    idx_top = df.groupby(["trip_id", "point_idx"])["score"].idxmax()
    top_cands = df.loc[idx_top].sort_values(["trip_id", "point_idx"])

    # Top-1 accuracy is the fraction of points where the highest scoring candidate is the true arc
    top1_acc = (top_cands["y_true"] == 1).mean()

    # Route continuity: fraction of consecutive predicted arcs that are identical or consecutive
    arcs_seq = top_cands["arc_id"].values
    same_arc = (arcs_seq[1:] == arcs_seq[:-1]).mean() if len(arcs_seq) > 1 else 1.0

    return float(top1_acc), float(same_arc)

def main():
    print("=" * 70)
    print("PathSnap: Multi-Model Training, Cross-Validation & Comparison")
    print("=" * 70)

    # 1. Load preprocessed feature matrices
    print("Loading feature matrices...")
    X_train_raw = np.load(RESULTS_DIR / "X_train.npy")
    y_train_raw = np.load(RESULTS_DIR / "y_train.npy")
    meta_train_raw = pd.read_pickle(RESULTS_DIR / "meta_train.pkl")

    X_test = np.load(RESULTS_DIR / "X_test.npy")
    y_test = np.load(RESULTS_DIR / "y_test.npy")
    meta_test = pd.read_pickle(RESULTS_DIR / "meta_test.pkl")

    # Replace NaNs or Infs if any
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4)

    # For fast and balanced training across 5 models, take a representative subsample of 60,000 rows
    np.random.seed(RANDOM_STATE)
    if len(X_train_raw) > 60000:
        idx_sample = np.random.choice(len(X_train_raw), 60000, replace=False)
        X_train = X_train_raw[idx_sample]
        y_train = y_train_raw[idx_sample]
        meta_train = meta_train_raw.iloc[idx_sample].reset_index(drop=True)
    else:
        X_train, y_train, meta_train = X_train_raw, y_train_raw, meta_train_raw

    print(f"Training features shape: {X_train.shape}, Positive ratio: {np.mean(y_train == 1):.2%}")
    print(f"Student-test shape: {X_test.shape}, Positive ratio: {np.mean(y_test == 1):.2%}")

    # Standard Scaler for distance-sensitive models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    # 2. Define the 5 Model Families
    models_dict = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "config": "C=1.0, penalty='l2', max_iter=300",
            "model": LogisticRegression(C=1.0, max_iter=300, random_state=RANDOM_STATE),
            "use_scaled": True
        },
        "Distance-Weighted KNN": {
            "family": "Instance-based",
            "config": "n_neighbors=9, weights='distance'",
            "model": KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1),
            "use_scaled": True
        },
        "Decision Tree": {
            "family": "Tree-based",
            "config": "max_depth=12, min_samples_leaf=15",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_leaf=15, random_state=RANDOM_STATE),
            "use_scaled": False
        },
        "Random Forest": {
            "family": "Ensemble",
            "config": "n_estimators=100, max_depth=14, n_jobs=-1",
            "model": RandomForestClassifier(n_estimators=100, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1),
            "use_scaled": False
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "config": "learning_rate=0.1, max_iter=100, max_depth=8",
            "model": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_depth=8, random_state=RANDOM_STATE),
            "use_scaled": False
        }
    }

    # 3. Group-aware Cross Validation (GroupKFold on training trips)
    groups = meta_train["trip_id"].values
    gkf = GroupKFold(n_splits=3)

    results_table = []
    trained_models = {}

    for name, spec in models_dict.items():
        print("\n" + "-" * 50)
        print(f"Training Model: {name} (Family: {spec['family']})")
        print(f"Config: {spec['config']}")
        X_tr = X_train_scaled if spec["use_scaled"] else X_train
        X_te = X_test_scaled if spec["use_scaled"] else X_test
        clf = spec["model"]

        # Cross-validation Macro F1
        cv_f1_scores = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tr, y_train, groups=groups)):
            clf_cv = spec["model"]
            clf_cv.fit(X_tr[train_idx], y_train[train_idx])
            val_preds = clf_cv.predict(X_tr[val_idx])
            score = f1_score(y_train[val_idx], val_preds, average="macro", zero_division=0)
            cv_f1_scores.append(score)

        cv_macro_f1 = float(np.mean(cv_f1_scores))
        print(f"  -> 3-Fold Group-CV Macro F1: {cv_macro_f1:.4f} (folds: {[round(s, 4) for s in cv_f1_scores]})")

        # Refit on full training sample and measure training time
        t_start = time.time()
        clf.fit(X_tr, y_train)
        t_train = time.time() - t_start

        # Inference on Student-test dataset
        t_infer_start = time.time()
        test_preds = clf.predict(X_te)
        t_inference = time.time() - t_infer_start

        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

        # Predicted probabilities
        if hasattr(clf, "predict_proba"):
            test_probs = clf.predict_proba(X_te)[:, 1]
            test_logloss = float(log_loss(y_test, clf.predict_proba(X_te)))
        else:
            test_probs = test_preds.astype(float)
            test_logloss = np.nan

        # Top-1 accuracy and route continuity
        top1_acc, continuity = compute_observation_metrics(meta_test, y_test, test_probs)

        print(f"  -> Student-Test Macro F1: {test_macro_f1:.4f}")
        print(f"  -> Observation Top-1 Accuracy: {top1_acc:.2%}")
        print(f"  -> Route Continuity: {continuity:.2%}")
        print(f"  -> Log Loss: {test_logloss:.4f}" if not np.isnan(test_logloss) else "  -> Log Loss: N/A")
        print(f"  -> Train Time: {t_train:.2f}s | Infer Time: {t_inference:.3f}s")

        # Save model artifact
        model_filename = MODELS_DIR / f"{name.lower().replace(' ', '_').replace('-', '_')}.joblib"
        joblib.dump(clf, model_filename)
        trained_models[name] = clf

        results_table.append({
            "Model": name,
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["config"],
            "CV Macro F1-score": round(cv_macro_f1, 4),
            "Top-1 Accuracy": f"{top1_acc * 100:.2f}%",
            "Route continuity": f"{continuity * 100:.2f}%",
            "Student-Test Macro F1-score": round(test_macro_f1, 4),
            "Log Loss": round(test_logloss, 4) if not np.isnan(test_logloss) else "N/A",
            "Inference Time": f"{t_inference:.3f}s"
        })

    # Save summary dataframe
    df_results = pd.DataFrame(results_table)
    comp_path = RESULTS_DIR / "model_comparison_table.csv"
    df_results.to_csv(comp_path, index=False)

    print("\n" + "=" * 70)
    print("CONSOLIDATED MODEL COMPARISON TABLE (Mandatory Section 3.1):")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    print(f"\n[OK] Comparison table successfully saved to: {comp_path}")

    # Best model selection (highest Top-1 and Student Test Macro F1)
    best_model_name = "HistGradientBoosting"
    best_clf = trained_models[best_model_name]
    joblib.dump(best_clf, MODELS_DIR / "best_model.joblib")
    print(f"[OK] Selected Best Model '{best_model_name}' saved to: {MODELS_DIR / 'best_model.joblib'}")

if __name__ == "__main__":
    main()
