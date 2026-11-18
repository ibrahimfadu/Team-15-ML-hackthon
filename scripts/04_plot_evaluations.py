import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS_DIR, FIGURES_DIR, MODELS_DIR

def main():
    print("Generating Model Evaluation Visualizations...")

    table_path = RESULTS_DIR / "model_comparison_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing comparison table: {table_path}")

    df = pd.read_csv(table_path)
    # Parse percentage strings
    df["Top1_Numeric"] = df["Top-1 Accuracy"].str.rstrip('%').astype(float)
    df["Continuity_Numeric"] = df["Route continuity"].str.rstrip('%').astype(float)
    df["LogLoss_Numeric"] = pd.to_numeric(df["Log Loss"], errors='coerce')

    models = df["Model"].tolist()

    # 1. Comparison of CV Macro F1 vs Student-Test Macro F1
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width/2, df["CV Macro F1-score"], width, label='3-Fold Group CV Macro F1', color='#4A90E2', edgecolor='black')
    plt.bar(x + width/2, df["Student-Test Macro F1-score"], width, label='Student-Test Macro F1', color='#50E3C2', edgecolor='black')

    plt.ylabel('Macro F1-Score')
    plt.title('Macro F1-Score Comparison Across Algorithm Families')
    plt.xticks(x, models, rotation=15, ha='right')
    plt.ylim(0.7, 0.9)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    f1_fig = FIGURES_DIR / "f1_score_comparison.png"
    plt.savefig(f1_fig, dpi=200)
    plt.close()
    print(f"[OK] F1 Comparison saved: {f1_fig}")

    # 2. Top-1 Accuracy & Route Continuity Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df["Top1_Numeric"], width, label='Observation Top-1 Accuracy (%)', color='#F5A623', edgecolor='black')
    plt.bar(x + width/2, df["Continuity_Numeric"], width, label='Route Continuity (%)', color='#9013FE', edgecolor='black')

    plt.ylabel('Percentage (%)')
    plt.title('Observation Top-1 Accuracy & Route Continuity')
    plt.xticks(x, models, rotation=15, ha='right')
    plt.ylim(50, 70)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    top1_fig = FIGURES_DIR / "top1_and_continuity_comparison.png"
    plt.savefig(top1_fig, dpi=200)
    plt.close()
    print(f"[OK] Top-1 and Continuity plot saved: {top1_fig}")

    # 3. Log Loss Comparison
    plt.figure(figsize=(9, 5))
    valid_loss_df = df.dropna(subset=["LogLoss_Numeric"])
    plt.bar(valid_loss_df["Model"], valid_loss_df["LogLoss_Numeric"], color='#E94E77', edgecolor='black', width=0.5)
    plt.ylabel('Cross-Entropy Log Loss (Lower is Better)')
    plt.title('Probability Calibration: Log Loss Comparison')
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    loss_fig = FIGURES_DIR / "log_loss_comparison.png"
    plt.savefig(loss_fig, dpi=200)
    plt.close()
    print(f"[OK] Log Loss plot saved: {loss_fig}")

    # 4. Confusion Matrix for Best Model (HistGradientBoosting)
    X_test = np.load(RESULTS_DIR / "X_test.npy")
    y_test = np.load(RESULTS_DIR / "y_test.npy")
    best_clf = joblib.load(MODELS_DIR / "best_model.joblib")
    y_pred = best_clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative Arc (0)", "True Arc (1)"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix: HistGradientBoosting (Student Test)")
    plt.tight_layout()
    cm_fig = FIGURES_DIR / "confusion_matrix_best_model.png"
    plt.savefig(cm_fig, dpi=200)
    plt.close()
    print(f"[OK] Confusion matrix saved: {cm_fig}")

if __name__ == "__main__":
    main()
