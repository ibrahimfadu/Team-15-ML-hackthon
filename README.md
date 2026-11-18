# PathSnap: Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A - Machine Learning  
**Department of Computer Science and Engineering, PES University**

## 1. Project Overview
PathSnap is an end-to-end machine learning and sequential inference system designed to match noisy GPS trajectories against an underlying road network to accurately reconstruct vehicle routes. It combines:
1. Spatial candidate generation using spatial indices (KD-Tree on projected metric coordinates).
2. Supervised machine learning classifiers to predict candidate road segment likelihoods.
3. Hidden Markov Models (HMM) and Viterbi decoding to enforce road network topology and sequential path coherence.
4. Distributed deployment using a FastAPI inference server and an evaluator UI.

---

## 2. Repository Structure
```
├── artifacts/
│   ├── maps/                   # Generated Folium HTML interactive maps
│   ├── models/                 # Saved trained model artifacts (.joblib)
│   └── results/                # Comparison tables, CSVs, and visualization plots
├── notebooks/
│   └── 01_PathSnap_Eval1_Colab.ipynb   # Complete self-contained evaluation notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Parsers for .track, .nodes, .arcs, .route
│   ├── geometry.py             # Metric projection, bearings, point-to-segment distance
│   ├── feature_engineering.py  # Spatial and kinematic feature extraction
│   └── hmm_decoder.py          # HMM & Viterbi sequential decoder
├── scripts/
│   ├── run_eda.py              # Exploratory data analysis & Folium generation
│   └── train_eval_models.py    # Multi-model training and evaluation script
├── README.md
└── requirements.txt
```

---

## 3. Quickstart & Installation

```bash
git clone <YOUR_REPO_URL>
cd hackthon
pip install -r requirements.txt
```

### Running on Google Colab
1. Upload the repository or `notebooks/01_PathSnap_Eval1_Colab.ipynb` to Google Colab.
2. Upload the `Student folder` (containing `train` and `student_test`).
3. Run all cells to perform feature extraction, cross-validation, and training of 5 distinct models.

### Running EDA Locally
```bash
python scripts/run_eda.py
```
This generates the interactive Folium map at `artifacts/maps/sample_trip_eda.html`.
