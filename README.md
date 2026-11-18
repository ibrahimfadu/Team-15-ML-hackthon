# PathSnap: Distributed Map-Matching & Route Reconstruction
**Team 15 | PES University - UE24CS352A Machine Learning Hackathon**

PathSnap is an end-to-end GPS trajectory map-matching and route reconstruction system. It combines localized spatial projection, spatial candidate generation via KD-Tree, machine learning scoring across 5 algorithm families, and HMM/Viterbi sequential decoding to match noisy GPS observations to physical road networks.

---

## 1. Project Architecture

```
Team-15-ML-hackthon/
├── src/
│   ├── config.py                 # Paths, constants, projection parameters
│   ├── projection.py             # Metric equirectangular projection & point-segment math
│   ├── data_loader.py            # .track, .nodes, .arcs, .route parsing
│   ├── candidate_generator.py    # cKDTree arc spatial index & radius search
│   ├── feature_extractor.py      # 11 geometric & kinematic features
│   └── alignment.py              # Monotonic GPS-to-route ground truth alignment
├── scripts/
│   ├── 01_eda_and_summary.py     # EDA, sampling analysis, trajectory distributions
│   ├── 02_extract_features.py    # Spatial candidate filtering & feature matrices
│   ├── 03_train_and_compare_models.py # Trains 5 models with GroupKFold CV
│   └── 04_plot_evaluations.py    # Generates F1, Top-1, LogLoss & CM plots
├── models_saved/                 # Serialized model artifacts (.joblib)
├── results/
│   ├── eda_summary.csv           # Detailed metrics per trip
│   ├── model_comparison_table.csv# Official Section 3.1 comparison table
│   └── figures/                  # Generated diagnostic visualisations
├── reports/                      # Technical report & documentation
├── requirements.txt
└── README.md
```

---

## 2. Methodology & Key Highlights

### A. Preprocessing & Local Metric Projection
- Angular GPS degrees ($^\circ\text{lon}, ^\circ\text{lat}$) are projected into local Cartesian coordinates (meters) centered at the trip's origin:
  $$x = R \cdot (\text{lon} - \text{lon}_0) \cos(\text{lat}_0) \cdot \frac{\pi}{180}, \quad y = R \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$
  This avoids severe longitudinal distortion ($1^\circ\text{lon} \approx 71.7\text{ km}$ vs $1^\circ\text{lat} \approx 111.2\text{ km}$).

### B. Spatial Candidate Generation
- A `scipy.spatial.cKDTree` indexes arc midpoints within a search radius of $45\text{ meters}$.
- Candidate miss rate is rigorously monitored and handled with fallback nearest arc association.

### C. Feature Engineering (11 Features)
For each GPS point and candidate road segment:
1. `perp_dist`: Orthogonal distance to segment (meters).
2. `proj_t`: Scalar projection factor along segment $[0, 1]$.
3. `clamp_penalty`: Clamp overshoot penalty $|t_{\text{raw}} - t_{\text{clamped}}|$.
4. `dist_u`: Distance to from-node.
5. `dist_v`: Distance to to-node.
6. `arc_len`: Segment length.
7. `heading_cos`: Heading alignment $\cos(\Delta \theta) = \frac{\vec{v}_{\text{GPS}} \cdot \vec{v}_{\text{arc}}}{\|\vec{v}_{\text{GPS}}\| \|\vec{v}_{\text{arc}}\|}$.
8. `angle_diff_rad`: Heading angular difference.
9. `gps_speed`: Vehicle instantaneous speed ($\Delta d / \Delta t$).
10. `dist_rank`: Distance rank among competing candidates.
11. `candidate_count`: Total candidates at timestep.

---

## 3. Model Comparison & Experimental Results (Phase 1)

All models were evaluated using **3-Fold GroupKFold Cross-Validation** (grouped by `trip_id` to prevent data leakage) and independently tested on the 10 Student-Test trips:

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | $C=1.0$, penalty='l2', max_iter=300 | 0.8244 | 61.65% | 61.83% | 0.8171 | 0.1389 | 0.012s |
| **Distance-Weighted KNN** | Instance-based | $k=9$, weights='distance' | 0.7857 | 59.38% | 59.77% | 0.8024 | 0.4919 | 10.473s |
| **Decision Tree** | Tree-based | max_depth=12, min_samples_leaf=15 | 0.8063 | 59.94% | 59.49% | 0.8149 | 0.3980 | 0.021s |
| **Random Forest** | Ensemble | $N=100$, max_depth=14 | 0.8222 | 62.43% | 61.64% | 0.8279 | 0.1390 | 0.172s |
| **HistGradientBoosting** | Boosting | $\eta=0.1$, max_iter=100, depth=8 | **0.8238** | **62.41%** | **62.53%** | **0.8288** | **0.1311** | 0.300s |

**Best Model Selected**: **HistGradientBoosting** achieved the highest test macro F1 (0.8288), top route continuity (62.53%), and the lowest log loss (0.1311), providing well-calibrated emission probabilities for the subsequent HMM/Viterbi decoder.

---

## 4. How to Reproduce

```bash
# 1. Run Exploratory Data Analysis & Quality Inspection
python scripts/01_eda_and_summary.py

# 2. Extract Spatial Candidates and 11 Feature Sets
python scripts/02_extract_features.py

# 3. Train & Compare all 5 Models with Group-CV
python scripts/03_train_and_compare_models.py

# 4. Generate Visualization Plots
python scripts/04_plot_evaluations.py
```
