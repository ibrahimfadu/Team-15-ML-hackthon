# Team 15 — Section 5-D

| Member | SRN | Primary ownership | GitHub area |
|---|---|---|---|
| Yash Raghavendra A | PES2UG24CS913 | Dataset audit, preprocessing, spatial candidate generation, feature engineering, EDA/Folium, data/API integration | `src/pathsnap/data*`, `src/pathsnap/features*`, `scripts/`, `reports/eda/`, API data integration |
| Deeraj S | PES2UG25CS806 | At least five model experiments, group-aware CV, evaluation metrics, model comparison, Bayesian optimisation, experiment logs/plots | `src/pathsnap/models/`, `experiments/`, `reports/results/` |
| Ibrahim Khaleel | PES2UG25CS809 | HMM/Viterbi, transition/emission design, route continuity analysis, FastAPI/UI integration testing, Streamlit evaluator interface, two-laptop deployment | `src/pathsnap/decoder*`, `backend/`, `ui/`, `tests/integration/`, `docs/` |

## Equal-contribution safeguards

- Each member owns and implements their assigned modules; each reviews one other member's work.
- Use separate branches: `feature/yash-data`, `feature/deeraj-models`, `feature/ibrahim-decoder-ui`.
- Each person makes commits under their own GitHub identity after doing the work. Do not fabricate authorship or commit activity.
- Merge through pull requests into `main`; the team should keep the repository history and experiment logs.
- Keep a short record of each member's tasks, files changed, tests run and review contributions for the viva.
