# PathSnap — Team 15, Section 5-D

**Team members**
- Yash Raghavendra A — PES2UG24CS913
- Deeraj S — PES2UG25CS806
- Ibrahim Khaleel — PES2UG25CS809

A Python-only starter workspace for the PathSnap GPS trajectory map-matching hackathon. **Node.js, Next.js and React are not required.** The evaluator UI uses Streamlit; the inference service uses FastAPI/Uvicorn.

## Supplied data partitions

The uploaded student package contains `train/` (80 trips) and `student_test/` (10 trips). The reserved evaluator partition is not present in that package. Use training trips for fitting and group-aware cross-validation; use student-testing trips for development-stage comparisons. Do not pretend that student-testing trips are the reserved evaluator trips.

## Setup

See [`docs/two_laptop_setup.md`](docs/two_laptop_setup.md). Laptop 1 installs `requirements-laptop1.txt`; Laptop 2 installs `requirements-laptop2.txt`.

## Dataset inventory

Run:

```cmd
python scripts/check_dataset.py "C:\path\to\Student folder"
```

The input directory must contain `train/` and `student_test/` subfolders. The script writes `reports/eda/trip_inventory.csv`.

## Run the API and UI

Laptop 1:

```cmd
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Laptop 2:

```cmd
python -m streamlit run ui/app.py --server.port 8501
```

Set the UI's **Laptop 1 API URL** to `http://<Laptop-1-Wi-Fi-IPv4>:8000`.

## Current implementation status

This is a setup/integration starter, not a completed scoring submission. It includes the data readers, inventory script, API/UI contract and two-laptop setup. The candidate generator, five trained models, Bayesian optimisation, final HMM/Viterbi decoder, and real prediction pipeline still need to be implemented and tested. The API deliberately returns `501` for prediction until the real pipeline is connected; it never fabricates a route or metric.

## Academic integrity

All model weights, configurations, data-derived artifacts and reported metrics must come from this team's actual independent experiments. Keep the evaluator `.route` files server-side. Never hardcode predictions.
