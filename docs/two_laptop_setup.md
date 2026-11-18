# Two-laptop demo setup (Windows CMD)

## Laptop 1 — model + API

1. Connect to the same Wi-Fi/hotspot as Laptop 2.
2. In CMD, open the repository folder and activate the project's virtual environment.
3. Install `requirements-laptop1.txt`.
4. Put the organisers' **reserved evaluator partition** in `data/evaluator_test/` (one folder per trip; each folder contains `.track`, `.nodes`, `.arcs`, and the server-only `.route`). Do not substitute `student_test/` for this partition.
5. Set the evaluator folder if it is elsewhere:

   `set PATHSNAP_EVALUATOR_DIR=C:\Path\To\evaluator_test`

6. Start the API so another laptop can reach it:

   `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`

7. On Laptop 1, open `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`.
8. Run `ipconfig` and note the **IPv4 Address** of the active Wi-Fi adapter. Share only that LAN address with Laptop 2.

## Laptop 2 — Streamlit evaluator UI

1. Connect to the same Wi-Fi/hotspot.
2. Install Python and `requirements-laptop2.txt` in a virtual environment.
3. Run:

   `python -m streamlit run ui/app.py --server.port 8501`

4. In the UI sidebar, set the API URL to `http://LAPTOP1_IPV4:8000`, for example `http://192.168.137.1:8000`.
5. Click **Check API connection**. Then select a trip and run inference.

## Windows firewall

If Laptop 2 cannot connect, open **Command Prompt as Administrator** on Laptop 1 and add a rule on the trusted demo network:

`netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000`

From Laptop 2 PowerShell, test:

`Test-NetConnection LAPTOP1_IPV4 -Port 8000`

If the campus Wi-Fi blocks device-to-device traffic, use a phone hotspot or one laptop's hotspot, with both laptops on that same network. Do not disable the firewall globally.

## Final demo checklist

- API process stays open on Laptop 1; do not close its terminal.
- Laptop 1 is plugged in and will not sleep.
- `/health` reports the expected evaluator trip count.
- `/evaluation/trips` returns ten IDs and no labels.
- UI runs on Laptop 2 and uses Laptop 1's LAN IPv4 address, not `127.0.0.1`.
- Select and test more than one reserved trip before the evaluator arrives.
- Keep the FastAPI terminal visible; save screenshots of API request logs and the working UI.
- Ground-truth `.route` files remain on Laptop 1; never send them to the UI.
