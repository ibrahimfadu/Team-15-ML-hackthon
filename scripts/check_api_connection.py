import argparse
import requests

p = argparse.ArgumentParser()
p.add_argument("api_url", help="Example: http://192.168.137.1:8000")
args = p.parse_args()
try:
    r = requests.get(args.api_url.rstrip("/") + "/health", timeout=5)
    print("HTTP", r.status_code)
    print(r.text)
    r.raise_for_status()
except requests.RequestException as exc:
    raise SystemExit(f"API connection failed: {exc}")
