import os
import sys
import time

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TIMEOUT = 90
RETRY_CODES = (429, 500, 503)
FALLBACKS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]


def try_model(model, headers):
    body = {"contents": [{"parts": [{"text": "Reply with exactly: Factory online."}]}]}
    for attempt in range(1, 4):
        try:
            r = requests.post(
                f"{BASE}/models/{model}:generateContent",
                headers=headers, json=body, timeout=TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"[{model}] attempt {attempt}: {type(e).__name__}")
            time.sleep(5 * attempt)
            continue
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"[{model}] attempt {attempt}: HTTP {r.status_code}")
        if r.status_code in RETRY_CODES:
            time.sleep(5 * attempt)
            continue
        print(r.text[:300])
        return None
    return None


def main():
    if not KEY:
        print("ERROR: GEMINI_API_KEY secret is missing or empty.")
        sys.exit(1)
    headers = {"x-goog-api-key": KEY}

    models = [os.environ.get("GEMINI_MODEL", "").strip()] + FALLBACKS
    seen = set()
    for model in models:
        if not model or model in seen:
            continue
        seen.add(model)
        print("Testing model:", model)
        text = try_model(model, headers)
        if text:
            print("Gemini says:", text)
            print("WORKING MODEL:", model)
            print("MILESTONE 1 PASSED")
            return
    print("ERROR: every model failed. See messages above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
