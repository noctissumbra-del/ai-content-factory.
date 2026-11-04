import json
import os
import time

import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"
RETRY = (429, 500, 503)


def load_config():
    with open("config/config.json", encoding="utf-8") as f:
        return json.load(f)


def generate(prompt, as_json=False):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    cfg = load_config()
    models = [os.environ.get("GEMINI_MODEL", "").strip()] + cfg["models"]
    headers = {"x-goog-api-key": key}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if as_json:
        body["generationConfig"] = {"responseMimeType": "application/json"}

    last = "no models tried"
    seen = set()
    for model in models:
        if not model or model in seen:
            continue
        seen.add(model)
        for attempt in range(1, 4):
            try:
                r = requests.post(
                    f"{BASE}/models/{model}:generateContent",
                    headers=headers, json=body, timeout=90,
                )
            except requests.exceptions.RequestException as e:
                last = f"{model}: {type(e).__name__}"
                time.sleep(5 * attempt)
                continue
            if r.status_code == 200:
                try:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    last = f"{model}: empty response"
                    break
                if as_json:
                    try:
                        data = json.loads(text)
                    except ValueError:
                        last = f"{model}: invalid JSON"
                        continue
                    print(f"[gemini] used {model}")
                    return data
                print(f"[gemini] used {model}")
                return text.strip()
            last = f"{model}: HTTP {r.status_code}"
            if r.status_code in RETRY:
                time.sleep(5 * attempt)
                continue
            break
    raise RuntimeError(f"All Gemini models failed. Last error: {last}")
