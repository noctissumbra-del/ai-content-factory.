import os
import sys
import requests

BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY = os.environ.get("GEMINI_API_KEY", "").strip()


def main():
    if not KEY:
        print("ERROR: GEMINI_API_KEY secret is missing or empty.")
        sys.exit(1)
    headers = {"x-goog-api-key": KEY}

    r = requests.get(f"{BASE}/models", headers=headers, timeout=30)
    if r.status_code != 200:
        print(f"ERROR listing models: HTTP {r.status_code}")
        print(r.text[:500])
        sys.exit(1)
    models = [
        m["name"].split("/")[-1]
        for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    flash = [m for m in models if "flash" in m and "image" not in m]
    print("Available text models:", ", ".join(flash) or "none found")

    model = os.environ.get("GEMINI_MODEL", "").strip() or (flash[0] if flash else "")
    if not model:
        print("ERROR: no usable model found.")
        sys.exit(1)
    print("Testing model:", model)

    body = {"contents": [{"parts": [{"text": "Reply with exactly: Factory online."}]}]}
    r = requests.post(
        f"{BASE}/models/{model}:generateContent",
        headers=headers, json=body, timeout=60,
    )
    if r.status_code != 200:
        print(f"ERROR generating: HTTP {r.status_code}")
        print(r.text[:500])
        sys.exit(1)
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    print("Gemini says:", text.strip())
    print("MILESTONE 1 PASSED")


if __name__ == "__main__":
    main()
