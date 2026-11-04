import base64
import os
import re
import time
import wave

import requests

from generate_voice import ensure_voice, synth

TEXT = (
    "Your phone is secretly bending time. Every day, satellites above you "
    "run slightly fast, and engineers have to correct them. Without that fix, "
    "your map would drift by kilometers. Einstein was right, and your GPS proves it."
)
OUT = "output/voicelab"
PIPER_VOICES = [
    "en_US-ryan-high",
    "en_US-lessac-high",
    "en_GB-cori-high",
    "en_US-hfc_male-medium",
    "en_US-ljspeech-high",
]
GEMINI_MODELS = [
    "gemini-3.1-flash-tts-preview",
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-2.5-flash-preview-tts",
]
GEMINI_VOICES = ["Kore", "Puck", "Charon"]
BASE = "https://generativelanguage.googleapis.com/v1beta"


def piper_samples():
    for voice in PIPER_VOICES:
        try:
            model = ensure_voice(voice)
            synth(TEXT, model, f"{OUT}/piper_{voice}.wav")
            print("OK   piper", voice)
        except Exception as e:
            print("FAIL piper", voice, str(e)[:150])


def gemini_samples():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("FAIL gemini: no key")
        return
    headers = {"x-goog-api-key": key}
    prompt = "Say in an energetic, curious documentary tone: " + TEXT
    for model in GEMINI_MODELS:
        worked = False
        for voice in GEMINI_VOICES:
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
                },
            }
            r = None
            for attempt in range(2):
                try:
                    r = requests.post(f"{BASE}/models/{model}:generateContent",
                                      headers=headers, json=body, timeout=120)
                except requests.exceptions.RequestException as e:
                    print("FAIL gemini", model, voice, type(e).__name__)
                    r = None
                    time.sleep(5)
                    continue
                if r.status_code in (429, 500, 503):
                    time.sleep(8)
                    continue
                break
            if r is None:
                continue
            if r.status_code != 200:
                print("FAIL gemini", model, voice, "HTTP", r.status_code, r.text[:160].replace("\n", " "))
                if r.status_code in (400, 403, 404):
                    break
                continue
            try:
                part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
                pcm = base64.b64decode(part["data"])
                m = re.search(r"rate=(\d+)", part.get("mimeType", ""))
                rate = int(m.group(1)) if m else 24000
            except (KeyError, IndexError, ValueError):
                print("FAIL gemini", model, voice, "unexpected response")
                continue
            with wave.open(f"{OUT}/gemini_{model}_{voice}.wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            print("OK   gemini", model, voice)
            worked = True
        if worked:
            break


def main():
    os.makedirs(OUT, exist_ok=True)
    piper_samples()
    gemini_samples()
    print("VOICE LAB DONE. Download the 'voicelab' artifact and listen.")


if __name__ == "__main__":
    main()
