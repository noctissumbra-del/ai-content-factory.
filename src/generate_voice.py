import base64
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import wave

import requests

from gemini_client import load_config

BASE = "https://generativelanguage.googleapis.com/v1beta"
HF = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
VOICE_DIR = "assets/voices"


# ---------- Gemini TTS ----------
def gemini_tts(text, cfg, out_path):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    headers = {"x-goog-api-key": key}
    body = {
        "contents": [{"parts": [{"text": f"{cfg['voice_style']}: {text}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {
                "voiceName": cfg["gemini_voice"]}}},
        },
    }
    last = "no models tried"
    for model in cfg["gemini_tts_models"]:
        for attempt in range(1, 4):
            try:
                r = requests.post(f"{BASE}/models/{model}:generateContent",
                                  headers=headers, json=body, timeout=120)
            except requests.exceptions.RequestException as e:
                last = f"{model}: {type(e).__name__}"
                time.sleep(5 * attempt)
                continue
            if r.status_code == 200:
                try:
                    part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
                    pcm = base64.b64decode(part["data"])
                    m = re.search(r"rate=(\d+)", part.get("mimeType", ""))
                    rate = int(m.group(1)) if m else 24000
                except (KeyError, IndexError, ValueError):
                    last = f"{model}: response had no audio"
                    break
                with wave.open(out_path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(pcm)
                return model
            last = f"{model}: HTTP {r.status_code}"
            if r.status_code in (429, 500, 503):
                time.sleep(8 * attempt)
                continue
            break
    raise RuntimeError(last)


# ---------- Piper fallback ----------
def voice_urls(voice):
    region, name, quality = voice.split("-")
    lang = region.split("_")[0]
    base = f"{HF}/{lang}/{region}/{name}/{quality}/{voice}"
    return base + ".onnx", base + ".onnx.json"


def ensure_voice(voice):
    os.makedirs(VOICE_DIR, exist_ok=True)
    onnx = os.path.join(VOICE_DIR, voice + ".onnx")
    for url, path in zip(voice_urls(voice), (onnx, onnx + ".json")):
        if not os.path.exists(path):
            print("Downloading", url)
            urllib.request.urlretrieve(url, path)
    return onnx


def piper_synth(text, model, out_path):
    last = ""
    for flag in ("--output_file", "--output-file", "-f"):
        cmd = [sys.executable, "-m", "piper", "--model", model, flag, out_path]
        r = subprocess.run(cmd, input=text, text=True, capture_output=True)
        if r.returncode == 0 and os.path.exists(out_path):
            return
        last = (r.stderr or r.stdout)[-400:]
    raise RuntimeError("Piper failed: " + last)


def piper_model(cfg):
    for voice in (cfg["voice"], cfg["fallback_voice"]):
        try:
            model = ensure_voice(voice)
            piper_synth("Test.", model, "output/_test.wav")
            os.remove("output/_test.wav")
            return voice, model
        except Exception as e:
            print(f"Piper voice {voice} failed: {e}")
    raise RuntimeError("No working Piper voice.")


# ---------- shared ----------
def render_scenes(script, cfg, out_dir, provider):
    """Returns (label, list of (wav_path, scene))."""
    parts = []
    if provider == "gemini":
        label = None
        for i, scene in enumerate(script["scenes"]):
            wav = f"{out_dir}/scene_{i:02d}.wav"
            model = gemini_tts(scene["text"], cfg, wav)
            label = f"gemini:{model}:{cfg['gemini_voice']}"
            parts.append((wav, scene))
            time.sleep(2)
        return label, parts
    voice, model = piper_model(cfg)
    for i, scene in enumerate(script["scenes"]):
        wav = f"{out_dir}/scene_{i:02d}.wav"
        piper_synth(scene["text"], model, wav)
        parts.append((wav, scene))
    return f"piper:{voice}", parts


def latest_script():
    files = glob.glob("data/scripts/script_*.json")
    if not files:
        raise RuntimeError("No script in data/scripts/. Run 'Generate Script' first.")
    return max(files, key=os.path.getmtime)


def main():
    cfg = load_config()
    if cfg.get("emergency_stop"):
        print("Emergency stop is ON. Exiting.")
        return

    os.makedirs("output", exist_ok=True)
    with open(latest_script(), encoding="utf-8") as f:
        script = json.load(f)
    print("Script:", script["title"])

    out_dir = f"output/voice_{script['id']}"
    os.makedirs(out_dir, exist_ok=True)

    label, parts = None, None
    if cfg.get("tts_provider", "gemini") == "gemini":
        try:
            label, parts = render_scenes(script, cfg, out_dir, "gemini")
        except Exception as e:
            print("Gemini TTS failed, falling back to Piper:", e)
    if parts is None:
        label, parts = render_scenes(script, cfg, out_dir, "piper")
    print("Voice used:", label)

    with wave.open(parts[0][0], "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
    pause = float(cfg["scene_pause_seconds"])
    silence = b"\x00" * (int(pause * rate) * width * channels)

    timings, t = [], 0.0
    final = f"{out_dir}/narration.wav"
    with wave.open(final, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        for i, (wav, scene) in enumerate(parts):
            with wave.open(wav, "rb") as w:
                frames = w.readframes(w.getnframes())
                dur = w.getnframes() / w.getframerate()
            out.writeframes(frames)
            end = t + dur
            timings.append({"index": i, "text": scene["text"], "visual": scene["visual"],
                            "start": round(t, 2), "end": round(end, 2)})
            if i < len(parts) - 1:
                out.writeframes(silence)
                end += pause
            t = end
    for wav, _ in parts:
        os.remove(wav)

    with wave.open(final, "rb") as w:
        total = w.getnframes() / w.getframerate()
    with open(f"{out_dir}/timings.json", "w", encoding="utf-8") as f:
        json.dump({"id": script["id"], "voice": label, "total_seconds": round(total, 2),
                   "scenes": timings}, f, indent=2, ensure_ascii=False)

    print(f"Narration length: {total:.1f} seconds")
    if total < 15 or total > 75:
        raise RuntimeError(f"Narration is {total:.1f}s, outside the 15-75s safe range.")
    if not 25 <= total <= 62:
        print("WARNING: aim for 30-60 seconds.")
    print("MILESTONE 3 PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)
