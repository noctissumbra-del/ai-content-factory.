import array
import base64
import glob
import json
import math
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
GEMINI_BUDGET_SECONDS = 300
MAX_SPEEDUP = 1.15


def log(*args):
    print(*args, flush=True)


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


# ---------- tail cleanup (removes the TTS glitch burst at the end) ----------
def clean_tail(path):
    with wave.open(path, "rb") as w:
        params = w.getparams()
        data = w.readframes(w.getnframes())
    if params.sampwidth != 2 or params.nchannels != 1:
        return
    rate = params.framerate
    samples = array.array("h")
    samples.frombytes(data)
    win = int(rate * 0.05)
    n = len(samples) // win
    if n < 10:
        return
    rms = [math.sqrt(sum(x * x for x in samples[i * win:(i + 1) * win]) / win)
           for i in range(n)]
    p90 = sorted(rms)[int(n * 0.9)]
    thr = max(p90 * 0.05, 30)
    loud = [x > thr for x in rms]

    i = n - 1
    while i >= 0 and not loud[i]:
        i -= 1
    if i < 0:
        return
    seg_end = i
    while i >= 0 and loud[i]:
        i -= 1
    seg_len = (seg_end - i) * 0.05
    gap = 0
    j = i
    while j >= 0 and not loud[j]:
        gap += 1
        j -= 1

    if seg_len < 0.6 and gap * 0.05 >= 0.3 and j >= 0:
        keep = (j + 1) * win
        log(f"Removed {(len(samples) - keep) / rate:.2f}s glitch burst at the end")
    else:
        keep = (seg_end + 1) * win
    keep = min(len(samples), keep + int(rate * 0.12))
    out = samples[:keep]

    fade = min(len(out), int(rate * 0.08))
    for k in range(fade):
        idx = len(out) - fade + k
        out[idx] = int(out[idx] * (1 - (k + 1) / fade))

    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(out.tobytes())


# ---------- Gemini TTS (one request for the whole narration) ----------
def gemini_tts(text, cfg, out_path, deadline):
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
            if time.time() > deadline:
                raise RuntimeError("Gemini time budget used up. Last: " + last)
            try:
                r = requests.post(f"{BASE}/models/{model}:generateContent",
                                  headers=headers, json=body, timeout=60)
            except requests.exceptions.RequestException as e:
                last = f"{model}: {type(e).__name__}"
                log("  retry:", last)
                time.sleep(5)
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
            log("  retry:", last)
            if r.status_code == 429:
                time.sleep(20)
                continue
            if r.status_code in (500, 503):
                time.sleep(5)
                continue
            break
    raise RuntimeError(last)


def proportional_timings(script, total):
    chars = [max(1, len(s["text"])) for s in script["scenes"]]
    whole = sum(chars)
    timings, t = [], 0.0
    for i, (scene, c) in enumerate(zip(script["scenes"], chars)):
        dur = total * c / whole
        timings.append({"index": i, "text": scene["text"], "visual": scene["visual"],
                        "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur
    return timings


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
            log("Downloading", url)
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


def piper_narration(script, cfg, out_dir, final):
    model, label = None, None
    for voice in (cfg["voice"], cfg["fallback_voice"]):
        try:
            model = ensure_voice(voice)
            piper_synth("Test.", model, f"{out_dir}/_test.wav")
            os.remove(f"{out_dir}/_test.wav")
            label = f"piper:{voice}"
            break
        except Exception as e:
            log(f"Piper voice {voice} failed: {e}")
    if model is None:
        raise RuntimeError("No working Piper voice.")

    parts = []
    for i, scene in enumerate(script["scenes"]):
        wav = f"{out_dir}/scene_{i:02d}.wav"
        piper_synth(scene["text"], model, wav)
        log(f"Scene {i + 1}/{len(script['scenes'])} voiced by piper")
        parts.append((wav, scene))

    with wave.open(parts[0][0], "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
    pause = float(cfg["scene_pause_seconds"])
    silence = b"\x00" * (int(pause * rate) * width * channels)

    timings, t = [], 0.0
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
    return label, timings


# ---------- speed adjust ----------
def speed_up(path, factor):
    tmp = path + ".fast.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-filter:a", f"atempo={factor:.3f}", tmp],
        check=True,
    )
    os.replace(tmp, path)


def latest_script():
    files = glob.glob("data/scripts/script_*.json")
    if not files:
        raise RuntimeError("No script in data/scripts/. Run 'Generate Script' first.")
    return max(files, key=os.path.getmtime)


def main():
    cfg = load_config()
    if cfg.get("emergency_stop"):
        log("Emergency stop is ON. Exiting.")
        return

    os.makedirs("output", exist_ok=True)
    with open(latest_script(), encoding="utf-8") as f:
        script = json.load(f)
    log("Script:", script["title"])

    out_dir = f"output/voice_{script['id']}"
    os.makedirs(out_dir, exist_ok=True)
    final = f"{out_dir}/narration.wav"

    label, timings = None, None
    if cfg.get("tts_provider", "gemini") == "gemini":
        try:
            full_text = " ".join(s["text"] for s in script["scenes"])
            model = gemini_tts(full_text, cfg, final,
                               time.time() + GEMINI_BUDGET_SECONDS)
            clean_tail(final)
            label = f"gemini:{model}:{cfg['gemini_voice']}"
            timings = proportional_timings(script, wav_duration(final))
        except Exception as e:
            log("Gemini TTS failed, falling back to Piper:", e)
    if timings is None:
        label, timings = piper_narration(script, cfg, out_dir, final)
    log("Voice used:", label)

    total = wav_duration(final)
    words = sum(len(s["text"].split()) for s in script["scenes"])
    log(f"Raw narration length: {total:.1f} seconds for {words} words")
    if total > words / 1.2:
        log("WARNING: audio is much longer than the script. The voice may have "
            "read the style instruction aloud. Listen to the first seconds.")

    target = float(cfg.get("target_max_seconds", 55))
    if total > target:
        factor = min(MAX_SPEEDUP, total / target)
        speed_up(final, factor)
        new_total = wav_duration(final)
        scale = new_total / total
        for t in timings:
            t["start"] = round(t["start"] * scale, 2)
            t["end"] = round(t["end"] * scale, 2)
        log(f"Sped up x{factor:.2f}")
        total = new_total

    with open(f"{out_dir}/timings.json", "w", encoding="utf-8") as f:
        json.dump({"id": script["id"], "voice": label, "total_seconds": round(total, 2),
                   "scenes": timings}, f, indent=2, ensure_ascii=False)

    log(f"Final narration length: {total:.1f} seconds")
    if total < 15 or total > 62:
        raise RuntimeError(f"Narration is {total:.1f}s, outside the 15-62s range.")
    log("MILESTONE 3 PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FAILED:", e)
        sys.exit(1)
