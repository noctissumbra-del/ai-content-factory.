import glob
import json
import os
import subprocess
import sys
import urllib.request
import wave

from gemini_client import load_config

HF = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
VOICE_DIR = "assets/voices"


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


def synth(text, model, out_path):
    last = ""
    for flag in ("--output_file", "--output-file", "-f"):
        cmd = [sys.executable, "-m", "piper", "--model", model, flag, out_path]
        r = subprocess.run(cmd, input=text, text=True, capture_output=True)
        if r.returncode == 0 and os.path.exists(out_path):
            return
        last = (r.stderr or r.stdout)[-400:]
    raise RuntimeError("Piper failed: " + last)


def prepare_voice(cfg):
    for voice in (cfg["voice"], cfg["fallback_voice"]):
        try:
            model = ensure_voice(voice)
            synth("Test.", model, "output/_test.wav")
            os.remove("output/_test.wav")
            print("Using voice:", voice)
            return voice, model
        except Exception as e:
            print(f"Voice {voice} failed: {e}")
    raise RuntimeError("No working voice found.")


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def latest_script():
    files = glob.glob("data/scripts/script_*.json")
    if not files:
        raise RuntimeError("No script found in data/scripts/. Run 'Generate Script' first.")
    return max(files, key=os.path.getmtime)


def main():
    cfg = load_config()
    if cfg.get("emergency_stop"):
        print("Emergency stop is ON. Exiting.")
        return

    os.makedirs("output", exist_ok=True)
    path = latest_script()
    with open(path, encoding="utf-8") as f:
        script = json.load(f)
    print("Script:", script["title"])

    voice, model = prepare_voice(cfg)
    out_dir = f"output/voice_{script['id']}"
    os.makedirs(out_dir, exist_ok=True)

    parts = []
    for i, scene in enumerate(script["scenes"]):
        wav = f"{out_dir}/scene_{i:02d}.wav"
        synth(scene["text"], model, wav)
        parts.append((wav, scene))

    with wave.open(parts[0][0], "rb") as w:
        params = w.getparams()
    rate, width, channels = params.framerate, params.sampwidth, params.nchannels
    pause = int(cfg["scene_pause_seconds"] * rate)
    silence = b"\x00" * (pause * width * channels)

    timings, t = [], 0.0
    final = f"{out_dir}/narration.wav"
    with wave.open(final, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        for i, (wav, scene) in enumerate(parts):
            with wave.open(wav, "rb") as w:
                frames = w.readframes(w.getnframes())
                dur = w.getnframes() / rate
            out.writeframes(frames)
            end = t + dur
            timings.append({
                "index": i, "text": scene["text"], "visual": scene["visual"],
                "start": round(t, 2), "end": round(end, 2),
            })
            if i < len(parts) - 1:
                out.writeframes(silence)
                end += cfg["scene_pause_seconds"]
            t = end
    for wav, _ in parts:
        os.remove(wav)

    total = wav_duration(final)
    with open(f"{out_dir}/timings.json", "w", encoding="utf-8") as f:
        json.dump({"id": script["id"], "voice": voice, "total_seconds": round(total, 2),
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
