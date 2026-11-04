import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from generate_script import make_script
from gemini_client import load_config
from research import load_history, pick_topic, save_history


def main():
    cfg = load_config()
    if cfg.get("emergency_stop"):
        print("Emergency stop is ON in config/config.json. Exiting.")
        return

    script, picked, tried = None, None, []
    for _ in range(3):
        picked = pick_topic(exclude=tried)
        tried.append(picked["topic"])
        print("Topic:", picked["topic"])
        try:
            script = make_script(picked["topic"], picked["angle"])
            break
        except RuntimeError as e:
            print("Topic rejected:", e)
    if script is None:
        raise RuntimeError("Three different topics all failed. Try again later.")

    vid = hashlib.sha1(picked["topic"].encode()).hexdigest()[:10]
    script["id"] = vid
    script["topic"] = picked["topic"]

    os.makedirs("output", exist_ok=True)
    with open(f"output/script_{vid}.json", "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    history = load_history()
    history.append({
        "id": vid,
        "topic": picked["topic"],
        "title": script["title"],
        "created": datetime.now(timezone.utc).isoformat(),
        "status": "scripted",
    })
    save_history(history)

    print(json.dumps(script, indent=2, ensure_ascii=False))
    print("MILESTONE 2 PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)
