import difflib
import json
import os
import re

from gemini_client import generate, load_config

HISTORY = "data/history.json"


def load_history():
    if os.path.exists(HISTORY):
        with open(HISTORY, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(history):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def is_duplicate(topic, history):
    n = _norm(topic)
    return any(
        difflib.SequenceMatcher(None, n, _norm(h["topic"])).ratio() > 0.75
        for h in history
    )


def pick_topic(exclude=()):
    cfg = load_config()
    history = load_history()
    recent = [h["topic"] for h in history[-100:]] + list(exclude)
    prompt = (
        f"You are a researcher for a short-video channel about {cfg['niche']}: "
        f"{cfg['niche_description']}.\n"
        "Suggest 10 specific, surprising, well-established topics that would make "
        "a gripping 40-second video. Only topics supported by solid science. "
        "Avoid vague topics. Do NOT repeat or closely resemble these already-covered topics:\n"
        f"{json.dumps(recent)}\n"
        'Return JSON: a list of objects like {"topic": "...", "angle": "..."}.'
    )
    ideas = generate(prompt, as_json=True)
    if isinstance(ideas, dict):
        ideas = next((v for v in ideas.values() if isinstance(v, list)), [])
    for idea in ideas:
        topic = str(idea.get("topic", "")).strip()
        if topic and topic not in exclude and not is_duplicate(topic, history):
            return {"topic": topic, "angle": str(idea.get("angle", "")).strip()}
    raise RuntimeError("No new topic found (all ideas were duplicates).")
