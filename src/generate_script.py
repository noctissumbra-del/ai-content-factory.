from gemini_client import generate, load_config


def validate(s, cfg):
    problems = []
    for k in ("hook", "narration", "title", "description", "hashtags", "scenes"):
        if not s.get(k):
            problems.append(f"missing {k}")
    if problems:
        return problems
    words = len(s["narration"].split())
    if not cfg["min_words"] <= words <= cfg["max_words"]:
        problems.append(f"narration has {words} words, need {cfg['min_words']}-{cfg['max_words']}")
    if len(s["hook"].split()) > 15:
        problems.append("hook longer than 15 words")
    if not s["narration"].startswith(s["hook"][:20]):
        problems.append("narration must start with the hook")
    if len(s["title"]) > 90:
        problems.append("title longer than 90 characters")
    if not isinstance(s["hashtags"], list) or len(s["hashtags"]) < 3:
        problems.append("need at least 3 hashtags")
    if not isinstance(s["scenes"], list) or not 4 <= len(s["scenes"]) <= 10:
        problems.append("need 4-10 scenes")
    return problems


def fact_check(narration):
    prompt = (
        "You are a science fact-checker for a popular-science channel. Flag ONLY clear "
        "factual errors or claims that are plainly false or misleading. Do NOT flag "
        "rounding, simplification, or wording style. "
        'Return JSON: {"ok": true/false, "issues": ["..."]}.\n\n' + narration
    )
    res = generate(prompt, as_json=True)
    if isinstance(res, list) and res:
        res = res[0]
    return bool(res.get("ok")), res.get("issues", [])


def make_script(topic, angle):
    cfg = load_config()
    feedback = ""
    for attempt in range(1, 4):
        prompt = (
            f"Write an original script for a vertical short video (30-50 seconds) about: {topic}.\n"
            f"Angle: {angle}\n"
            "Rules: first sentence is a hook that creates curiosity in under 2 seconds "
            "(max 15 words, no 'Did you know'). Plain spoken English, short sentences, "
            f"{cfg['min_words']}-{cfg['max_words']} words of narration total, ends with a "
            "punchy closing line. Only well-established facts. Use words like 'about' "
            "or 'roughly' for numbers and never exaggerate (no 'instantly', 'would "
            "collapse', etc). Do not mention any "
            "copyrighted characters, brands, or people's quotes.\n"
            f"{feedback}\n"
            "Return JSON with keys: hook (string), narration (full spoken text, starting "
            "with the hook), scenes (list of 4-10 objects {\"text\": the narration part, "
            "\"visual\": simple description of an original illustration or diagram}), "
            "title (max 90 chars, no clickbait lies), description (2 sentences), "
            "hashtags (list of 5 strings starting with #)."
        )
        script = generate(prompt, as_json=True)
        if isinstance(script, list) and script:
            script = script[0]
        problems = validate(script, cfg)
        if not problems:
            ok, issues = fact_check(script["narration"])
            if ok:
                return script
            problems = ["fact-check failed: " + "; ".join(map(str, issues))]
        print(f"Attempt {attempt} rejected: {problems}")
        feedback = "Fix these problems from the last attempt: " + "; ".join(problems)
    raise RuntimeError("Could not produce a valid script after 3 attempts.")
