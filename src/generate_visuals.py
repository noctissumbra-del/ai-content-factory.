import glob
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from gemini_client import generate, load_config

W, H = 1080, 1920
NASA_SEARCH = "https://images-api.nasa.gov/search"
NASA_ASSET = "https://images-api.nasa.gov/asset/"

# Skip images likely to show identifiable people or carry third-party rights.
AVOID = (
    "astronaut", "portrait", "crew", "administrator", "secretary", "president",
    "senator", "congress", "ceremony", "interview", "press conference",
    "employee", "staff", "team members", "visit", "award",
)
RIGHTS_FLAGS = ("copyright", "\u00a9", "courtesy of", "used with permission")
STOP = {
    "a", "an", "the", "of", "and", "in", "on", "with", "to", "for", "from",
    "showing", "shows", "show", "clean", "simple", "image", "illustration",
    "diagram", "animation", "original", "drawing", "view", "that", "this",
    "into", "over", "around", "while", "very", "some", "style",
}
PALETTES = [  # (top, bottom, nebula glow)
    ((4, 7, 22), (38, 18, 88), (110, 60, 190)),
    ((3, 10, 26), (8, 66, 108), (40, 140, 200)),
    ((10, 5, 20), (88, 24, 68), (200, 70, 130)),
    ((3, 9, 18), (16, 84, 78), (50, 190, 160)),
]
PLANETS = [  # (lit, dark, atmosphere)
    ((90, 140, 220), (8, 16, 44), (110, 170, 255)),
    ((220, 140, 90), (40, 14, 8), (255, 170, 110)),
    ((130, 210, 180), (8, 36, 36), (150, 240, 210)),
    ((200, 180, 240), (28, 18, 58), (220, 200, 255)),
]


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------- inputs
def latest_script():
    files = glob.glob("data/scripts/script_*.json")
    if not files:
        raise RuntimeError("No script in data/scripts/. Run 'Generate Script' first.")
    return max(files, key=os.path.getmtime)


def get_timings(script):
    path = f"output/voice_{script['id']}/timings.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if len(data["scenes"]) == len(script["scenes"]):
            log("Using real voice timings.")
            return [(s["start"], s["end"]) for s in data["scenes"]]
    log("No voice timings found, estimating scene lengths from the text.")
    words = sum(len(s["text"].split()) for s in script["scenes"])
    total = max(30.0, min(58.0, words / 2.4))
    chars = [max(1, len(s["text"])) for s in script["scenes"]]
    out, t = [], 0.0
    for c in chars:
        d = total * c / sum(chars)
        out.append((round(t, 2), round(t + d, 2)))
        t += d
    return out


# ---------------------------------------------------------------- search terms
def fallback_query(visual):
    words = [w for w in re.findall(r"[a-z]+", visual.lower())
             if w not in STOP and len(w) > 2]
    return {"query": " ".join(words[:3]) or "space",
            "alt_query": words[0] if words else "space"}


def scene_queries(script):
    scenes = script["scenes"]
    lines = "\n".join(
        f'{i + 1}. narration: "{s["text"]}" | idea: {s["visual"]}'
        for i, s in enumerate(scenes)
    )
    prompt = (
        "For each scene below, write search terms for NASA's public image library "
        "that would find a real, relevant space or science photo or image.\n"
        "Rules: 'query' is 2-4 words of concrete nouns (examples: 'GPS satellite orbit', "
        "'Earth from space', 'atomic clock'). 'alt_query' is a broader 1-2 word "
        "fallback (example: 'satellite'). No people names, no brand names, no "
        "astronauts.\n"
        f"Scenes:\n{lines}\n"
        'Return a JSON list with exactly one object per scene, in order: '
        '{"query": "...", "alt_query": "..."}.'
    )
    try:
        res = generate(prompt, as_json=True)
        if isinstance(res, dict):
            res = next((v for v in res.values() if isinstance(v, list)), [])
        if len(res) != len(scenes):
            raise ValueError(f"expected {len(scenes)} items, got {len(res)}")
        out = []
        for r, s in zip(res, scenes):
            fb = fallback_query(s["visual"])
            out.append({
                "query": str(r.get("query", "")).strip() or fb["query"],
                "alt_query": str(r.get("alt_query", "")).strip() or fb["alt_query"],
            })
        return out
    except Exception as e:
        log("Gemini search terms failed, using simple keywords:", str(e)[:200])
        return [fallback_query(s["visual"]) for s in scenes]


# ---------------------------------------------------------------- NASA images
def nasa_find(query, used_ids):
    try:
        r = requests.get(
            NASA_SEARCH,
            params={"q": query, "media_type": "image", "page_size": 25},
            timeout=30,
        )
        if r.status_code != 200:
            log(f"  NASA search HTTP {r.status_code} for '{query}'")
            return None
        items = r.json()["collection"]["items"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        log(f"  NASA search failed for '{query}': {type(e).__name__}")
        return None
    for it in items:
        d = (it.get("data") or [{}])[0]
        nid = d.get("nasa_id")
        title = d.get("title", "") or ""
        blob = " ".join([title, d.get("description", "") or "",
                         " ".join(d.get("keywords", []) or [])]).lower()
        if not nid or nid in used_ids or not d.get("center"):
            continue
        if any(w in blob for w in AVOID) or any(w in blob for w in RIGHTS_FLAGS):
            continue
        return {"id": nid, "title": title, "center": d.get("center", "")}
    return None


def pick_hrefs(hrefs):
    imgs = [h.replace("http://", "https://") for h in hrefs
            if re.search(r"\.(jpe?g|png)$", h, re.I)]

    def rank(h):
        for k, tag in enumerate(("~large", "~medium", "~orig")):
            if tag in h:
                return k
        return 9

    return sorted(imgs, key=rank)[:3]


def nasa_download(nid):
    try:
        r = requests.get(NASA_ASSET + nid, timeout=30)
        if r.status_code != 200:
            return None
        hrefs = [i.get("href", "") for i in r.json()["collection"]["items"]]
    except (requests.exceptions.RequestException, KeyError, ValueError):
        return None
    for url in pick_hrefs(hrefs):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200 or len(resp.content) > 15_000_000:
                continue
            img = Image.open(io.BytesIO(resp.content))
            img.load()
        except Exception:
            continue
        if min(img.size) >= 500:
            return img.convert("RGB")
    return None


# ---------------------------------------------------------------- drawing
def darken(img):
    """Dark gradient at top and bottom so captions stay readable."""
    mask = Image.new("L", (1, H))
    for y in range(H):
        t = y / H
        a = 0
        if t < 0.15:
            a = int(110 * (1 - t / 0.15))
        if t > 0.50:
            a = max(a, int(185 * ((t - 0.50) / 0.50) ** 1.3))
        mask.putpixel((0, y), a)
    mask = mask.resize((W, H))
    return Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), img, mask)


def cover(img, w, h):
    s = max(w / img.width, h / img.height)
    nw, nh = max(w, round(img.width * s)), max(h, round(img.height * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    x, y = (nw - w) // 2, (nh - h) // 2
    return img.crop((x, y, x + w, y + h))


def compose_photo(img):
    if img.width / img.height < 0.8:
        frame = ImageEnhance.Brightness(cover(img, W, H)).enhance(0.9)
    else:
        bg = cover(img, W, H).filter(ImageFilter.GaussianBlur(40))
        bg = ImageEnhance.Brightness(bg).enhance(0.45)
        fg_w, fg_h = W, round(img.height * W / img.width)
        if fg_h > H * 0.62:
            fg_h = int(H * 0.62)
            fg_w = round(img.width * fg_h / img.height)
        fg = img.resize((fg_w, fg_h), Image.LANCZOS)
        frame = bg.copy()
        frame.paste(fg, ((W - fg_w) // 2, int(H * 0.40 - fg_h / 2)))
    return darken(frame)


def procedural(seed_text):
    """Original space art drawn from scratch. Same text always gives same picture."""
    rnd = random.Random(int(hashlib.md5(seed_text.encode()).hexdigest(), 16))
    top, bottom, glow = rnd.choice(PALETTES)
    grad = Image.linear_gradient("L").resize((W, H))
    img = Image.composite(Image.new("RGB", (W, H), bottom),
                          Image.new("RGB", (W, H), top), grad).convert("RGBA")

    neb = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    nd = ImageDraw.Draw(neb)
    for _ in range(rnd.randint(4, 7)):
        cx, cy = rnd.randint(0, W), rnd.randint(0, int(H * 0.8))
        rx, ry = rnd.randint(200, 480), rnd.randint(250, 600)
        col = tuple(max(0, min(255, c + rnd.randint(-20, 40))) for c in glow)
        nd.ellipse((cx - rx, cy - ry, cx + rx, cy + ry),
                   fill=col + (rnd.randint(60, 120),))
    img = Image.alpha_composite(img, neb.filter(ImageFilter.GaussianBlur(130)))

    # stars (drawn first so the planet sits in front of them)
    stars = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stars)
    for _ in range(420):
        x, y, r = rnd.randint(0, W - 1), rnd.randint(0, H - 1), rnd.choice((1, 1, 1, 2))
        sd.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, rnd.randint(90, 220)))
    big = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    for _ in range(16):
        x, y, r = rnd.randint(0, W - 1), rnd.randint(0, H - 1), rnd.randint(3, 5)
        bd.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 230))
    img = Image.alpha_composite(img, stars)
    img = Image.alpha_composite(img, big.filter(ImageFilter.GaussianBlur(2.5)))
    img = Image.alpha_composite(img, big)

    # planet (with optional ring)
    lit, dark, atmo = rnd.choice(PLANETS)
    d = rnd.randint(380, 640)
    px = rnd.randint(int(W * 0.30), int(W * 0.70))
    py = rnd.randint(int(H * 0.18), int(H * 0.50))
    if rnd.random() < 0.5:
        ring = Image.new("RGBA", (W * 2, W * 2), (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        rw, rh = int(d * 1.9), int(d * 0.45)
        rd.ellipse((W - rw // 2, W - rh // 2, W + rw // 2, W + rh // 2),
                   outline=atmo + (150,), width=6)
        ring = ring.rotate(rnd.randint(-28, 28), resample=Image.BICUBIC)
        img.alpha_composite(ring.crop((W - px, W - py, 2 * W - px, 2 * W - py)))
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse(
        (px - d // 2 - 30, py - d // 2 - 30, px + d // 2 + 30, py + d // 2 + 30),
        fill=atmo + (120,))
    img = Image.alpha_composite(img, halo.filter(ImageFilter.GaussianBlur(45)))
    g = Image.radial_gradient("L").resize((2 * d, 2 * d))
    off = int(0.65 * d)
    g = g.crop((off, off, off + d, off + d))
    surface = Image.composite(Image.new("RGB", (d, d), dark),
                              Image.new("RGB", (d, d), lit), g)
    pm = Image.new("L", (d * 4, d * 4), 0)
    ImageDraw.Draw(pm).ellipse((0, 0, d * 4 - 1, d * 4 - 1), fill=255)
    pm = pm.resize((d, d), Image.LANCZOS)
    img.paste(surface, (px - d // 2, py - d // 2), pm)
    return darken(img.convert("RGB"))


def contact_sheet(paths, out_path):
    cw, ch, gap = 270, 480, 12
    cols = 3
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * (cw + gap) + gap, rows * (ch + gap) + gap), (20, 20, 20))
    for i, p in enumerate(paths):
        im = Image.open(p).resize((cw, ch), Image.LANCZOS)
        sheet.paste(im, (gap + (i % cols) * (cw + gap), gap + (i // cols) * (ch + gap)))
    sheet.save(out_path, quality=85)


# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    if cfg.get("emergency_stop"):
        log("Emergency stop is ON. Exiting.")
        return

    with open(latest_script(), encoding="utf-8") as f:
        script = json.load(f)
    log("Script:", script["title"])
    scenes = script["scenes"]
    timings = get_timings(script)
    queries = scene_queries(script)

    out_dir = f"output/visuals_{script['id']}"
    os.makedirs(out_dir, exist_ok=True)
    use_nasa = cfg.get("use_nasa_images", True)
    used, manifest, paths = set(), [], []

    for i, scene in enumerate(scenes):
        frame, source, credit, q_used = None, "procedural", None, None
        if use_nasa:
            for q in (queries[i]["query"], queries[i]["alt_query"]):
                hit = nasa_find(q, used)
                if not hit:
                    continue
                img = nasa_download(hit["id"])
                if img is not None:
                    used.add(hit["id"])
                    frame = compose_photo(img)
                    source, q_used = "nasa", q
                    credit = f"NASA ({hit['center']}): {hit['title']} [{hit['id']}]"
                    break
                time.sleep(0.5)
        if frame is None:
            frame = procedural(f"{script['id']}-{i}-{scene['visual']}")
        path = f"{out_dir}/scene_{i:02d}.jpg"
        frame.save(path, quality=92)
        paths.append(path)
        log(f"Scene {i + 1}/{len(scenes)}: {source}"
            + (f" (query '{q_used}')" if q_used else ""))
        manifest.append({
            "index": i, "file": path, "source": source, "credit": credit,
            "query": q_used, "text": scene["text"],
            "start": timings[i][0], "end": timings[i][1],
        })

    contact_sheet(paths, f"{out_dir}/preview.jpg")
    credits = [m["credit"] for m in manifest if m["credit"]]
    with open(f"{out_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump({"id": script["id"], "scenes": manifest, "credits": credits},
                  f, indent=2, ensure_ascii=False)

    for m in manifest:
        with Image.open(m["file"]) as im:
            if im.size != (W, H):
                raise RuntimeError(f"{m['file']} has wrong size {im.size}")
    if len(manifest) < 4:
        raise RuntimeError("Fewer than 4 scenes. The script needs at least 4.")

    n_nasa = sum(1 for m in manifest if m["source"] == "nasa")
    log(f"Summary: {n_nasa} NASA images, {len(manifest) - n_nasa} original drawings")
    log("MILESTONE 4 PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FAILED:", e)
        sys.exit(1)
