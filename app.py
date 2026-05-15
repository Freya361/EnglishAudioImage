import os
import io
import re
import zipfile
import threading
import subprocess
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from pydub import AudioSegment

app = Flask(__name__)

PHRASES_MD    = os.path.join(os.path.dirname(__file__), "phrases.md")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "static", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(number: int, phrase: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9 _-]", "_", phrase)[:30].strip()
    return f"{number:02d}_{safe}"


def file_urls(base: str) -> dict:
    return {
        "image_url": f"/static/generated/{base}.png",
        "audio_url": f"/static/generated/{base}.mp3",
        "video_url": f"/static/generated/{base}.mp4",
    }


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def get_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    for path, idx in [
        (r"C:\Windows\Fonts\msyh.ttc", 0),
        (r"C:\Windows\Fonts\msjh.ttc", 0),
        (r"C:\Windows\Fonts\simsun.ttc", 0),
        (r"C:\Windows\Fonts\simhei.ttf", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ]:
        try:
            return ImageFont.truetype(path, size, index=idx)
        except Exception:
            continue
    return get_font(size)


# ── Text wrapping ─────────────────────────────────────────────────────────────

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if draw.textbbox((0, 0), " ".join(current), font=font)[2] > max_width and len(current) > 1:
            current.pop()
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


# ── Highlighted sentence renderer ─────────────────────────────────────────────

def draw_highlighted_line(draw, x, y, line, phrase, font, text_color, hl_bg, hl_fg):
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    cursor, last = x, 0
    for m in pattern.finditer(line):
        if m.start() > last:
            seg = line[last:m.start()]
            draw.text((cursor, y), seg, fill=text_color, font=font)
            cursor += int(draw.textlength(seg, font=font))
        seg   = m.group()
        seg_w = int(draw.textlength(seg, font=font))
        seg_h = draw.textbbox((0, 0), seg, font=font)[3]
        pad   = 4
        draw.rounded_rectangle(
            [(cursor - pad, y - pad), (cursor + seg_w + pad, y + seg_h + pad)],
            radius=5, fill=hl_bg,
        )
        draw.text((cursor, y), seg, fill=hl_fg, font=font)
        cursor += seg_w
        last = m.end()
    if last < len(line):
        draw.text((cursor, y), line[last:], fill=text_color, font=font)


# ── Markdown parser ───────────────────────────────────────────────────────────

def parse_phrases_md() -> list[dict]:
    if not os.path.exists(PHRASES_MD):
        return []
    entries: list[dict] = []
    current: dict | None = None
    sentences: list[str] = []
    for line in open(PHRASES_MD, encoding="utf-8"):
        line = line.rstrip()
        m = re.match(r'^(\d+)\.\s+(.+)$', line)
        if m:
            if current is not None:
                current["sentences"] = sentences
                entries.append(current)
            full = m.group(2).strip()
            tm   = re.match(r'^(.+?)\s*\((.+)\)\s*$', full)
            current = {
                "index":       len(entries),
                "number":      int(m.group(1)),
                "phrase":      tm.group(1).strip() if tm else full,
                "translation": tm.group(2).strip() if tm else "",
            }
            sentences = []
        elif current is not None and line.strip():
            sentences.append(line.strip())
    if current is not None:
        current["sentences"] = sentences
        entries.append(current)
    return entries


# ── Image creation (1080 × 1440) ─────────────────────────────────────────────

def create_image(number: int, phrase: str, translation: str, sentences: list[str], out_path: str):
    W, H  = 1080, 1440
    PAD   = 64
    BG_TOP  = (8,  14,  38);  BG_BOT  = (3,   7,  18)
    ACCENT  = (99, 102, 241); GOLD    = (250, 204,  21)
    TRANS_C = (147, 197, 253); TEXT   = (226, 232, 240)
    MUTED   = (100, 116, 139); NUM_BG = (79,  70,  229)
    DIVIDER = (28,  38,  70);  FOOT_BG = (10,  16,  40)
    HL_BG   = (250, 204,  21); HL_FG  = (15,  23,  42)

    phrase_font = get_font(76);  trans_font = get_cjk_font(44)
    body_font   = get_font(38);  label_font = get_font(28)
    num_font    = get_font(26);  badge_font = get_font(24)

    t   = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(BG_TOP, float) * (1 - t) + np.array(BG_BOT, float) * t).astype(np.uint8)
    arr = np.broadcast_to(arr, (H, W, 3)).copy()
    img  = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Number badge
    badge_txt = f"#{number:02d}"
    bb = draw.textbbox((0, 0), badge_txt, font=badge_font)
    bw, bh = bb[2] - bb[0] + 28, bb[3] - bb[1] + 14
    bx1 = W - PAD - bw
    draw.rounded_rectangle([(bx1, 52), (bx1 + bw, 52 + bh)], radius=10, fill=NUM_BG)
    draw.text((bx1 + bw // 2, 52 + bh // 2), badge_txt, fill="white", font=badge_font, anchor="mm")

    # Phrase title
    probe_draw = ImageDraw.Draw(Image.new("RGB", (W, 10)))
    y = 150
    for line in wrap_text(probe_draw, f'"{phrase}"', phrase_font, W - PAD * 2):
        draw.text((W // 2, y), line, fill=GOLD, font=phrase_font, anchor="mt")
        y += 92

    # Chinese translation
    y += 12
    if translation:
        draw.text((W // 2, y), f"({translation})", fill=TRANS_C, font=trans_font, anchor="mt")
        y += 62

    # Divider
    y += 28
    draw.rectangle([(PAD, y), (W - PAD, y + 3)], fill=ACCENT)
    r = 9
    draw.ellipse([(W // 2 - r, y - r + 1), (W // 2 + r, y + r + 2)], fill=ACCENT)
    y += 36
    draw.text((W // 2, y), "Example Sentences", fill=MUTED, font=label_font, anchor="mt")
    y += 52
    draw.rectangle([(PAD, y), (W - PAD, y + 1)], fill=DIVIDER)
    y += 30

    # Sentences
    SLOT   = (H - y - 80) // max(len(sentences), 1)
    LINE_H = 52
    for i, sentence in enumerate(sentences, 1):
        slot_y = y + (i - 1) * SLOT
        r_badge = 22
        cx, cy  = PAD + r_badge, slot_y + r_badge + 10
        draw.ellipse([(cx - r_badge, cy - r_badge), (cx + r_badge, cy + r_badge)], fill=NUM_BG)
        draw.text((cx, cy), str(i), fill="white", font=num_font, anchor="mm")
        for j, ln in enumerate(wrap_text(probe_draw, sentence, body_font, W - PAD * 2 - 70)):
            draw_highlighted_line(draw, PAD + 62, slot_y + 10 + j * LINE_H, ln, phrase, body_font, TEXT, HL_BG, HL_FG)
        if i < len(sentences):
            div_y = slot_y + SLOT - 1
            draw.rectangle([(PAD + 62, div_y), (W - PAD, div_y + 1)], fill=DIVIDER)

    # Footer
    draw.rectangle([(0, H - 72), (W, H)], fill=FOOT_BG)
    draw.text((W // 2, H - 36), "English Phrase Learning Card", fill=MUTED, font=badge_font, anchor="mm")

    img.save(out_path, format="PNG", optimize=True)


# ── Audio creation ────────────────────────────────────────────────────────────

def _tts_segment(text: str) -> AudioSegment:
    buf = io.BytesIO()
    gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
    buf.seek(0)
    return AudioSegment.from_mp3(buf)


def create_audio(phrase: str, sentences: list[str], out_path: str):
    silence  = AudioSegment.silent(duration=1500)
    combined = _tts_segment(phrase + ".")
    for sentence in sentences:
        combined += silence + _tts_segment(sentence)
    combined.export(out_path, format="mp3")


# ── Video creation ────────────────────────────────────────────────────────────

def create_video(img_path: str, aud_path: str, vid_path: str):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", img_path,
            "-i", aud_path,
            "-c:v", "libx264", "-tune", "stillimage", "-preset", "fast",
            "-crf", "28",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            vid_path,
        ],
        check=True, capture_output=True,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/phrases")
def get_phrases():
    return jsonify(parse_phrases_md())


@app.route("/generated")
def get_generated():
    """Return metadata for phrases whose files already exist on disk."""
    results = []
    for e in parse_phrases_md():
        base = safe_filename(e["number"], e["phrase"])
        paths = [os.path.join(GENERATED_DIR, f"{base}.{ext}") for ext in ("png", "mp3", "mp4")]
        if all(os.path.exists(p) for p in paths):
            results.append({
                "index":       e["index"],
                "number":      e["number"],
                "phrase":      e["phrase"],
                "translation": e["translation"],
                "filename":    base,
                **file_urls(base),
            })
    return jsonify(results)


@app.route("/generate", methods=["POST"])
def generate():
    data    = request.get_json(silent=True) or {}
    idx     = data.get("index")
    entries = parse_phrases_md()

    if idx is None or not (0 <= idx < len(entries)):
        return jsonify({"error": "Invalid phrase index."}), 400

    entry = entries[idx]
    base  = safe_filename(entry["number"], entry["phrase"])
    img_path = os.path.join(GENERATED_DIR, f"{base}.png")
    aud_path = os.path.join(GENERATED_DIR, f"{base}.mp3")
    vid_path = os.path.join(GENERATED_DIR, f"{base}.mp4")

    errors: list[str] = []

    def gen_image():
        try:
            create_image(entry["number"], entry["phrase"],
                         entry["translation"], entry["sentences"], img_path)
        except Exception as exc:
            errors.append(f"Image: {exc}")

    def gen_audio():
        try:
            create_audio(entry["phrase"], entry["sentences"], aud_path)
        except Exception as exc:
            errors.append(f"Audio: {exc}")

    t1 = threading.Thread(target=gen_image)
    t2 = threading.Thread(target=gen_audio)
    t1.start(); t2.start()
    t1.join();  t2.join()

    if errors:
        return jsonify({"error": "; ".join(errors)}), 500

    try:
        create_video(img_path, aud_path, vid_path)
    except Exception as exc:
        return jsonify({"error": f"Video: {exc}"}), 500

    return jsonify({
        "index":       entry["index"],
        "number":      entry["number"],
        "phrase":      entry["phrase"],
        "translation": entry["translation"],
        "filename":    base,
        **file_urls(base),
    })


@app.route("/zip/<int:idx>")
def download_zip(idx):
    entries = parse_phrases_md()
    if not (0 <= idx < len(entries)):
        return jsonify({"error": "Invalid index."}), 404

    entry = entries[idx]
    base  = safe_filename(entry["number"], entry["phrase"])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in ("png", "mp3", "mp4"):
            path = os.path.join(GENERATED_DIR, f"{base}.{ext}")
            if os.path.exists(path):
                zf.write(path, f"{base}.{ext}")
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{base}.zip",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
