"""The storyboard box: plain text in, a timeline for rolling_timeline out.

Nothing typed means nothing added. Every layer below is optional, and a missing one never raises and never
leaves a placeholder, an empty section or a half-built line in what H3 receives.

1. Just a prompt (the prompt box). Sent as typed, with every loaded picture, to every piece.

2. Pieces, in the storyboard box.  >  starts a shot,  ~  starts a bridge. Everything up to the next marker is
   that piece's text, line breaks or not. If the prompt has {beat} in it the text goes there; if it doesn't,
   the text goes on the end; with no prompt at all the text is the prompt.

3. Settings: bare lowercase words right after the marker, before the text.
     7s  or  158f    length (seconds snap to the nearest length H3 can make: 17k+5 frames, at least 22)
     image 3         hand picture_3 to this piece and every piece after it
     -image 2        stop handing picture_2 over, from this piece on
     take 2          another roll of this piece (a different seed); take 1 is the normal one
     seed 5001       this exact seed for this piece
     solo            render only this piece (a bridge brings its two shots)
   On a bridge only:
     hold 5          this bridge holds 5 frames of each neighbour
     hold 22         ... 22 frames of each neighbour, starting 17 frames before the shot's end and drawing
                     that ending itself (carries pace across the join)
   Without it a bridge uses the node's bridge_hold setting. It applies to that one bridge only.

4. Image descriptions, above the first piece.
     image 1: <Subject 1> is the clown woman from the multi-view sheet <Picture>.
       keep: her appearance and clothing from <Picture>
     image 2, 3, 4: <Subject 2> is the lecture room in <Picture>.
     start: image 1, image 2
   For each piece the node writes one subject_definitions line and one retention_analysis line per described
   image that piece has, with the picture numbers filled in. An image with no description is just handed
   over. "keep:" left off = "as shown in <Picture N>". "start:" left off = every loaded picture.

<Picture N> typed by hand always means picture_N; it is rewritten to the number H3 will know that picture by
in each piece. A tag for a picture a piece doesn't have is reported in the plan and left alone.
"""
import re

FPS = 24
MIN_LEN = 22

MARK = re.compile(r"(?m)(?:^[ \t]*|(?<=[ \t]))([>~])(?=[ \t\r\n]|$)|^[ \t]*([>~])")
SETTINGS = [
    ("seconds", re.compile(r"\[?(\d+(?:\.\d+)?)s\]?(?=[\s,]|$)")),
    ("frames", re.compile(r"\[?(\d+)f\]?(?=[\s,]|$)")),
    ("image", re.compile(r"\[?([+-]?)image\s+(\d+)\]?(?=[\s,]|$)")),
    ("take", re.compile(r"\[?take\s+(\d+)\]?(?=[\s,]|$)")),
    ("seed", re.compile(r"\[?seed\s+(\d+)\]?(?=[\s,]|$)")),
    ("solo", re.compile(r"\[?solo\]?(?=[\s,]|$)")),
    ("hold", re.compile(r"\[?hold\s+(5|22)\]?(?=[\s,]|$)")),
]
IMAGE_LINE = re.compile(r"(?i)^\s*images?\s+(\d+(?:\s*,\s*(?:images?\s+)?\d+)*)\s*:\s*(.*)$")
KEEP_LINE = re.compile(r"(?i)^\s*keep\s*:\s*(.*)$")
START_LINE = re.compile(r"(?i)^\s*start\s*:\s*(.*)$")


def nearest_length(frames):
    """The closest length H3 can make (17k+5), at least 22."""
    k = max(1, round((frames - 5) / 17))
    return 17 * k + 5


def parse(text):
    """Storyboard text -> {"images": {n: {"desc", "keep"}}, "start": [n] or None, "pieces": [...], "notes": [...]}."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    marks = list(MARK.finditer(text))
    header = text[:marks[0].start()] if marks else text
    notes, images, start = [], {}, None

    cur, last = None, None  # the entries being written, and which of their fields the last line went to
    for line in header.split("\n"):
        if not line.strip():
            continue
        m = IMAGE_LINE.match(line)
        if m:
            cur = {"desc": m.group(2).strip(), "keep": None}
            last = "desc"
            for n in re.findall(r"\d+", m.group(1)):
                images[int(n)] = cur
            continue
        m = KEEP_LINE.match(line)
        if m:
            if cur is None:
                notes.append("a \"keep:\" line with no \"image N:\" line above it was ignored")
            else:
                cur["keep"] = m.group(1).strip()
                last = "keep"
            continue
        m = START_LINE.match(line)
        if m:
            start = [int(n) for n in re.findall(r"\d+", m.group(1))]
            cur = None
            continue
        if cur is not None:  # a description or keep that runs onto another line
            cur[last] = (cur[last] + " " + line.strip()).strip()
        else:
            notes.append(f"storyboard line ignored (it is above the first > or ~ and isn't image, keep or start): "
                         f"{line.strip()[:60]!r}")

    pieces = []
    for j, m in enumerate(marks):
        body = text[m.end():marks[j + 1].start() if j + 1 < len(marks) else len(text)]
        p = {"kind": "shot" if (m.group(1) or m.group(2)) == ">" else "bridge", "add": [], "remove": []}
        while True:
            body = body.lstrip(" \t\r\n,")
            for key, rx in SETTINGS:
                s = rx.match(body)
                if not s:
                    continue
                if key == "image":
                    p["remove" if s.group(1) == "-" else "add"].append(int(s.group(2)))
                elif key == "solo":
                    p["solo"] = True
                elif key == "hold":
                    p["hold"] = [22, 17] if s.group(1) == "22" else [5, 0]
                elif key == "seconds":
                    p["seconds"] = float(s.group(1))
                else:
                    p[key] = int(s.group(1))
                body = body[s.end():]
                break
            else:
                break
        p["text"] = body.strip()
        pieces.append(p)
    return {"images": images, "start": start, "pieces": pieces, "notes": notes}


def _fill_beat(prompt, beat):
    """Put the beat where {beat} is. With no beat, {beat} goes and so does any line it leaves empty."""
    if "{beat}" not in prompt:
        if not beat:
            return prompt
        return (prompt + "\n" + beat) if prompt else beat
    if beat:
        return prompt.replace("{beat}", beat)
    out = []
    for line in prompt.split("\n"):
        if "{beat}" in line:
            line = re.sub(r"[ \t]{2,}", " ", line.replace("{beat}", "")).rstrip()
            if not line.strip():
                continue
        out.append(line)
    return "\n".join(out)


def _insert_under(text, header, lines, before=None):
    """Put lines under the section called header. If the prompt has no such section one is made: ahead of the
    section `before` if that exists, otherwise at the top."""
    block = "\n".join(lines)
    m = re.search(rf"(?mi)^[ \t]*{re.escape(header)}[ \t]*$", text)
    if m:
        return text[:m.end()] + "\n" + block + text[m.end():]
    made = f"{header}\n{block}"
    b = re.search(rf"(?mi)^[ \t]*{re.escape(before)}[ \t]*$", text) if before else None
    if b:
        return text[:b.start()] + made + "\n\n" + text[b.start():]
    return (made + "\n\n" + text) if text else made


def compose(prompt, beat, have, images, where, notes):
    """The words one piece is sent. have = the picture slots this piece is handed, in order."""
    pos = {slot: k + 1 for k, slot in enumerate(have)}
    text = _fill_beat((prompt or "").replace("\r\n", "\n").strip(), (beat or "").strip())

    def renumber(m):
        n = int(m.group(1))
        if n in pos:
            return f"<Picture {pos[n]}>"
        notes.append(f"{where}: the words mention <Picture {n}> and this piece isn't handed picture_{n}; "
                     "the tag was left as typed")
        return m.group(0)

    text = re.sub(r"<Picture (\d+)>", renumber, text)
    if "<Picture>" in text:
        notes.append(f"{where}: a <Picture> with no number is in the prompt or the piece's text; numbers are "
                     "only filled in inside \"image N:\" descriptions")

    defs, keeps = [], []
    for slot in have:
        d = images.get(slot)
        if not d or not d["desc"]:
            continue
        tag = f"<Picture {pos[slot]}>"
        defs.append(d["desc"].replace("<Picture>", tag))
        subject = re.search(r"<Subject \d+>", d["desc"])
        kept = (d["keep"] or "").replace("<Picture>", tag).strip()
        if kept.startswith("<"):  # a whole retention line typed by hand
            line = kept
        elif subject:
            line = f"{subject.group(0)} (throughout [Shot 1]): fully_preserved - {kept or 'as shown in ' + tag}"
        else:
            if kept:
                notes.append(f"{where}: image {slot} has a keep line and no <Subject N> in its description to hang "
                             "it on; no retention line was written for it")
            continue
        keeps.append(line if line[-1] in ".!?" else line + ".")
    if defs:
        text = _insert_under(text, "subject_definitions:", defs)
    if keeps:
        if re.search(r"(?mi)^[ \t]*retention_analysis:[ \t]*$", text):
            text = _insert_under(text, "retention_analysis:", keeps)
        elif re.search(r"(?mi)^[ \t]*detailed_description:[ \t]*$", text):
            text = _insert_under(text, "retention_analysis:", keeps, before="detailed_description:")
        else:  # no sections of its own: straight after the definitions
            end = text.index(defs[-1]) + len(defs[-1])
            text = text[:end] + "\n\nretention_analysis:\n" + "\n".join(keeps) + text[end:]
    return text.strip()


def build(storyboard, prompt, shots, seed, loaded, has_video=False, has_audio=False, beat=""):
    """Everything the node was given -> {"pieces": timeline pieces, "solo": piece index or None, "notes": [...]}.
    loaded = the picture slot numbers (1-9) that have a picture plugged in."""
    sb = parse(storyboard)
    notes, images = sb["notes"], sb["images"]
    raw = sb["pieces"]
    if not raw:  # no pieces typed: the same words for every piece
        raw = []
        for k in range(max(1, int(shots))):
            if k:
                raw.append({"kind": "bridge", "add": [], "remove": [], "text": ""})
            raw.append({"kind": "shot", "add": [], "remove": [], "text": ""})
    for i, p in enumerate(raw):
        want = "shot" if i % 2 == 0 else "bridge"
        if p["kind"] != want:
            sym = {"shot": ">", "bridge": "~"}
            raise ValueError(
                f"storyboard, piece {i + 1}: a {p['kind']} ({sym[p['kind']]}) where a {want} ({sym[want]}) belongs. "
                "Pieces go shot, bridge, shot ... and start and end on a shot. It begins: "
                f"{p['text'][:50]!r}")
    if len(raw) % 2 == 0:
        raise ValueError("storyboard: it ends on a bridge (~). A bridge needs a shot (>) after it to arrive at.")
    if not (prompt or "").strip() and not (beat or "").strip() and not any(p["text"] for p in raw) and not images:
        raise ValueError("there is nothing to send H3: the prompt box and the storyboard are both empty")

    have = set(loaded)
    if sb["start"] is not None:
        missing = [n for n in sb["start"] if n not in have]
        if missing:
            notes.append("start: names " + ", ".join(f"image {n}" for n in missing) + " and nothing is plugged into "
                         + ("that slot" if len(missing) == 1 else "those slots"))
        have = {n for n in sb["start"] if n in have}
    for n in sorted(images):
        if n not in loaded:
            notes.append(f"image {n} has a description and nothing is plugged into picture_{n}")

    pieces, solo = [], None
    for i, p in enumerate(raw):
        where = f"shot {i // 2 + 1}" if i % 2 == 0 else f"bridge {(i + 1) // 2}"
        for n in p["add"]:
            if n in loaded:
                have.add(n)
            else:
                notes.append(f"{where}: image {n} asked for and nothing is plugged into picture_{n}")
        for n in p["remove"]:
            have.discard(n)
        mine = sorted(have)
        if len(mine) > 9:
            mine = mine[:9]
        out = {"kind": p["kind"], "final": True,
               "prompt": compose(prompt, p["text"] or beat, mine, images, where, notes),
               "media": [f"picture_{loaded.index(n) + 1}" for n in mine]
                        + ["video"] * bool(has_video) + ["audio"] * bool(has_audio)}
        if not out["prompt"]:
            notes.append(f"{where} has no words (nothing after its marker and no prompt in the prompt box); H3 gets an empty prompt")
        if "frames" in p:
            out["frames"] = max(MIN_LEN, p["frames"])
        elif "seconds" in p:
            out["frames"] = nearest_length(max(MIN_LEN, round(p["seconds"] * FPS)))
        if "seed" in p:
            out["seed"] = p["seed"]
        elif p.get("take", 1) > 1:
            base = seed + i // 2 if i % 2 == 0 else seed + 50000 + (i - 1) // 2
            out["seed"] = base + (p["take"] - 1) * 1000
        if "hold" in p:
            if p["kind"] == "bridge":
                out["hold"] = p["hold"]
            else:
                notes.append(f"{where}: \"hold\" is a bridge setting (it says how a ~ line grips its two shots); "
                             "on a shot it does nothing")
        if p.get("solo"):
            if solo is None:
                solo = i
            else:
                notes.append(f"{where}: a second \"solo\"; only the first one counts")
        pieces.append(out)
    return {"pieces": pieces, "solo": solo, "notes": notes}
