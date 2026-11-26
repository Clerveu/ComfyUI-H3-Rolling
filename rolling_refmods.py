"""RefMods in the Timeline and the Royale.

A RefMod is a saved, already-encoded reference from the Fantastic H3 Prompt Builder pack (a look made from
pictures or a clip, a voice, or both under one name). Its RefMod Stack node sends a bundle on an H3_REF_MODS
wire: a list of (mod, strength), strength 0..1, a weight above 1 written as extra rows of the same mod.

When a bundle is wired in, every piece is encoded by that pack's RefMod Text Encode instead of Comfy's own
Reference to Video. It does the same job and also shows the RefMods to H3's text encoder, so the prompt can
name them. The order H3 reads everything in: the pictures, the reference video, the reference audio, then
the RefMods, each kind counted in one sequence. A look made from several pictures is a <Video n>, a look
made from one picture is a <Picture n>, a voice is an <Audio n>.

Here a RefMod is one name with a look weight and a voice weight (either may be missing), which is what an
arm changes:  refmod 1 look = 0.6   refmod 1 voice = 0   refmod mira = off

No Comfy imports at the top, so the arm handling can be tested on its own.
"""
import os
import re

ENCODER = "MiniMaxH3FantasticRefModTextEncode"
PACK = "the Fantastic H3 Prompt Builder node pack (ComfyUI-Fantastic-MiniMaxH3-PromptBuilder)"
MAX_WEIGHT = 10.0  # the stack's own ceiling
CHANNELS = ("look", "voice")
_SUFFIX = re.compile(r"_(visual|audio|Video|Audio)$")
_TAG = {"image": "Picture", "video": "Video", "audio": "Audio"}


def label(mod):
    """The name the library shows: the file's name without its look / voice ending."""
    path = getattr(mod, "path", "") or ""
    return _SUFFIX.sub("", os.path.basename(path)) if path else str(getattr(mod, "name", "?"))


def expand(weight):
    """A weight as the rows the bundle carries: 2.7 -> [1, 1, 0.7]. 0 -> nothing."""
    w = min(MAX_WEIGHT, float(weight or 0))
    if not w > 0:
        return []
    whole = int(w + 1e-9)
    rem = round(w - whole, 4)
    return [1.0] * whole + ([rem] if rem > 1e-6 else [])


def groups(mods):
    """The bundle as RefMods, in the order they sit in the stack:
    [{"name", "look": mod or None, "voice": mod or None, "weights": {"look": w or None, "voice": w or None}}]"""
    if mods is None:
        return []
    if not isinstance(mods, (list, tuple)):
        raise ValueError("what is wired into mods isn't a RefMod bundle (it comes from a RefMod Stack)")
    out, by_name = [], {}
    for entry in mods:
        try:
            mod, strength = entry
            kind, strength = mod.kind, float(strength)
        except (TypeError, ValueError, AttributeError):
            raise ValueError("what is wired into mods isn't a RefMod bundle (it comes from a RefMod Stack)")
        if kind not in _TAG:
            raise ValueError(f"RefMod {label(mod)} is a '{kind}', which H3 has no tag for")
        name = label(mod)
        g = by_name.get(name)
        if g is None:
            g = by_name[name] = {"name": name, "look": None, "voice": None, "weights": {"look": None, "voice": None}}
            out.append(g)
        ch = "voice" if kind == "audio" else "look"
        if g[ch] is None:
            g[ch] = mod
        g["weights"][ch] = round((g["weights"][ch] or 0.0) + strength, 4)
    return out


def state(gs):
    """The part of the groups that is plain data: what an arm changes and what goes in the manifest."""
    return [{"name": g["name"], "look": g["weights"]["look"], "voice": g["weights"]["voice"]} for g in gs]


def rows(gs, st):
    """The bundle to hand the encoder for these weights: look then voice for each RefMod, in stack order."""
    out = []
    for g, s in zip(gs, st):
        for ch in CHANNELS:
            if g[ch] is not None:
                out += [(g[ch], x) for x in expand(s.get(ch))]
    return out


def find(st, which):
    """An arm's 'refmod 2' or 'refmod mira' -> its place in the list."""
    if not st:
        raise ValueError("no RefMods are wired in (the mods input takes a RefMod Stack)")
    if which.isdigit():
        i = int(which) - 1
        if not 0 <= i < len(st):
            raise ValueError(f"the stack has {len(st)} RefMod{'s' if len(st) != 1 else ''}: " + ", ".join(
                f"{n} = {s['name']}" for n, s in enumerate(st, 1)))
        return i
    low = which.lower()
    hits = [i for i, s in enumerate(st) if s["name"].lower() == low] or [i for i, s in enumerate(st) if low in s["name"].lower()]
    if len(hits) != 1:
        raise ValueError(f"\"{which}\" matches {len(hits)} of the RefMods in the stack: " + ", ".join(s["name"] for s in st))
    return hits[0]


def change(st, which, channel, value):
    """One arm change, in place.  channel: 'look', 'voice' or None for both.  value: a weight, on or off."""
    s = st[find(st, which)]
    if channel and s[channel] is None:
        raise ValueError(f"{s['name']} has no {channel}")
    v = value.strip().lower()
    if v in ("on", "off"):
        w = None
    else:
        try:
            w = float(v)
        except ValueError:
            raise ValueError(f"\"{value}\" isn't a weight, on or off")
        if not 0 <= w <= MAX_WEIGHT:
            raise ValueError(f"a RefMod weight is 0 to {MAX_WEIGHT:g} (up to 1 is strength, above 1 adds copies)")
    for ch in ([channel] if channel else CHANNELS):
        if s[ch] is None:
            continue
        if w is not None:
            s[ch] = w
        elif v == "off":
            s[ch] = 0.0
        elif not s[ch]:
            s[ch] = 1.0


def diff(st, base):
    """What differs from the baseline, as short words."""
    out = []
    for a, b in zip(st or [], base or []):
        ch = [c for c in CHANNELS if a[c] != b[c]]
        word = lambda c: "off" if not a[c] else f"{a[c]:g}"
        has = [c for c in CHANNELS if a[c] is not None]
        if ch == has and len({word(c) for c in ch}) == 1 and (len(ch) == 2 or word(ch[0]) == "off"):
            out.append(f"{a['name']} {word(ch[0])}")
        else:
            out += [f"{a['name']} {c} {word(c)}" for c in ch]
    return out


def tags(bundle, n_pictures=0, has_video=False, has_audio=False):
    """What the prompt calls each RefMod when a piece is handed all the media: the media are numbered first."""
    count = {"image": n_pictures, "video": int(bool(has_video)), "audio": int(bool(has_audio))}
    lines = []
    for mod, _ in bundle:
        count[mod.kind] += 1
        n = count[mod.kind]
        if lines and lines[-1][0] is mod:
            lines[-1][2] = n
        else:
            lines.append([mod, n, n])
    out = []
    for mod, a, b in lines:
        what = "voice" if mod.kind == "audio" else "look"
        out.append((f"<{_TAG[mod.kind]} {a}>" if a == b else f"<{_TAG[mod.kind]} {a}-{b}> ({b - a + 1} copies, cite the first)")
                   + f" = {label(mod)}'s {what}")
    return out


def encoder():
    """The pack's RefMod Text Encode, found in Comfy's own list of nodes."""
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(ENCODER)
    if cls is None:
        raise ValueError(f"RefMods are wired in and {PACK} isn't installed (or didn't load): its RefMod Text Encode does the encoding")
    return cls()
