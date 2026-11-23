"""H3 Rolling Royale: one scene rendered several ways, for comparing settings.

The node's own widgets are the BASELINE (arm 0). The arms box lists the other arms, one per line, and each
line only says what is different from the baseline. Every arm is rendered with the same seed(s) by the
Timeline's own run (rolling_timeline.run_timeline), so shots and bridges, the storyboard box and the piece
cache all behave exactly as they do there.

The arms box
  one arm per line; several changes on a line are separated by  ;
      hold = 22
      lora 2 = on; lora 1 = off; steps = 8
  a sweep: several values separated by  |  make one arm each
      lora 1 strength = 0.6 | 0.8 | 1.0
  two sweeps on one line need  cross:  in front and run every combination
      cross: steps = 4 | 6 | 8; lora 1 strength = 0.8 | 1.0
  control  = the baseline again with the piece cache off (how much two identical runs differ)
  lines starting with # are ignored

What a line can change
  model = <name or a unique part of it>
  lora N = on | off | <strength>          lora N strength = 0.8          lora N file = <name or part>
  steps = 8        sampler = euler        scheduler = simple
  size = 1024x576  (or width = / height =)
  shot seconds = 7        bridge seconds = 3        hold = 5 | 22
  upscale = on | off      upscale mp = 1.5     upscale steps = 2     upscale denoise = 0.4
  pictures = match | max
  keystone = on | off                     (shot 1's first frame locked into frame 0 of every later shot)
  seeds = 3                               (this arm's own number of seeds)
  a = some words                          (what goes where {a} is in the prompt or storyboard; b, c ... too)

Slots: {a} in the prompt or the storyboard is a slot. {a: handheld, slight sway} gives it the words the
baseline uses. An arm that sets a slot is rendered on at least 3 seeds unless its line says seeds = N,
because new words on one seed are a different roll, not the same take nudged.

Each run gets a folder, output/h3_royale/<time>/, with one mp4 per take and royale.json (rewritten after
every take, so a page reading it fills in while the run is going).
"""
import copy
import gc
import itertools
import json
import os
import re
import statistics
import time

from . import rolling_core, rolling_timeline
from . import storyboard as storyboard_mod

TAG = "[H3 Rolling Royale]"
SLOT = re.compile(r"\{([a-z])(?::([^{}]*))?\}")
HOLDS = {5: (5, 0), 22: (22, 17)}
SEED_STEP = 1000      # take k of an arm uses seed + 1000 * k (Seed Hunter's spacing)
SLOT_SEEDS = 3
BASE_PIXELS = 864 * 480
BASE_FRAMES = 175
K_DEFAULT = 2.65      # seconds per step at 864x480, 175 frames, pictures only (bench, 10-15)
PIECE_OVERHEAD = 4.0  # encode + decode per piece, roughly


# ---------------------------------------------------------------- slots

def slots_in(*texts):
    """{letter: baseline words} for every slot in the texts."""
    found = {}
    for t in texts:
        for m in SLOT.finditer(t or ""):
            words = (m.group(2) or "").strip()
            if m.group(1) not in found or (words and not found[m.group(1)]):
                found[m.group(1)] = words
    return found


def fill_slots(text, values):
    def sub(m):
        if m.group(1) in values:
            return values[m.group(1)]
        return (m.group(2) or "").strip()
    return SLOT.sub(sub, text or "")


# ---------------------------------------------------------------- the arms box

def parse_arms(text):
    """-> list of {"label", "changes": [(key, value)], "control": bool, "line": n}."""
    arms = []
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cross = False
        if line.lower().startswith("cross:"):
            cross, line = True, line[6:].strip()
        parts = [c.strip() for c in line.split(";") if c.strip()]
        control, fixed, sweeps = False, [], []
        for part in parts:
            if part.lower() == "control":
                control = True
                continue
            key, eq, val = part.partition("=")
            if not eq:
                raise ValueError(f"arms line {n}: \"{part}\" isn't  setting = value  (or the word control)")
            key = " ".join(key.lower().split())
            vals = ["" if v.strip().lower() == "(empty)" else v.strip() for v in val.split("|")]
            if len(vals) > 1:
                sweeps.append((key, vals))
            else:
                fixed.append((key, vals[0]))
        if len(sweeps) > 1 and not cross:
            raise ValueError(f"arms line {n} has {len(sweeps)} sweeps. Start the line with  cross:  to run every "
                             f"combination ({' x '.join(str(len(v)) for _, v in sweeps)} arms), or put them on separate lines.")
        if control and (fixed or sweeps):
            raise ValueError(f"arms line {n}: control is the baseline as it is; it can't also change something")
        combos = itertools.product(*[[(k, v) for v in vals] for k, vals in sweeps]) if sweeps else [()]
        for combo in combos:
            changes = fixed + list(combo)
            label = "control" if control else "; ".join(f"{k} = {v if v != '' else '(empty)'}" for k, v in changes)
            arms.append({"label": label, "changes": changes, "control": control, "line": n})
    return arms


def _match(value, names, what):
    if value in names:
        return value
    low = value.lower()
    hits = [x for x in names if low in x.lower()]
    exact = [x for x in hits if os.path.splitext(os.path.basename(x))[0].lower() == low]
    if len(exact) == 1:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no {what} matches \"{value}\"")
    raise ValueError(f"\"{value}\" matches {len(hits)} {what}s: {', '.join(hits[:8])}. Give more of the name.")


def _onoff(v):
    v = v.lower()
    if v in ("on", "true", "yes", "1"):
        return True
    if v in ("off", "false", "no", "0"):
        return False
    raise ValueError(f"\"{v}\" isn't on or off")


def apply_change(S, key, value, unets, loras, slots):
    """Change one setting in S (in place). Raises ValueError with what was wrong."""
    m = re.fullmatch(r"lora (\d)(?: (strength|file))?", key)
    if m:
        i = int(m.group(1)) - 1
        if not 0 <= i < len(S["loras"]):
            raise ValueError(f"there are {len(S['loras'])} LoRA slots")
        slot = S["loras"][i]
        if m.group(2) == "strength":
            slot["strength"] = float(value)
        elif m.group(2) == "file":
            slot["file"] = _match(value, loras, "LoRA")
        else:
            try:
                slot["strength"], slot["on"] = float(value), True
            except ValueError:
                slot["on"] = _onoff(value)
        if slot["on"] and slot["file"] in ("", "none"):
            raise ValueError(f"lora {i + 1} is on and has no file (pick one on the node, or  lora {i + 1} file = ...)")
        return
    if key == "model":
        S["model"] = _match(value, unets, "model")
    elif key == "steps":
        S["steps"] = int(value)
    elif key in ("sampler", "sampler name"):
        import comfy.samplers
        S["sampler"] = _match(value, list(comfy.samplers.KSampler.SAMPLERS), "sampler")
    elif key == "scheduler":
        import comfy.samplers
        S["scheduler"] = _match(value, list(comfy.samplers.KSampler.SCHEDULERS), "scheduler")
    elif key in ("size", "resolution"):
        mm = re.fullmatch(r"(\d+)\s*x\s*(\d+)", value.lower())
        if not mm:
            raise ValueError("size is written 1024x576")
        S["width"], S["height"] = int(mm.group(1)), int(mm.group(2))
    elif key in ("width", "height"):
        S[key] = int(value)
    elif key in ("shot seconds", "shot"):
        S["shot_seconds"] = float(value.rstrip("s"))
    elif key in ("bridge seconds", "bridge"):
        S["bridge_seconds"] = float(value.rstrip("s"))
    elif key in ("hold", "bridge hold"):
        h = int(value.split()[0])
        if h not in HOLDS:
            raise ValueError("hold is 5 or 22")
        S["hold"] = h
    elif key in ("upscale", "upscale pass"):
        S["upscale"] = _onoff(value)
    elif key in ("upscale mp", "upscale megapixels"):
        S["upscale_mp"] = float(value)
    elif key == "upscale steps":
        S["upscale_steps"] = int(value)
    elif key == "upscale denoise":
        S["upscale_denoise"] = float(value)
    elif key in ("pictures", "ref image size"):
        if value not in ("match", "max"):
            raise ValueError("pictures is match or max")
        S["ref_image_size"] = value
    elif key == "keystone":
        S["keystone"] = _onoff(value)
    elif key == "seeds":
        S["seeds"], S["seeds_set"] = max(1, int(value)), True
    elif re.fullmatch(r"[a-z]", key):
        if key not in slots:
            raise ValueError(f"the prompt and storyboard have no {{{key}}} slot")
        S["slots"][key] = value
        S["slot_changed"] = True
    else:
        raise ValueError(f"\"{key}\" isn't a setting an arm can change")


def chain_of(S):
    return (S["model"], tuple((l["file"], round(float(l["strength"]), 4)) for l in S["loras"]
                              if l["on"] and l["file"] not in ("", "none")))


def diff_lines(S, base):
    """What differs from the baseline, as short words for a column header."""
    out = []
    if S["model"] != base["model"]:
        out.append("model " + os.path.splitext(os.path.basename(S["model"]))[0])
    for i, (a, b) in enumerate(zip(S["loras"], base["loras"]), 1):
        an = a["on"] and a["file"] not in ("", "none")
        bn = b["on"] and b["file"] not in ("", "none")
        nm = os.path.splitext(os.path.basename(a["file"]))[0]
        if an and not bn:
            out.append(f"+ {nm} @ {a['strength']:g}")
        elif bn and not an:
            out.append(f"- {os.path.splitext(os.path.basename(b['file']))[0]}")
        elif an and bn and a["file"] != b["file"]:
            out.append(f"lora {i}: {nm} @ {a['strength']:g}")
        elif an and bn and float(a["strength"]) != float(b["strength"]):
            out.append(f"{nm} @ {a['strength']:g}")
    for key, word in (("steps", "steps"), ("sampler", ""), ("scheduler", ""), ("hold", "hold"),
                      ("shot_seconds", "shots"), ("bridge_seconds", "bridges"), ("ref_image_size", "pictures"),
                      ("upscale_mp", "upscale MP"), ("upscale_steps", "upscale steps"), ("upscale_denoise", "upscale denoise")):
        if S[key] != base[key]:
            unit = "s" if key.endswith("seconds") else ""
            out.append(f"{word} {S[key]:g}{unit}".strip() if isinstance(S[key], float) else f"{word} {S[key]}{unit}".strip())
    if (S["width"], S["height"]) != (base["width"], base["height"]):
        out.append(f"{S['width']}x{S['height']}")
    if S["upscale"] != base["upscale"]:
        out.append("upscale on" if S["upscale"] else "upscale off")
    if S.get("keystone", False) != base.get("keystone", False):
        out.append("keystone on" if S.get("keystone") else "keystone off")
    for k in sorted(S["slots"]):
        if S["slots"][k] != base["slots"].get(k, ""):
            out.append(f"{{{k}}}: {S['slots'][k] or '(empty)'}")
    return out


# ---------------------------------------------------------------- time estimate

def _timing_file():
    return os.path.join(os.path.dirname(__file__), "royale_timing.json")


def _k():
    """Seconds per step at the bench size, from this machine's own Royale takes if there are any."""
    try:
        with open(_timing_file(), encoding="utf-8") as f:
            ks = json.load(f)["k"][-20:]
        if ks:
            return statistics.median(ks), len(ks)
    except (OSError, ValueError, KeyError):
        pass
    return K_DEFAULT, 0


def _units(S, frames_list):
    """Work in one take: sum over its pieces of steps x size x length, in bench units."""
    px = (S["width"] // 32 * 32) * (S["height"] // 32 * 32) / BASE_PIXELS
    u = sum(S["steps"] * px * f / BASE_FRAMES for f in frames_list)
    if S["upscale"]:
        u += sum(S["upscale_steps"] * (S["upscale_mp"] * 1024 * 1024 / BASE_PIXELS) * f / BASE_FRAMES for f in frames_list)
    return u


def _remember(S, frames_list, seconds):
    u = _units(S, frames_list)
    if u <= 0:
        return
    k = (seconds - PIECE_OVERHEAD * len(frames_list)) / u
    if not 0.2 < k < 60:
        return
    try:
        with open(_timing_file(), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {"k": []}
    d["k"] = (d.get("k", []) + [round(k, 3)])[-50:]
    with open(_timing_file(), "w", encoding="utf-8") as f:
        json.dump(d, f)


def _hms(s):
    s = int(round(s))
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


# ---------------------------------------------------------------- the run

def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def run_royale(clip, vae, audio_vae, prompt, storyboard, arms_text, base, shots, seed, soundtrack, frame_labels,
               mode, pictures, ref_video=None, ref_audio=None):
    import comfy.model_management as mm
    import folder_paths
    import nodes

    unets = folder_paths.get_filename_list("diffusion_models")
    loras = folder_paths.get_filename_list("loras")
    slots = slots_in(prompt, storyboard)
    base = copy.deepcopy(base)
    base.setdefault("keystone", False)
    base.update({"slots": dict(slots), "slot_changed": False, "seeds_set": False})
    for i, l in enumerate(base["loras"], 1):
        if l["on"] and l["file"] in ("", "none"):
            l["on"] = False

    # ---- the arms: baseline first, then the box
    arms = [{"n": 0, "label": "baseline", "changes": [], "control": False, "S": base}]
    for a in parse_arms(arms_text):
        S = copy.deepcopy(base)
        try:
            for key, value in a["changes"]:
                apply_change(S, key, value, unets, loras, slots)
        except ValueError as e:
            raise ValueError(f"arms line {a['line']} ({a['label']}): {e}")
        if S["slot_changed"] and not S["seeds_set"]:
            S["seeds"] = max(S["seeds"], SLOT_SEEDS)
        arms.append({"n": len(arms), "label": a["label"], "changes": a["changes"], "control": a["control"], "S": S})
    for a in arms:
        a["diff"] = ["control (cache off)"] if a["control"] else diff_lines(a["S"], base)
        a["chain"] = chain_of(a["S"])
    seen = {}
    for a in arms[1:]:
        if not a["control"] and not a["diff"]:
            raise ValueError(f"arm \"{a['label']}\" is the same as the baseline. Use the word  control  for a repeat.")
        sig = json.dumps([a["S"], a["control"]], sort_keys=True, default=str)
        if sig in seen and not a["control"]:
            raise ValueError(f"arm \"{a['label']}\" is the same as arm \"{seen[sig]}\"")
        seen[sig] = a["label"]

    # ---- pieces of each arm (text only), for the plan and the estimate
    loaded = [i for i in range(1, 10) if pictures[i - 1] is not None]

    def timeline_for(S, take_seed):
        p_text, s_text = fill_slots(prompt, S["slots"]), fill_slots(storyboard, S["slots"]).strip()
        if s_text[:1] in ("[", "{"):
            return s_text, p_text, None, []
        built = storyboard_mod.build(s_text, p_text, shots, take_seed, loaded, ref_video is not None,
                                     ref_audio is not None, "")
        return {"pieces": built["pieces"]}, p_text, built["solo"], built["notes"]

    k, k_n = _k()
    takes_total, est_total, rows = 0, 0.0, []
    for a in arms:
        S = a["S"]
        tl, p_text, _, _ = timeline_for(S, seed)
        pieces, _ = rolling_timeline.parse_timeline(tl, p_text, "", round(S["shot_seconds"] * rolling_core.FPS),
                                                    round(S["bridge_seconds"] * rolling_core.FPS))
        a["frames_list"] = [p["frames"] + (17 if (i % 2 == 0 and i > 0) else 0) for i, p in enumerate(pieces)]
        a["est"] = _units(S, a["frames_list"]) * k + PIECE_OVERHEAD * len(pieces)
        takes_total += S["seeds"]
        est_total += a["est"] * S["seeds"]
        rows.append(f"  arm {a['n']}: {' | '.join(a['diff']) or 'baseline'}  -  {S['seeds']} seed(s), "
                    f"{len(pieces)} pieces, up to ~{_hms(a['est'])} a take")

    # ---- order: baseline, then arms grouped by model + LoRA chain so each chain loads once
    order, chains = [arms[0]], [arms[0]["chain"]]
    for a in arms[1:]:
        if a["chain"] not in chains:
            chains.append(a["chain"])
    first_of = {}
    for i, c in enumerate(chains):
        first_of.setdefault(c[0], i)
    for ch in sorted(chains, key=lambda c: (first_of[c[0]], chains.index(c))):
        order += [a for a in arms[1:] if a["chain"] == ch]
    plan = (f"{len(arms)} arms, {takes_total} takes, {len(chains)} model chain(s). Rough ceiling ~{_hms(est_total)} "
            f"({'from ' + str(k_n) + ' earlier takes here' if k_n else 'from the October bench numbers; it tunes itself after the first run'}; "
            f"pieces two arms share render once, so the real time is usually less).\n" + "\n".join(rows))
    if slots:
        plan += "\n  slots: " + "; ".join(f"{{{s}}} = {v or '(empty)'}" for s, v in sorted(slots.items()))
    print(f"{TAG} {plan}")
    if mode == "plan only":
        return plan, ""

    # ---- the run folder and its manifest
    run_id = time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(folder_paths.get_output_directory(), "h3_royale", run_id)
    os.makedirs(folder, exist_ok=True)
    man = {"id": run_id, "status": "running", "started": time.time(), "fps": rolling_core.FPS, "seed": seed,
           "prompt": prompt, "storyboard": storyboard, "arms_box": arms_text, "plan": plan,
           "baseline": {kk: v for kk, v in base.items() if kk not in ("slot_changed", "seeds_set")},
           "arms": [{"n": a["n"], "label": a["label"], "diff": a["diff"], "control": a["control"],
                     "settings": {kk: v for kk, v in a["S"].items() if kk not in ("slot_changed", "seeds_set")},
                     "takes": [{"seed": seed + SEED_STEP * t, "status": "waiting"} for t in range(a["S"]["seeds"])]}
                    for a in arms]}
    mpath = os.path.join(folder, "royale.json")
    _write(mpath, man)

    model_cache = {"unet": None, "base": None, "chain": None, "model": None}

    def model_for(S):
        ch = chain_of(S)
        if model_cache["chain"] == ch:
            return model_cache["model"]
        # a different chain: let go of the patched copy before the next one loads (two patched copies of a
        # 20 GB model on the card at once is the 3x slowdown from the October bench)
        model_cache["model"] = model_cache["chain"] = None
        if model_cache["unet"] != S["model"]:
            model_cache["base"] = None
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if model_cache["base"] is None:
            print(f"{TAG} loading {S['model']}")
            model_cache["base"] = nodes.UNETLoader().load_unet(S["model"], "default")[0]
            model_cache["unet"] = S["model"]
        m = model_cache["base"]
        for file, strength in ch[1]:
            m = nodes.LoraLoaderModelOnly().load_lora_model_only(m, file, strength)[0]
        model_cache["chain"], model_cache["model"] = ch, m
        return m

    done = 0
    try:
        for a in order:
            S, entry = a["S"], man["arms"][a["n"]]
            for t in range(S["seeds"]):
                take = entry["takes"][t]
                take_seed = seed + SEED_STEP * t
                name = f"arm{a['n']:02d}_seed{t + 1}.mp4"
                take.update({"status": "rendering", "file": name})
                _write(mpath, man)
                print(f"{TAG} take {done + 1}/{takes_total}: arm {a['n']} ({' | '.join(a['diff']) or 'baseline'}), seed {take_seed}")
                t0 = time.time()
                rep = {}
                try:
                    model = model_for(S)
                    tl, p_text, solo, notes = timeline_for(S, take_seed)
                    frames, _, take_plan = rolling_timeline.run_timeline(
                        model, clip, vae, audio_vae, tl, p_text, "", round(S["shot_seconds"] * rolling_core.FPS),
                        round(S["bridge_seconds"] * rolling_core.FPS), S["width"], S["height"], S["steps"],
                        S["sampler"], S["scheduler"], take_seed, S["ref_image_size"], pin_mode="mask",
                        soundtrack=soundtrack, upscale_pass=S["upscale"], upscale_megapixels=S["upscale_mp"],
                        upscale_steps=S["upscale_steps"], upscale_denoise=S["upscale_denoise"],
                        frame_labels=frame_labels, pictures=pictures, ref_video=ref_video, ref_audio=ref_audio,
                        solo=solo, extra_notes=notes, hold=HOLDS[S["hold"]],
                        piece_cache=False if a["control"] else None, stream_to=os.path.join(folder, name), report=rep,
                        keystone=S["keystone"])
                    del frames
                    secs = time.time() - t0
                    used = rep.get("used", {})
                    take.update({"status": "done", "seconds": round(secs, 1), "frames": rep.get("frames"),
                                 "width": rep.get("width"), "height": rep.get("height"), "pieces": rep.get("pieces"),
                                 "rendered": len(used.get("rendered", [])), "upscaled": len(used.get("upscaled", [])),
                                 "cached": len(used.get("cached", [])), "plan": take_plan})
                    if not used.get("cached") and not used.get("upscaled") and rep.get("pieces"):
                        _remember(S, a["frames_list"], secs)
                except mm.InterruptProcessingException:
                    take.update({"status": "stopped"})
                    raise
                except Exception as e:
                    take.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)})
                    print(f"{TAG} arm {a['n']} seed {take_seed} failed: {type(e).__name__}: {e}")
                done += 1
                _write(mpath, man)
        man["status"] = "done"
    except mm.InterruptProcessingException:
        man["status"] = "stopped"
        raise
    finally:
        if man["status"] == "running":
            man["status"] = "failed"
        man["finished"] = time.time()
        _write(mpath, man)
        model_cache.clear()

    lines = []
    for a in arms:
        for tk in man["arms"][a["n"]]["takes"]:
            if tk["status"] == "done":
                src = f"{tk['rendered']} rendered" + (f", {tk['cached']} from cache" if tk["cached"] else "") + (
                    f", {tk['upscaled']} upscaled from cache" if tk["upscaled"] else "")
                lines.append(f"  arm {a['n']} seed {tk['seed']}: {_hms(tk['seconds'])} ({src}) -> {tk['file']}")
            else:
                lines.append(f"  arm {a['n']} seed {tk['seed']}: {tk['status']}" + (f" - {tk['error']}" if tk.get("error") else ""))
    report = plan + f"\n\npage: /h3royale?run={run_id} on this Comfy's address\nfolder: {folder}\ntook {_hms(man['finished'] - man['started'])}\n" + "\n".join(lines)
    return report, folder
