"""Start a new Royale run from a take of an old one (the results page's "Upscale" and "Royale from this").

Every run saves the Comfy job it was started with as job.json in its folder: the whole graph, so the pictures,
text encoder and VAEs that were wired into the node come along. build_job() takes that saved job and one take
of the run, and returns the same job with the Royale node's own settings replaced by that take's settings, so
that take becomes arm 0 (the baseline) of the new run. Its pieces are in the piece cache, so the baseline costs
almost nothing and only the new arms render.

No Comfy imports here, so it can be tested on its own.
"""
import copy
import json
import re

SLOT = re.compile(r"\{([a-z])(?::([^{}]*))?\}")  # the same slots as rolling_royale
LORA_SLOTS = 4
ASPECTS = {"1:1 (Square)": (1, 1), "2:3 (Portrait Photo)": (2, 3), "3:2 (Photo)": (3, 2), "3:4 (Portrait Standard)": (3, 4),
           "4:3 (Standard)": (4, 3), "9:16 (Portrait Widescreen)": (9, 16), "16:9 (Widescreen)": (16, 9),
           "21:9 (Ultrawide)": (21, 9)}  # the same list as rolling_royale


def fill_slots(text, values):
    def sub(m):
        if m.group(1) in values:
            return values[m.group(1)]
        return (m.group(2) or "").strip()
    return SLOT.sub(sub, text or "")


def build_job(job, manifest, arm_n, take_i, what, arms_text="", mode="run"):
    """-> (graph, extra_pnginfo). what: 'upscale' or 'royale'. Raises ValueError with words for the page."""
    graph = copy.deepcopy(job["prompt"])
    nid = str(job["node_id"])
    if nid not in graph or graph[nid].get("class_type") != "H3RollingRoyale":
        raise ValueError("the saved job has no Royale node in it")
    arm = next((a for a in manifest["arms"] if a["n"] == arm_n), None)
    if arm is None or not 0 <= take_i < len(arm["takes"]):
        raise ValueError("no such take in this run")
    take, S = arm["takes"][take_i], arm["settings"]
    if take.get("status") != "done":
        raise ValueError("that take isn't finished")
    if what == "upscale":
        if S.get("upscale"):
            raise ValueError("that take was already rendered with the upscale pass")
        arms_text = "upscale = on"
    elif what == "royale":
        arms_text = (arms_text or "").strip()
        if not arms_text:
            raise ValueError("the arms box is empty: say what the new arms change")
    else:
        raise ValueError(f"don't know how to start '{what}'")

    i = graph[nid]["inputs"]
    i["model"] = S["model"]
    loras = list(S.get("loras") or [])
    for k in range(1, LORA_SLOTS + 1):
        l = loras[k - 1] if k <= len(loras) else {"file": "none", "strength": 1.0, "on": False}
        i[f"lora_{k}"], i[f"lora_{k}_strength"], i[f"lora_{k}_on"] = l["file"] or "none", float(l["strength"]), bool(l["on"])
    i["steps"], i["sampler_name"], i["scheduler"] = int(S["steps"]), S["sampler"], S["scheduler"]
    # the slots are written into the words, so the take's own wording is the new baseline
    slots = S.get("slots") or {}
    i["prompt"] = fill_slots(manifest.get("prompt", ""), slots)
    i["storyboard"] = fill_slots(manifest.get("storyboard", ""), slots)
    i["arms"] = arms_text
    i["shot_seconds"], i["bridge_seconds"] = float(S["shot_seconds"]), float(S["bridge_seconds"])
    i["bridge_hold"] = f"{int(S['hold'])} frames"
    i["seed"], i["seeds_per_arm"] = int(take["seed"]), 1
    i["ref_image_size"] = S["ref_image_size"]
    i["upscale_pass"] = bool(S.get("upscale"))
    i["upscale_megapixels"], i["upscale_steps"], i["upscale_denoise"] = float(S["upscale_mp"]), int(S["upscale_steps"]), float(S["upscale_denoise"])
    i["keystone"] = bool(S.get("keystone", False))
    i["attention"] = S.get("attention", "default")
    i["shift"], i["audio_shift"] = float(S.get("shift", 12.0)), float(S.get("audio_shift", 3.0))
    i["mode"] = "plan only" if mode == "plan" else "run"
    base = {}
    if S.get("refmods"):  # the take's RefMod weights, handed over beside the job: the stack in it has the old baseline's
        base["refmods"] = S["refmods"]
    if S.get("aspect") and not S.get("exact"):
        i["aspect_ratio"], i["megapixels"] = S["aspect"], float(S["mp"])
    else:  # an exact size (or a run from before the node had aspect + megapixels): handed over beside the job
        w, h = int(S["width"]), int(S["height"])
        base["exact"] = [w, h]
        # and the two fields set to the nearest match, so an arm that says mp = or aspect = starts from this shape
        i["aspect_ratio"] = min(ASPECTS, key=lambda k: abs(ASPECTS[k][0] / ASPECTS[k][1] - w / h))
        i["megapixels"] = max(0.1, round(w * h / (1024 * 1024), 2))
    diff = arm.get("diff") or []
    origin = {"run": manifest["id"], "arm": arm_n, "take": take_i, "seed": int(take["seed"]), "what": what,
              "label": "baseline" if arm_n == 0 else " | ".join(diff), "file": take.get("file")}
    return graph, {"h3royale_origin": origin, "h3royale_base": base}


if __name__ == "__main__":
    import sys
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    man = json.load(open(sys.argv[2], encoding="utf-8"))
    g, extra = build_job(job, man, int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else "")
    node = g[str(job["node_id"])]["inputs"]
    print(json.dumps({k: v for k, v in node.items() if k not in ("prompt", "storyboard") and not isinstance(v, list)}, indent=1))
    print(json.dumps(extra, indent=1))
