"""H3 Rolling: render long reference-driven H3 videos as overlapping, guided chunks."""

import importlib
import os

import comfy.samplers
import folder_paths

from . import rolling_core, rolling_refmods, rolling_royale, rolling_timeline

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")

DEFAULT_PROMPT = """subject_definitions:
<Subject 1> is a cat whose appearance comes from <Picture 1>. Her position, posture, actions, and timing come from the skateboarder in <Video 1>.
<Video 1> is the source video for the target video edit.
<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.

summary:
[video editing + audio reuse] The target video is an edited version of <Video 1>. The skateboarder is entirely replaced by <Subject 1>, a real cat, who rides the skateboard and performs every trick in place with the same timing. The camera, framing, location, lighting, and timing of <Video 1> are unchanged, and <Audio 1> is reused as the soundtrack.

retention_analysis:
<Subject 1>: fully_preserved - appearance from <Picture 1>; position, actions, and timing from the skateboarder in <Video 1>.
<Video 1>: partially_preserved - camera, composition, setting, skateboard, lighting, and timing are retained; the skateboarder is fully replaced by <Subject 1>.
<Audio 1>: fully_copy - reused 1:1 as the soundtrack.

detailed_description:
[Shot 1] A single continuous shot from beginning to end. <Subject 1>, a real cat, rides the skateboard exactly where the skateboarder was in <Video 1>, performing every push, ollie, and trick with the same timing as <Video 1>, paws planted on the deck.

overall_soundscape:
The copied ambience from <Audio 1> continues throughout: wheels on pavement, board impacts.

non_diegetic_music:
N/A"""


def _video_files():
    root = folder_paths.get_input_directory()
    found = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                found.append(os.path.relpath(os.path.join(dirpath, f), root).replace("\\", "/"))
    return sorted(found)


def _image_files():
    """Every picture in the input folder, subfolders included (cast/zara.png)."""
    root = folder_paths.get_input_directory()
    found = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                found.append(os.path.relpath(os.path.join(dirpath, f), root).replace("\\", "/"))
    return sorted(found)


EMPTY_PICTURE = "(empty).png"


def _ensure_empty_picture():
    """The empty slot is a real file in the input folder, so the picture widget has something to show
    instead of 'Image failed to load'. The node never loads it."""
    path = os.path.join(folder_paths.get_input_directory(), EMPTY_PICTURE)
    if os.path.exists(path):
        return
    try:
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (512, 512), (21, 23, 28))
        d = ImageDraw.Draw(im)
        d.rectangle([24, 24, 487, 487], outline=(58, 63, 75), width=3)
        d.line([24, 24, 487, 487], fill=(40, 44, 54), width=2)
        d.line([24, 487, 487, 24], fill=(40, 44, 54), width=2)
        im.save(path)
    except Exception as e:
        print(f"[H3 Rolling] couldn't write {EMPTY_PICTURE}: {e}")


def _is_empty(image):
    return not image or image in ("none", EMPTY_PICTURE)


class H3RollingPicture:
    """A picture slot that can be left empty. '(empty).png' hands nothing on, so the node it feeds treats the slot as
    not plugged in; core Load Image always needs a file and stops the run without one."""

    CATEGORY = "MiniMax H3/rolling"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ([EMPTY_PICTURE] + [f for f in _image_files() if f != EMPTY_PICTURE],
                      {"default": EMPTY_PICTURE, "image_upload": True,
                       "tooltip": "(empty).png = this slot is empty and is ignored. Pictures in subfolders of the input folder are listed too."}),
        }}

    @classmethod
    def IS_CHANGED(cls, image):
        if _is_empty(image):
            return "none"
        try:
            return f"{image}|{os.path.getmtime(folder_paths.get_annotated_filepath(image))}"
        except Exception:
            return image

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if _is_empty(image):
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Picture not found in the input folder: {image}"
        return True

    def load(self, image):
        if _is_empty(image):
            return (None,)
        import nodes
        return (nodes.LoadImage().load_image(image)[0],)


class H3RollingSource:
    """Long source video for rolling renders: scrub to set start/end, frames come out at 24 fps."""

    CATEGORY = "MiniMax H3/rolling"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "info")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": (_video_files() or [""], {"tooltip": "Any video in ComfyUI's input folder (use Upload in the player below to add one). Any frame rate; it's resampled to 24 fps."}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01}),
            "end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01,
                                      "tooltip": "0 = play to the end of the file"}),
            "short_edge": ("INT", {"default": 480, "min": 0, "max": 4096, "step": 16,
                                   "tooltip": "Shrink the source so its short edge is this many pixels (0 = keep). It's only choreography, ~480 is plenty and keeps every chunk cheap."}),
        }}

    @classmethod
    def IS_CHANGED(cls, video, start_seconds, end_seconds, short_edge):
        try:
            mtime = os.path.getmtime(folder_paths.get_annotated_filepath(video))
        except Exception:
            mtime = 0
        return f"{video}|{mtime}|{start_seconds}|{end_seconds}|{short_edge}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not video or not folder_paths.exists_annotated_filepath(video):
            return f"Video not found in input folder: {video}"
        return True

    def load(self, video, start_seconds, end_seconds, short_edge):
        importlib.reload(rolling_core)
        return rolling_core.load_source(folder_paths.get_annotated_filepath(video),
                                        start_seconds, end_seconds, short_edge)


class H3RollingRef2VA:
    """Render a long source as overlapping H3 Ref2VA chunks, each pinned to the end of the last."""

    CATEGORY = "MiniMax H3/rolling"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "plan")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "Video VAE"}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE"}),
                "frames": ("IMAGE", {"tooltip": "Source frames at 24 fps (from H3 Rolling Source). Each chunk sees its own slice as <Video 1>."}),
                "prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT,
                                      "tooltip": "One prompt for every chunk. Tags: <Picture N> for pictures, <Video 1> for the source slice, <Audio 1> for its soundtrack (only if source audio is used as reference). With dialogue on, {dialogue} marks where each piece's own lines go."}),
                "megapixels": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 2.0, "step": 0.05,
                                         "tooltip": "Output size; aspect comes from the source. 0.4 = 864x480 at 16:9."}),
                "chunk_seconds": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 15.0, "step": 0.25,
                                            "tooltip": "Length of each rendered piece. Shorter = less VRAM per piece, more seams. H3 is trained on 5-15s."}),
                "overlap_frames": (["5", "22", "39", "56"], {"default": "5",
                                                            "tooltip": "Frames of the previous piece pinned at the start of the next. More = smoother handoff, slower."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "bridge/none: every chunk uses this seed. pixel/latent: chunk N uses seed + N"}),
                "ref_image_size": (["match", "max"], {"default": "match",
                                                      "tooltip": "'max' = 2048px short edge for pictures (sharper identity, slower)"}),
                "soundtrack": (["generated", "source", "silent"], {"default": "generated",
                                                                   "tooltip": "Final audio: H3's generated track (crossfaded at seams), the untouched source audio, or none"}),
                "source_audio_as_reference": ("BOOLEAN", {"default": True,
                                                          "tooltip": "Give each chunk its slice of the source soundtrack as <Audio 1>"}),
                "upscale_pass": ("BOOLEAN", {"default": False,
                                             "tooltip": "Per chunk: latent-upscale with the Minimax H3 3D upscaler, then a short polish pass (Seed Hunter's stage 2). The next chunk is pinned to the upscaled result."}),
                "upscale_megapixels": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 4.0, "step": 0.1}),
                "upscale_steps": ("INT", {"default": 2, "min": 1, "max": 30,
                                          "tooltip": "Polish steps at the big size (Seed Hunter: 2 with the turbo LoRA)"}),
                "upscale_denoise": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05,
                                              "tooltip": "How much the polish pass may repaint"}),
                "handoff": (["bridge", "none", "pixel", "latent"], {"default": "bridge",
                                                                    "tooltip": "How chunks join. bridge: independent chunks, each seam covered by a short bridge pinned to both sides (best for long clips, nothing compounds). none: independent chunks crossfaded over overlap_frames. pixel / latent: each chunk pinned to the end of the last (smooth, but artifacts compound after ~6 hand-offs). overlap_frames only matters for none/pixel/latent."}),
                "bridge_frames": (["22", "39", "56", "73"], {"default": "39",
                                                             "tooltip": "Length of each seam bridge (bridge hand-off only). Longer = gentler blend between takes, more to render."}),
                "structure_lock": ("BOOLEAN", {"default": False,
                                               "tooltip": "SAM3 tracks the person being replaced; an edge drawing of the source with them cut out drives the H3 Fun ControlNet, so framing, camera and zoom follow the source while the person is drawn fresh. Needs sam3.1_multiplex_fp16 (checkpoints) and the Fun ControlNet Union patch (model_patches). ~30% slower sampling plus a few minutes of tracking."}),
                "lock_track": ("STRING", {"default": "person",
                                          "tooltip": "Structure lock: who SAM3 should track, in plain words (\"man\", \"woman in red\")"}),
                "lock_grow": ("INT", {"default": 40, "min": 0, "max": 200, "step": 4,
                                      "tooltip": "Structure lock: pixels of slack around the tracked person, so the new character can be a different shape"}),
                "lock_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                            "tooltip": "Structure lock: how hard the edge drawing holds the scene"}),
                "lock_preview_only": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Structure lock: only track and write output/h3_rolling/lock_preview_*.mp4 (tracked person in red | edge drawing), render nothing. Check it before a long run."}),
                "lock_max_people": ("INT", {"default": 2, "min": 1, "max": 8,
                                            "tooltip": "Structure lock: how many matches of lock_track SAM3 may keep per window. SAM3 finds every match of the description, so with 1 it keeps only the best match; use 1 when there's only one person to replace."}),
                "dialogue": (["off", "on", "preview only"], {"default": "off",
                                                              "tooltip": "Transcribe the source audio (WhisperX, diarized, cached) and give every chunk and bridge only the lines spoken inside its own frames, as <d>[English] ...</d>. They go where {dialogue} is in the prompt, else at the end of detailed_description. 'preview only' transcribes, lists the speaker labels and each piece's lines, and renders nothing. Needs WhisperX (H3_WHISPER_PY) and HF_TOKEN."}),
                "speakers": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "Who each diarization label is, one per line: SPEAKER_00 = <Subject 1>, SPEAKER_01 = Data from <Video 1>. Map two labels to the same text if diarization split one voice. Add (voiceover) for an off-screen voice; '= ignore' leaves a label out. Run 'preview only' first to see the labels."}),
                "pin_mode": (["keyframe", "mask"], {"default": "keyframe",
                                                    "tooltip": "How a bridge holds the frames it shares with its two chunks (bridge hand-off only). keyframe: passed as H3 keyframe guides, and the bridge draws its own copy of them. mask: the chunks' real latents are written into the bridge itself and masked off, so the sampler keeps them and they decode as the originals (roycho87's Clown Context / Clown Anchoring method)."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Source soundtrack (from H3 Rolling Source)"}),
                "sampler": ("SAMPLER", {"tooltip": "Overrides sampler_name (e.g. from a workflow's shared video settings)"}),
                "sigmas": ("SIGMAS", {"tooltip": "Overrides steps + scheduler"}),
                "picture_1": ("IMAGE",),
                "picture_2": ("IMAGE",),
                "picture_3": ("IMAGE",),
                "picture_4": ("IMAGE",),
                "refmod_conditioning": ("CONDITIONING", {"tooltip": "Output of a RefMod Apply node: its RefMod blocks are added to every chunk"}),
            },
        }

    def run(self, model, clip, vae, audio_vae, frames, prompt, megapixels, chunk_seconds, overlap_frames, steps,
            sampler_name, scheduler, seed, ref_image_size, soundtrack, source_audio_as_reference,
            upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4,
            handoff="bridge", bridge_frames="39", structure_lock=False, lock_track="person", lock_grow=40,
            lock_strength=1.0, lock_preview_only=False, lock_max_people=2, dialogue="off", speakers="",
            pin_mode="keyframe", audio=None, sampler=None, sigmas=None, picture_1=None, picture_2=None, picture_3=None, picture_4=None,
            refmod_conditioning=None):
        importlib.reload(rolling_core)
        return rolling_core.run_rolling(
            model, clip, vae, audio_vae, frames, prompt, megapixels, chunk_seconds, overlap_frames, steps,
            sampler_name, scheduler, seed, ref_image_size, soundtrack, source_audio_as_reference,
            audio=audio, pictures=(picture_1, picture_2, picture_3, picture_4),
            sampler=sampler, sigmas=sigmas, refmod_conditioning=refmod_conditioning,
            upscale_pass=upscale_pass, upscale_megapixels=upscale_megapixels,
            upscale_steps=upscale_steps, upscale_denoise=upscale_denoise,
            options={"handoff": handoff, "bridge_frames": int(bridge_frames), "structure_lock": structure_lock,
                     "lock_track": lock_track, "lock_grow": lock_grow, "lock_strength": lock_strength,
                     "lock_preview_only": lock_preview_only, "lock_max_people": lock_max_people,
                     "dialogue": dialogue, "speakers": speakers, "pin_mode": pin_mode})


STORYBOARD_TIP = """Optional. Empty = the prompt above, for every piece ('shots' of them).

>  starts a shot,  ~  starts a bridge. Everything up to the next marker is that piece's text. It goes where {beat} is in the prompt, or on the end of the prompt if there is no {beat}, or is the whole prompt if the prompt box is empty.

Settings are bare lowercase words right after the marker:
  7s or 158f = length    image 3 = start handing picture_3 over    -image 2 = stop handing picture_2 over
  take 2 = another roll of this piece    seed 5001 = this exact seed    solo = render only this piece

Above the first piece, optional:
  image 1: <Subject 1> is the woman in <Picture>.
    keep: her appearance and clothing from <Picture>
  image 2, 3: <Subject 2> is the room in <Picture>.
  start: image 1, image 2
The node then writes the subject_definitions and retention_analysis lines for each piece from the images that piece has, with the picture numbers filled in.

<Picture N> typed by hand always means picture_N. JSON (a list of pieces) is still accepted here."""


KEYSTONE_TIP = ("Shot 1's first frame is locked into the hidden first frame of every later shot, so each shot starts from shot 1's place, framing and pose without another reference picture (roycho87's Geiru Keystone, here on shots rendered on their own). It keeps the place the same from shot to shot. Every later shot opens close to how shot 1 opened, so it suits a scene that stays in one spot; a scene that moves on gets pulled back. The locked frame is inside the run-up and is never shown.")

BRIDGE_HOLDS = {
    "5 frames": (5, 0),
    "22 frames": (22, 17),
}


class H3RollingTimeline:
    """A long H3 video from pictures and prompts, no source video. The video is a list of pieces: shots rendered
    on their own, and bridges pinned between each pair of shots. Nothing is ever rendered from a bridge.
    The storyboard box says what the pieces are; leave it empty and every piece gets the prompt as typed."""

    CATEGORY = "MiniMax H3/rolling"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "plan")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "sampler": ("SAMPLER", {"tooltip": "Not wired = er_sde"}),
            "sigmas": ("SIGMAS", {"tooltip": "Not wired = 6 steps, beta"}),
        }
        for i in range(1, 10):
            optional[f"picture_{i}"] = ("IMAGE", {"tooltip": f"image {i} in the storyboard, <Picture {i}> in the prompt"})
        optional["ref_video"] = ("IMAGE", {"tooltip": "Reference video frames at 24 fps (2-15s): <Video 1> in the prompt."})
        optional["ref_audio"] = ("AUDIO", {"tooltip": "Standalone reference audio: <Audio 1> in the prompt."})
        optional["mods"] = ("H3_REF_MODS", {"tooltip": MODS_TIP})
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "Video VAE"}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Sent to H3 as typed, for every piece, with the pictures that piece has. Optional: {beat} marks where each storyboard piece's text goes."}),
                "storyboard": ("STRING", {"multiline": True, "default": "", "tooltip": STORYBOARD_TIP}),
                "shots": ("INT", {"default": 1, "min": 1, "max": 500,
                                  "tooltip": "How many shots when the storyboard has no pieces in it (one bridge between each pair). Ignored otherwise."}),
                "shot_seconds": ("FLOAT", {"default": 5.0, "min": 0.9, "max": 15.0, "step": 0.1,
                                           "tooltip": "Length of a shot that doesn't give its own. Snapped to H3's 17k+5 frames."}),
                "bridge_seconds": ("FLOAT", {"default": 5.0, "min": 0.9, "max": 15.0, "step": 0.25,
                                             "tooltip": "Length of a bridge that doesn't give its own, including the 5 frames it shares with the shot on each side."}),
                "width": ("INT", {"default": 864, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "Shot N uses seed + N - 1; bridge N uses seed + 50000 + N - 1. 'take' and 'seed' in the storyboard change one piece."}),
                "ref_image_size": (["match", "max"], {"default": "max"}),
                "soundtrack": (["generated", "silent"], {"default": "generated"}),
                "upscale_pass": ("BOOLEAN", {"default": False,
                                             "tooltip": "Per piece: latent-upscale with the Minimax H3 3D upscaler, then a short polish pass. Pieces already rendered at preview size are reused."}),
                "upscale_megapixels": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 4.0, "step": 0.1}),
                "upscale_steps": ("INT", {"default": 2, "min": 1, "max": 30}),
                "upscale_denoise": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05}),
                "frame_labels": ("BOOLEAN", {"default": False,
                                             "tooltip": "Burn in which piece and which frame of it every frame is, plus the frame number and time. For review."}),
                "bridge_hold": (list(BRIDGE_HOLDS), {"default": "5 frames",
                                                     "tooltip": "How much of its two neighbours a bridge holds. 5 frames: the last 5 of the shot before and 5 of the shot after. Good for short storyboard transitions. 22 frames: 22 at each end, which gives H3 about a second of real motion to match, so pace carries across the join. The bridge starts 17 frames before the shot's end and draws that ending itself, because the end of a clip tends to go stiff. This is roycho87's Geiru Context 39 / drop_last 17 with Geiru Anchoring 39 / drop_first 17. With 22, shots and bridges need to be about 3 seconds or longer. One bridge can override this with 'hold 5' or 'hold 22' on its storyboard line."}),
                "keystone": ("BOOLEAN", {"default": False, "tooltip": KEYSTONE_TIP}),
            },
            "optional": optional,
        }

    def run(self, model, clip, vae, audio_vae, prompt, storyboard="", shots=1, shot_seconds=5.0, bridge_seconds=5.0,
            width=864, height=480, seed=0, ref_image_size="max", soundtrack="generated",
            upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4, frame_labels=False,
            bridge_hold="5 frames", keystone=False, sampler=None, sigmas=None, ref_video=None, ref_audio=None,
            mods=None,
            # not on the node any more; still honoured from an API graph or lab.json
            timeline="", beat="", steps=6, sampler_name="er_sde", scheduler="beta", pin_mode="mask", **pictures):
        from . import storyboard as storyboard_mod
        importlib.reload(rolling_core)
        importlib.reload(rolling_refmods)
        importlib.reload(rolling_timeline)
        importlib.reload(storyboard_mod)
        mod_groups = rolling_refmods.groups(mods)
        lab = rolling_core._lab()
        steps, sampler_name, scheduler = (lab.get("steps", steps), lab.get("sampler_name", sampler_name),
                                          lab.get("scheduler", scheduler))
        box = (storyboard or "").strip() or (timeline or "").strip()
        solo, notes = None, []
        if box[:1] in ("[", "{"):  # a timeline as JSON, as before
            tl = box
        else:
            loaded = [i for i in range(1, 10) if pictures.get(f"picture_{i}") is not None]
            built = storyboard_mod.build(box, prompt, shots, seed, loaded, ref_video is not None,
                                         ref_audio is not None, beat)
            tl, solo, notes = {"pieces": built["pieces"]}, built["solo"], built["notes"]
            for k, p in enumerate(built["pieces"]):
                print(f"[H3 Rolling Timeline] piece {k + 1} ({p['kind']}) is sent:\n{p['prompt']}\n")
        return rolling_timeline.run_timeline(
            model, clip, vae, audio_vae, tl, prompt, beat, round(shot_seconds * rolling_core.FPS),
            round(bridge_seconds * rolling_core.FPS), width, height, steps, sampler_name, scheduler, seed,
            ref_image_size, pin_mode=pin_mode, soundtrack=soundtrack, upscale_pass=upscale_pass,
            upscale_megapixels=upscale_megapixels, upscale_steps=upscale_steps, upscale_denoise=upscale_denoise,
            frame_labels=frame_labels, pictures=[pictures.get(f"picture_{i}") for i in range(1, 10)],
            ref_video=ref_video, ref_audio=ref_audio, sampler=sampler, sigmas=sigmas, solo=solo, extra_notes=notes,
            hold=BRIDGE_HOLDS.get(bridge_hold, (5, 0)), keystone=keystone,
            mods=rolling_refmods.rows(mod_groups, rolling_refmods.state(mod_groups)) if mods is not None else None)


ARMS_TIP = """The node's own settings are the baseline. Each line here is one more arm and only says what is different.

  hold = 22
  lora 2 = on; lora 1 = off; steps = 8          several changes on one line:  ;
  lora 1 strength = 0.6 | 0.8 | 1.0             a sweep, one arm per value:  |
  cross: steps = 4 | 6; lora 1 strength = 0.8 | 1.0     two sweeps, every combination
  a = locked-off tripod shot | (empty)          what goes where {a} is in the prompt or storyboard
  refmod 1 look = 0.6 | 1.0                     a RefMod from the stack wired into mods (by place or name)
  shift = 6 | 12                                where the steps are spent (12 = H3's own); audio shift = 3 | 6
  control                                       the baseline again with the piece cache off
  # a note

Settings: model, lora N (on / off / a strength), lora N strength, lora N file, steps, sampler, scheduler,
mp (megapixels), aspect (9:16), size (an exact 1024x576), shot seconds, bridge seconds, hold (5 / 22), upscale (on / off), upscale mp, upscale steps,
upscale denoise, pictures (match / max), keystone, attention, shift, audio shift, refmod N (on / off / a weight), refmod N look, refmod N voice,
seeds, and any slot letter.

{a: handheld, slight sway} in the prompt is a slot with the words the baseline uses. An arm that sets a slot runs on at least 3 seeds unless its line says seeds = N."""

LORA_SLOTS = 4
MODS_TIP = ("RefMods from a Fantastic H3 RefMod Stack. In the prompt they come after the media: a look made from several "
            "pictures is the next <Video n>, a voice the next <Audio n>; the report lists the exact tags.")


class H3RollingRoyale:
    """One Timeline scene rendered several ways on the same seeds, for comparing settings. The widgets are
    the baseline; the arms box lists what each other arm changes. Loads the model and LoRAs itself so arms
    can swap them. Writes one mp4 per take and royale.json to output/h3_royale/<time>/."""

    CATEGORY = "MiniMax H3/rolling"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "folder")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models")
        loras = ["none"] + folder_paths.get_filename_list("loras")
        required = {
            "clip": ("CLIP",),
            "vae": ("VAE", {"tooltip": "Video VAE"}),
            "audio_vae": ("VAE", {"tooltip": "Audio VAE"}),
            "model": (unets, {"tooltip": "The baseline's diffusion model. The node loads it itself, so an arm can swap it."}),
        }
        for i in range(1, LORA_SLOTS + 1):
            required[f"lora_{i}"] = (loras, {"default": "none"})
            required[f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
            required[f"lora_{i}_on"] = ("BOOLEAN", {"default": i == 1,
                                                    "tooltip": "Off = sitting in the slot but not used by the baseline; an arm can turn it on."})
        required.update({
            "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
            "prompt": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "As on the Timeline node. {a}, {b} ... are slots an arm can fill; {a: some words} gives the baseline's words."}),
            "storyboard": ("STRING", {"multiline": True, "default": "", "tooltip": STORYBOARD_TIP}),
            "arms": ("STRING", {"multiline": True, "default": "", "tooltip": ARMS_TIP}),
            "shots": ("INT", {"default": 1, "min": 1, "max": 500, "tooltip": "How many shots when the storyboard has no pieces in it."}),
            "shot_seconds": ("FLOAT", {"default": 5.0, "min": 0.9, "max": 15.0, "step": 0.1}),
            "bridge_seconds": ("FLOAT", {"default": 5.0, "min": 0.9, "max": 15.0, "step": 0.25}),
            "aspect_ratio": (list(rolling_royale.ASPECTS), {"default": "16:9 (Widescreen)",
                                                            "tooltip": "The shape of the picture. Same list as Seed Hunter's Video Settings."}),
            "megapixels": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 4.0, "step": 0.1, "display": "slider",
                                     "tooltip": "How big. 0.4 at 16:9 is 864x480, 0.6 is 1056x608; the plan shows the exact size. The same sum as Seed Hunter's Video Settings, snapped to 32."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                             "tooltip": "Every arm uses this seed. With more seeds per arm, take 2 uses seed + 1000, take 3 seed + 2000."}),
            "seeds_per_arm": ("INT", {"default": 1, "min": 1, "max": 10,
                                      "tooltip": "How many seeds every arm is rendered on. One take per arm compares luck as much as settings."}),
            "ref_image_size": (["match", "max"], {"default": "max"}),
            "soundtrack": (["generated", "silent"], {"default": "generated"}),
            "upscale_pass": ("BOOLEAN", {"default": False}),
            "upscale_megapixels": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 4.0, "step": 0.1}),
            "upscale_steps": ("INT", {"default": 2, "min": 1, "max": 30}),
            "upscale_denoise": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05}),
            "frame_labels": ("BOOLEAN", {"default": False,
                                         "tooltip": "Burn in which piece and which frame of it every frame is."}),
            "bridge_hold": (list(BRIDGE_HOLDS), {"default": "5 frames"}),
            "mode": (["run", "plan only"], {"default": "run",
                                            "tooltip": "plan only: list the arms, count the takes and estimate the time, render nothing."}),
            "keystone": ("BOOLEAN", {"default": False, "tooltip": "The baseline's setting; an arm can say keystone = on / off. " + KEYSTONE_TIP}),
            "attention": (list(rolling_royale.ATTENTIONS), {"default": "default", "tooltip": (
                "comfy kitchen: Comfy's INT8 attention (the Model Attention Backend node's 'comfy kitchen attention', which Seed Hunter "
                "has switched on). default: the model as loaded, what every royale before 11-26 used. The baseline's setting; "
                "an arm can say attention = kitchen / default. The same seed gives a slightly different take under each.")}),
            "shift": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.5, "tooltip": (
                "Where the steps are spent. Higher: more of them on layout and motion. Lower: more on fine detail. 12 is H3's own "
                "(with audio_shift at 3 the model is left untouched). LoRA authors use 12 with about 18 steps and no turbo, 6 with "
                "8 to 12 steps and turbo. An arm can say shift = 6 | 12.")}),
            "audio_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.5,
                                      "tooltip": "The same for the sound. 3 is H3's own. An arm can say audio shift = 3 | 6."}),
        })
        optional = {f"picture_{i}": ("IMAGE", {"tooltip": f"image {i} in the storyboard, <Picture {i}> in the prompt"})
                    for i in range(1, 10)}
        optional["ref_video"] = ("IMAGE", {"tooltip": "Reference video frames at 24 fps (2-15s): <Video 1> in the prompt."})
        optional["ref_audio"] = ("AUDIO", {"tooltip": "Standalone reference audio: <Audio 1> in the prompt."})
        optional["mods"] = ("H3_REF_MODS", {"tooltip": MODS_TIP + " An arm can say refmod 1 look = 0.6, refmod 1 voice = 0, refmod 1 = off."})
        # hidden: the whole job as queued (saved into the run folder so the results page can start a new run
        # from one of this run's takes), this node's id in it, and what the page sent along when it did
        return {"required": required, "optional": optional,
                "hidden": {"job_graph": "PROMPT", "node_id": "UNIQUE_ID", "extra": "EXTRA_PNGINFO"}}

    def run(self, clip, vae, audio_vae, model, steps, sampler_name, scheduler, prompt, storyboard, arms, shots,
            shot_seconds, bridge_seconds, aspect_ratio, megapixels, seed, seeds_per_arm, ref_image_size, soundtrack,
            upscale_pass, upscale_megapixels, upscale_steps, upscale_denoise, frame_labels, bridge_hold, mode,
            keystone=False, attention="default", shift=12.0, audio_shift=3.0, ref_video=None, ref_audio=None, mods=None,
            job_graph=None, node_id=None,
            extra=None, **kw):
        from . import storyboard as storyboard_mod
        for mod in (rolling_core, rolling_refmods, rolling_timeline, storyboard_mod, rolling_royale):
            importlib.reload(mod)
        base = {
            "model": model,
            "loras": [{"file": kw.get(f"lora_{i}", "none"), "strength": float(kw.get(f"lora_{i}_strength", 1.0)),
                       "on": bool(kw.get(f"lora_{i}_on", False))} for i in range(1, LORA_SLOTS + 1)],
            "steps": steps, "sampler": sampler_name, "scheduler": scheduler, "aspect": aspect_ratio, "mp": float(megapixels),
            "shot_seconds": shot_seconds, "bridge_seconds": bridge_seconds, "hold": BRIDGE_HOLDS.get(bridge_hold, (5, 0))[0],
            "upscale": upscale_pass, "upscale_mp": upscale_megapixels, "upscale_steps": upscale_steps,
            "upscale_denoise": upscale_denoise, "ref_image_size": ref_image_size, "seeds": seeds_per_arm,
            "keystone": bool(keystone), "attention": attention, "shift": float(shift), "audio_shift": float(audio_shift),
        }
        extra = extra if isinstance(extra, dict) else {}
        if (extra.get("h3royale_base") or {}).get("exact"):  # a run started from a take that had an exact size
            base["exact"] = [int(x) for x in extra["h3royale_base"]["exact"]]
        job = {"prompt": job_graph, "node_id": str(node_id)} if job_graph and node_id is not None else None
        report, folder = rolling_royale.run_royale(
            clip, vae, audio_vae, prompt, storyboard, arms, base, shots, seed, soundtrack, frame_labels, mode,
            [kw.get(f"picture_{i}") for i in range(1, 10)], ref_video=ref_video, ref_audio=ref_audio,
            job=job, origin=extra.get("h3royale_origin"), mods=mods,
            refmod_weights=(extra.get("h3royale_base") or {}).get("refmods"))
        return {"ui": {"text": [report]}, "result": (report, folder)}


class H3Notes:
    """A plain text box for notes. Its words are kept in a file in Comfy's user folder (h3_notes.txt) by
    web/h3_notes.js, not in the workflow, so they are the same in every workflow and survive one being rebuilt
    or never saved. Does nothing when a job runs."""

    CATEGORY = "MiniMax H3/rolling"
    RETURN_TYPES = ()
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"notes": ("STRING", {"multiline": True, "default": "",
                                                   "tooltip": "Kept outside the workflow: the same notes wherever this node is, saved as you type."})}}

    def run(self, notes=""):
        return ()


NODE_CLASS_MAPPINGS = {
    "H3Notes": H3Notes,
    "H3RollingSource": H3RollingSource,
    "H3RollingPicture": H3RollingPicture,
    "H3RollingRef2VA": H3RollingRef2VA,
    "H3RollingTimeline": H3RollingTimeline,
    "H3RollingRoyale": H3RollingRoyale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3Notes": "H3 Notes (kept outside the workflow)",
    "H3RollingSource": "H3 Rolling Source (scrub)",
    "H3RollingPicture": "H3 Rolling Picture (may be empty)",
    "H3RollingRef2VA": "H3 Rolling Ref2VA",
    "H3RollingTimeline": "H3 Rolling Timeline (shots + bridges)",
    "H3RollingRoyale": "H3 Rolling Royale (compare settings)",
}
WEB_DIRECTORY = "./web"

_ensure_empty_picture()

# The Royale results page: http://<this Comfy>:8188/h3royale
try:
    from server import PromptServer

    from . import royale_server
    royale_server.register(PromptServer.instance.routes)
except Exception as e:  # the nodes still work without the page
    print(f"[H3 Rolling Royale] results page not available: {e}")
