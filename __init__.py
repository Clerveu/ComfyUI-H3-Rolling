"""H3 Rolling: render long reference-driven H3 videos as overlapping, guided chunks."""

import importlib
import os

import comfy.samplers
import folder_paths

from . import rolling_anchor, rolling_core, rolling_gen, rolling_timeline

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")

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


GEN_PROMPT = """subject_definitions:
<Subject 1> is the cat whose appearance comes from <Picture 1>.

summary:
[reference generation] <Subject 1> rides a skateboard down a quiet street in one continuous shot.

retention_analysis:
<Subject 1> (throughout [Shot 1]): fully_preserved - her face, coat, and proportions from <Picture 1>.

detailed_description:
Photorealistic live-action footage in daylight.

[Shot 1] A tracking shot follows <Subject 1>, a real cat, standing on a skateboard as it rolls down a quiet street, her paws planted on the deck. The camera tracks alongside her at the same speed.

overall_soundscape:
Skateboard wheels on pavement and light wind.

non_diegetic_music:
N/A"""


class H3RollingGenerate:
    """A long H3 video from pictures and prompts, no source video: a chain of clips, each starting from the end of
    the one before (roycho87's Geiru Context Suite does the hand-off and the drift protection), joined as it renders."""

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
                "prompts": ("STRING", {"multiline": True, "default": GEN_PROMPT,
                                       "tooltip": "One prompt = every clip uses it (generic ongoing action). For progressive action or dialogue, write one prompt per clip, separated by a line of three dashes: ---  With fewer prompts than clips, the last one repeats. Tags: <Picture N>."}),
                "width": ("INT", {"default": 864, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 32}),
                "clip_seconds": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 15.0, "step": 0.25,
                                           "tooltip": "Length of each clip. H3 is trained on 5-15s."}),
                "clips": ("INT", {"default": 3, "min": 1, "max": 200,
                                  "tooltip": "How many clips to chain. Total length = clips x (clip - overlap) + overlap."}),
                "overlap_frames": (["39", "22", "56"], {"default": "39",
                                                        "tooltip": "Frames each clip shares with the one before (Geiru Context's head_frames)."}),
                "drop_last": (["17", "0", "34"], {"default": "17",
                                                  "tooltip": "Of the overlap, how many frames at its end are made fresh instead of held (they still fit the held sound). 39 with 17 = 22 held, 17 free."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "Clip N uses seed + N."}),
                "ref_image_size": (["match", "max"], {"default": "max"}),
                "hold_audio": ("BOOLEAN", {"default": True, "tooltip": "Hold the previous clip's sound across the overlap."}),
                "degradation_protection": ("BOOLEAN", {"default": True,
                                                       "tooltip": "Geiru's switch: off = plain chain (keystone, late lock and context noise all pass through). Turn it off around hard cuts and scene changes."}),
                "keystone": (rolling_gen.KEYSTONE_PLACEMENTS, {"default": rolling_gen.KEYSTONE_PLACEMENTS[0],
                                                              "tooltip": "Geiru Keystone: clip 1's first frame locked into the start of every later clip (and put back after), and/or a frame from clip 1's end tacked onto the back."}),
                "late_lock_step": ("INT", {"default": 4, "min": 0, "max": 100,
                                           "tooltip": "Geiru Late Lock: steps before the keystone/capstone come in, so the clip picks its own pose first. He recommends the last quarter of the steps."}),
                "late_lock_slots": (rolling_gen.LATE_LOCK_SLOTS, {"default": rolling_gen.LATE_LOCK_SLOTS[0]}),
                "context_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Geiru Context Noise: how much of the current noise the held frames get, fading to none at the join. 0 = off."}),
                "soundtrack": (["generated", "silent"], {"default": "generated"}),
                "keystone_attention": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 1000.0, "step": 0.5,
                                                 "tooltip": "Geiru Keystone Attention (suite 1.1+): how hard each clip looks at the keystone compared with the carried frames, in the 13 blocks he found carry brightness. 20 is his default. 1 or 0 = off."}),
            },
            "optional": {
                "sampler": ("SAMPLER", {"tooltip": "Overrides sampler_name"}),
                "sigmas": ("SIGMAS", {"tooltip": "Overrides steps + scheduler"}),
                "picture_1": ("IMAGE",),
                "picture_2": ("IMAGE",),
                "picture_3": ("IMAGE",),
                "picture_4": ("IMAGE",),
            },
        }

    def run(self, model, clip, vae, audio_vae, prompts, width, height, clip_seconds, clips, overlap_frames, drop_last,
            steps, sampler_name, scheduler, seed, ref_image_size, hold_audio=True, degradation_protection=True,
            keystone=rolling_gen.KEYSTONE_PLACEMENTS[0], late_lock_step=4, late_lock_slots=rolling_gen.LATE_LOCK_SLOTS[0],
            context_noise=1.0, soundtrack="generated", keystone_attention=20.0, sampler=None, sigmas=None,
            picture_1=None, picture_2=None, picture_3=None, picture_4=None):
        importlib.reload(rolling_core)
        importlib.reload(rolling_gen)
        return rolling_gen.run_generate(
            model, clip, vae, audio_vae, prompts, width, height, clip_seconds, clips, int(overlap_frames),
            int(drop_last), steps, sampler_name, scheduler, seed, ref_image_size, hold_audio=hold_audio,
            protection=degradation_protection, keystone=keystone, late_lock_step=late_lock_step,
            late_lock_slots=late_lock_slots, context_noise=context_noise, soundtrack=soundtrack,
            pictures=(picture_1, picture_2, picture_3, picture_4), sampler=sampler, sigmas=sigmas,
            keystone_attention=keystone_attention)


ANCHOR_PROMPT = """subject_definitions:
<Subject 1> is the woman whose appearance comes from <Picture 1>.
<Subject 2> is the room shown in {still}.

summary:
[reference generation] {beat}

retention_analysis:
<Subject 1> (throughout [Shot 1]): fully_preserved - her face, hair, build and clothing from <Picture 1>.
<Subject 2> (throughout [Shot 1]): fully_preserved - the room, its furniture and its light from {still}.

detailed_description:
Photorealistic live-action footage in a living room in daylight, filmed by a static camera.

[Shot 1] {beat}

overall_soundscape:
Quiet room tone.

non_diegetic_music:
N/A"""

ANCHOR_BEATS = """<Subject 1> stands at the window on the left side of the room.
<Subject 1> turns from the window, walks across the room to the couch on the right side of the room, and sits down on it.
<Subject 1> sits on the couch on the right side of the room."""


class H3RollingAnchors:
    """A long H3 video from pictures and a storyboard, no source video: short anchor clips rendered on their own,
    then a long bridge between each pair, pinned to both. Nothing is ever rendered from a bridge."""

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
                "prompt": ("STRING", {"multiline": True, "default": ANCHOR_PROMPT,
                                      "tooltip": "The prompt every piece shares. {beat} is replaced by that piece's line from the beats box. {still} is the tag of a frame taken from anchor 1 and given to every later piece as an extra picture; lines that mention {still} are left out of anchor 1's own prompt. Leave {still} out entirely and no frame is taken. Or skip the beats box and write whole prompts here, separated by lines of ---, in the order anchor, bridge, anchor ... anchor."}),
                "beats": ("STRING", {"multiline": True, "default": ANCHOR_BEATS,
                                     "tooltip": "One line per piece, alternating: where she is (anchor), how she gets to the next place (bridge), where she is then (anchor) ... ending on an anchor. 3 lines = 2 anchors + 1 bridge, 5 lines = 3 anchors + 2 bridges."}),
                "width": ("INT", {"default": 864, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 32}),
                "anchor_frames": (["22", "39", "56"], {"default": "22",
                                                       "tooltip": "Length of each anchor clip. Its first and last 5 frames are shared with the bridges on either side."}),
                "clip_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.25,
                                           "tooltip": "Length of each bridge: the time she has to get from one anchor to the next. H3 is trained on 5-15s."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "er_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "Anchor N uses seed + N; bridge N uses seed + 50000 + N."}),
                "ref_image_size": (["match", "max"], {"default": "max"}),
                "pin_mode": (["mask", "keyframe"], {"default": "mask",
                                                    "tooltip": "How a bridge holds the 5 frames it shares with each anchor. mask: the anchors' real latents are written into the bridge and masked off, so they decode as the anchors' own frames and the joins are plain cuts. keyframe: passed as H3 keyframe guides; the bridge draws its own copy and the joins are 5-frame crossfades."}),
                "soundtrack": (["generated", "silent"], {"default": "generated"}),
                "upscale_pass": ("BOOLEAN", {"default": False,
                                             "tooltip": "Per piece: latent-upscale with the Minimax H3 3D upscaler, then a short polish pass (Seed Hunter's stage 2). Bridges are pinned to the upscaled anchors."}),
                "upscale_megapixels": ("FLOAT", {"default": 1.5, "min": 0.2, "max": 4.0, "step": 0.1}),
                "upscale_steps": ("INT", {"default": 2, "min": 1, "max": 30}),
                "upscale_denoise": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "sampler": ("SAMPLER", {"tooltip": "Overrides sampler_name"}),
                "sigmas": ("SIGMAS", {"tooltip": "Overrides steps + scheduler"}),
                "picture_1": ("IMAGE",),
                "picture_2": ("IMAGE",),
                "picture_3": ("IMAGE",),
                "picture_4": ("IMAGE",),
            },
        }

    def run(self, model, clip, vae, audio_vae, prompt, beats, width, height, anchor_frames, clip_seconds, steps,
            sampler_name, scheduler, seed, ref_image_size, pin_mode="mask", soundtrack="generated",
            upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4,
            sampler=None, sigmas=None, picture_1=None, picture_2=None, picture_3=None, picture_4=None):
        importlib.reload(rolling_core)
        importlib.reload(rolling_anchor)
        return rolling_anchor.run_anchors(
            model, clip, vae, audio_vae, prompt, beats, width, height, int(anchor_frames), clip_seconds, steps,
            sampler_name, scheduler, seed, ref_image_size, pin_mode=pin_mode, soundtrack=soundtrack,
            upscale_pass=upscale_pass, upscale_megapixels=upscale_megapixels, upscale_steps=upscale_steps,
            upscale_denoise=upscale_denoise, pictures=(picture_1, picture_2, picture_3, picture_4),
            sampler=sampler, sigmas=sigmas)


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
            },
            "optional": optional,
        }

    def run(self, model, clip, vae, audio_vae, prompt, storyboard="", shots=1, shot_seconds=5.0, bridge_seconds=5.0,
            width=864, height=480, seed=0, ref_image_size="max", soundtrack="generated",
            upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4, frame_labels=False,
            sampler=None, sigmas=None, ref_video=None, ref_audio=None,
            # not on the node any more; still honoured from an API graph or lab.json
            timeline="", beat="", steps=6, sampler_name="er_sde", scheduler="beta", pin_mode="mask", **pictures):
        from . import storyboard as storyboard_mod
        importlib.reload(rolling_core)
        importlib.reload(rolling_timeline)
        importlib.reload(storyboard_mod)
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
            ref_video=ref_video, ref_audio=ref_audio, sampler=sampler, sigmas=sigmas, solo=solo, extra_notes=notes)


NODE_CLASS_MAPPINGS = {
    "H3RollingSource": H3RollingSource,
    "H3RollingRef2VA": H3RollingRef2VA,
    "H3RollingGenerate": H3RollingGenerate,
    "H3RollingAnchors": H3RollingAnchors,
    "H3RollingTimeline": H3RollingTimeline,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RollingSource": "H3 Rolling Source (scrub)",
    "H3RollingRef2VA": "H3 Rolling Ref2VA",
    "H3RollingGenerate": "H3 Rolling Generate (Geiru chain)",
    "H3RollingAnchors": "H3 Rolling Anchors (beats + bridges)",
    "H3RollingTimeline": "H3 Rolling Timeline (shots + bridges)",
}
WEB_DIRECTORY = "./web"
