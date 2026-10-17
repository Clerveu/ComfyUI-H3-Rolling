"""H3 Rolling: render long reference-driven H3 videos as overlapping, guided chunks."""

import importlib
import os

import comfy.samplers
import folder_paths

from . import rolling_core

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
                                      "tooltip": "One prompt for every chunk. Tags: <Picture N> for pictures, <Video 1> for the source slice, <Audio 1> for its soundtrack (only if source audio is used as reference)."}),
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
            lock_strength=1.0, lock_preview_only=False, lock_max_people=2, audio=None, sampler=None, sigmas=None, picture_1=None, picture_2=None, picture_3=None, picture_4=None,
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
                     "lock_preview_only": lock_preview_only, "lock_max_people": lock_max_people})


NODE_CLASS_MAPPINGS = {
    "H3RollingSource": H3RollingSource,
    "H3RollingRef2VA": H3RollingRef2VA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RollingSource": "H3 Rolling Source (scrub)",
    "H3RollingRef2VA": "H3 Rolling Ref2VA",
}
WEB_DIRECTORY = "./web"
