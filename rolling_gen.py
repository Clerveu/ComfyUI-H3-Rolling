"""H3 Rolling Generate: a long H3 video from pictures and prompts, no source video.

A chain of clips, each starting from the real latents at the end of the one before, joined into one file as it
renders. The chaining and the drift protection are roycho87's Geiru Context Suite (GPL-3.0), called here as
installed nodes, not copied: Geiru Context holds the previous clip's tail, Geiru Context Noise roughs up the held
frames, Geiru Keystone / Late Lock pull every clip back toward clip 1. https://github.com/roycho87/geiru_context_suite
"""
import re

import torch

from .rolling_core import FPS, FrameSink, _lab, _mark, _stream_path, _T, align

GEIRU = {"context": "H3ClownContextExperimental", "noise": "H3ClownContextNoise",
         "keystone": "H3ClownKeyframe", "late_lock": "H3ClownLateLock"}
KEYSTONE_PLACEMENTS = ["keystone front + capstone back", "keystone front", "keystone front + back",
                       "capstone front + back", "off"]
LATE_LOCK_SLOTS = ["capstone slot (back)", "keystone slot (front)", "keystone + capstone slots"]


def split_prompts(text):
    """One prompt per clip, separated by a line of three or more dashes. A single prompt is used for every clip;
    with fewer prompts than clips, the last one repeats."""
    parts = [p.strip() for p in re.split(r"(?m)^\s*-{3,}\s*$", text)]
    return [p for p in parts if p] or [""]


def _geiru():
    import nodes
    found = {k: nodes.NODE_CLASS_MAPPINGS.get(v) for k, v in GEIRU.items()}
    missing = [GEIRU[k] for k, v in found.items() if v is None]
    if missing:
        raise RuntimeError("H3 Rolling Generate needs roycho87's geiru_context_suite in custom_nodes "
                           f"(missing: {', '.join(missing)}): https://github.com/roycho87/geiru_context_suite")
    return {k: v() for k, v in found.items()}


def run_generate(model, clip, vae, audio_vae, prompts, width, height, clip_seconds, clips, head_frames, drop_last,
                 steps, sampler_name, scheduler, seed, ref_image_size, hold_audio=True, protection=True,
                 keystone="keystone front + capstone back", late_lock_step=4, late_lock_slots=LATE_LOCK_SLOTS[0],
                 context_noise=1.0, soundtrack="generated", pictures=(), sampler=None, sigmas=None):
    import comfy.model_management
    import comfy.samplers
    import nodes
    from comfy_extras.nodes_audio import vae_decode_audio
    from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    g = _geiru()
    L = align(round(clip_seconds * FPS))
    head, drop = int(head_frames), int(drop_last)
    N = max(1, int(clips))
    if N > 1 and head >= L:
        raise ValueError(f"the overlap ({head} frames) must be shorter than a clip ({L} frames)")
    if drop >= head:
        raise ValueError(f"drop_last ({drop}) must be smaller than the overlap ({head})")
    step = L - head
    F = step * (N - 1) + L
    w, h = max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)
    texts = split_prompts(prompts)
    clip_prompts = [texts[min(k, len(texts) - 1)] for k in range(N)]
    _T[0] = 0.0
    _mark("start")

    plan = (f"{N} clip(s) of {L} frames ({L / FPS:.2f}s) at {w}x{h}, overlap {head} (held {head - drop}, free {drop})"
            f" -> {F} frames ({F / FPS:.2f}s) | {len(texts)} prompt(s)"
            + (" | protection off" if not protection else
               f" | keystone: {keystone}, late lock step {late_lock_step} ({late_lock_slots}), context noise {context_noise}"))
    print(f"[H3 Rolling Generate] {plan}")

    # 1) encode each distinct prompt once, so the text encoder loads once
    ref_images = {f"ref_image_{i + 1}": p for i, p in enumerate(p for p in pictures if p is not None)} or None
    encoded = {}
    for text in clip_prompts:
        if text not in encoded:
            out = MiniMaxH3ReferenceToVideo.execute(
                clip=clip, prompt=text, width=w, height=h, length=L, ref_image_size=ref_image_size,
                vae=vae, audio_vae=audio_vae, ref_images=ref_images)
            encoded[text] = out.args
            _mark(f"encode prompt {len(encoded)}")

    if sigmas is None:
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
    if sampler is None:
        sampler = comfy.samplers.sampler_object(sampler_name)

    lab = _lab()
    sink, out_audio, sr = None, None, None
    first, prev = None, None
    for k in range(N):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cond, empty = encoded[clip_prompts[k]]
        latent = {kk: (v.clone() if hasattr(v, "clone") else v) for kk, v in empty.items()}
        mdl = model
        if k > 0:
            shot = k + 1
            latent = g["context"].anchor(latent, head, drop_last=drop, drop_first=0, hold_audio=hold_audio,
                                         first_shot=1, audio_drop_last=0, audio_feather_ticks=0, shot_number=shot,
                                         previous_shot=prev, degradation_protection=protection)[0]
            if context_noise > 0:
                latent, mdl = g["noise"].noise(latent, enabled=True, strength=context_noise, fade="linear",
                                               skip_first=True, model=mdl)
            if keystone != "off":
                latent, mdl = g["keystone"].graft(latent, enabled=True, keyframe=first, shot_number=shot,
                                                  placement=keystone, model=mdl, source_clip=1, latent_folder="")
                mdl, latent = g["late_lock"].patch(mdl, latent, swap_step=late_lock_step, slots=late_lock_slots)
        _mark(f"clip {k + 1} setup")

        guider = Guider_Basic(mdl)
        guider.set_conds(cond)
        out = SamplerCustomAdvanced.execute(Noise_RandomNoise(seed + k), guider, sampler, sigmas, latent)
        samples = (getattr(out, "args", None) or out.result)[0]["samples"]
        _mark(f"clip {k + 1} sample")
        prev = {"samples": samples}
        if first is None:
            first = prev

        s = k * step
        imgs = nodes.VAEDecode().decode(vae, {"samples": samples})[0].float().cpu()
        _mark(f"clip {k + 1} vae decode")
        if sink is None:
            Hh, Ww = imgs.shape[1], imgs.shape[2]
            sink = FrameSink(F, (Hh, Ww), _stream_path(lab, F, Hh, Ww))
        # the join: this clip's opening frames replace the previous clip's last `head` frames (the held ones are the
        # same frames; the dropped ones are this clip's take of that moment)
        sink.place(k, s, imgs[:min(imgs.shape[0], L, F - s)], 0)
        del imgs
        sink.flush((k + 1) * step if k + 1 < N else F)

        aud = vae_decode_audio(audio_vae, {"samples": samples})
        _mark(f"clip {k + 1} audio decode")
        wf = aud["waveform"].float().cpu()
        if out_audio is None:
            sr = aud["sample_rate"]
            out_audio = torch.zeros((1, wf.shape[1], round(F / FPS * sr)), dtype=torch.float32)
        a0 = round(s * sr / FPS)
        want = max(0, min(round(L * sr / FPS), out_audio.shape[-1] - a0))
        seg = wf[..., :want]
        if seg.shape[-1] < want:
            seg = torch.nn.functional.pad(seg, (0, want - seg.shape[-1]))
        ab = 0 if k == 0 else min(round(0.05 * sr), want)  # 50 ms fade where the two takes of the held sound meet
        if ab:
            ramp = torch.linspace(0, 1, ab + 2)[1:-1]
            out_audio[..., a0:a0 + ab] = out_audio[..., a0:a0 + ab] * (1 - ramp) + seg[..., :ab] * ramp
        out_audio[..., a0 + ab:a0 + want] = seg[..., ab:]
        print(f"[H3 Rolling Generate] clip {k + 1}/{N} done (frames {s}-{s + L})")

    final_audio = None if soundtrack == "silent" else {"waveform": out_audio, "sample_rate": sr}
    frames_out = sink.finish(final_audio)
    if sink.path:
        plan += f" | full video: {sink.path}"
    return frames_out, final_audio, plan
