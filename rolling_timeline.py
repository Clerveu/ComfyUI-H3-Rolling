"""H3 Rolling Timeline: a long H3 video from an ordered list of pieces.

A piece is a SHOT (rendered on its own from the pictures and its prompt) or a BRIDGE (a normal clip pinned
at its start to the last 5 frames of the shot before it and at its end to the first 5 frames of the shot
after it). Shots come from the pictures. Bridges come from two shots. Nothing is rendered from a bridge,
so nothing is ever more than one generation from a fresh render.

Every piece can have its own length, its own prompt (or a beat that goes into the shared prompt) and its
own choice of the loaded media. Anything a piece doesn't say comes from the run's defaults.

The timeline is plain data (see parse_timeline). simple_timeline() writes one from a few numbers; a UI
can write one later without this file changing.
"""
import json
import math
import os
import re
import time

import numpy as np
import torch

from .rolling_core import FPS, FrameSink, _lab, _mark, _stream_path, _T, align, video_latent_t

PIN = 5       # frames a bridge shares with the shot on each side (one 1-frame latent + one 4-frame latent)
MIN_LEN = 22  # shortest piece: 17 + 5


# ---------------------------------------------------------------- the timeline as data

def simple_timeline(shots, shot_frames=None, bridge_frames=None):
    """One prompt, N shots: shot, bridge, shot ... shot, every piece on the defaults."""
    pieces = []
    for k in range(max(1, int(shots))):
        if k:
            pieces.append({"kind": "bridge"})
        pieces.append({"kind": "shot"})
    out = {"pieces": pieces}
    d = {}
    if shot_frames:
        d["shot_frames"] = int(shot_frames)
    if bridge_frames:
        d["bridge_frames"] = int(bridge_frames)
    if d:
        out["defaults"] = d
    return out


def parse_timeline(timeline, prompt, beat, shot_frames, bridge_frames):
    """Timeline (dict, list of pieces, or JSON text of either) -> list of pieces with everything filled in:
    {"kind", "frames", "text", "media", "seed"}.

    {"defaults": {"shot_frames": 22, "bridge_frames": 124, "prompt": "...{beat}...", "beat": "...", "media": "all"},
     "pieces": [{"kind": "shot"}, {"kind": "bridge", "seconds": 4, "beat": "she walks to the couch"},
                {"kind": "shot", "prompt": "a whole prompt", "media": ["picture_1", "video"]}]}

    Per piece, all optional: frames or seconds, beat (goes where {beat} is in the shared prompt) or prompt
    (a whole prompt of its own, may also use {beat}), media ("all" or a list of picture_N / video / audio),
    seed. The arguments are the run's defaults; a "defaults" block in the timeline overrides them.
    """
    if isinstance(timeline, str):
        try:
            timeline = json.loads(timeline)
        except ValueError as e:
            raise ValueError(f"the timeline isn't valid JSON: {e}")
    if isinstance(timeline, list):
        timeline = {"pieces": timeline}
    d = timeline.get("defaults") or {}
    prompt = d.get("prompt", prompt)
    beat = d.get("beat", beat)
    lens = {"shot": d.get("shot_frames", shot_frames), "bridge": d.get("bridge_frames", bridge_frames)}
    media = d.get("media", "all")
    raw = timeline.get("pieces") or []
    if not raw:
        raise ValueError("the timeline has no pieces")
    out, notes = [], []
    for i, p in enumerate(raw):
        kind = p.get("kind")
        if kind not in ("shot", "bridge"):
            raise ValueError(f"piece {i + 1}: kind must be \"shot\" or \"bridge\", got {kind!r}")
        want = "shot" if i % 2 == 0 else "bridge"
        if kind != want:
            raise ValueError(f"piece {i + 1} is a {kind} where a {want} belongs: pieces go shot, bridge, shot ... "
                             "and start and end on a shot")
        if "frames" in p:
            asked = int(p["frames"])
        elif "seconds" in p:
            asked = round(float(p["seconds"]) * FPS)
        else:
            asked = int(lens[kind])
        frames = align(max(asked, MIN_LEN))
        if frames != asked and ("frames" in p or "seconds" in p):  # only a length the piece itself asked for
            notes.append(f"piece {i + 1} ({kind}): {asked} frames -> {frames} (lengths are 17k+5, at least {MIN_LEN})")
        text = p.get("prompt", prompt)
        b = p.get("beat", beat)
        if "{beat}" in text:
            if not (b or "").strip():
                raise ValueError(f"piece {i + 1}: the prompt has {{beat}} in it and this piece has no beat")
            text = text.replace("{beat}", b.strip())
        out.append({"kind": kind, "frames": frames, "text": text.strip(), "media": p.get("media", media),
                    "seed": p.get("seed")})
    if len(out) % 2 == 0:
        raise ValueError(f"{len(out)} pieces: they go shot, bridge, shot ... and must end on a shot, so the count is odd")
    return out, notes


def with_still(text, tag):
    """{still} is the tag of the frame taken from the first shot. Lines that mention it only exist once
    that frame exists: for the first shot itself (tag None) they are dropped."""
    if "{still}" not in text:
        return text
    if tag is None:
        return "\n".join(ln for ln in text.splitlines() if "{still}" not in ln)
    return text.replace("{still}", tag)


def pick_media(media, n_pics, has_video, has_audio, where):
    """A piece's media choice -> (indexes into the loaded pictures, use video, use audio)."""
    if media in (None, "all"):
        return list(range(n_pics)), has_video, has_audio
    if isinstance(media, str):
        media = [media]
    idx, vid, aud = [], False, False
    for m in media:
        m = str(m).strip().lower()
        mm = re.fullmatch(r"picture[_ ]?(\d+)", m)
        if mm:
            n = int(mm.group(1))
            if not 1 <= n <= n_pics:
                raise ValueError(f"{where}: media asks for picture_{n} and {n_pics} picture(s) are loaded")
            idx.append(n - 1)
        elif m == "video":
            if not has_video:
                raise ValueError(f"{where}: media asks for the video and none is loaded")
            vid = True
        elif m == "audio":
            if not has_audio:
                raise ValueError(f"{where}: media asks for the audio and none is loaded")
            aud = True
        else:
            raise ValueError(f"{where}: media entry {m!r} isn't picture_N, video or audio")
    return sorted(set(idx)), vid, aud


def renumber_pictures(text, idx, n_pics, where):
    """Prompts name pictures by their loaded number. H3 numbers them by position among the ones it is
    handed, so a piece that gets pictures 1 and 3 has to call the second one <Picture 2>."""
    if idx == list(range(n_pics)):
        return text
    pos = {old + 1: new + 1 for new, old in enumerate(idx)}

    def sub(m):
        n = int(m.group(1))
        if n > n_pics:
            return m.group(0)
        if n not in pos:
            raise ValueError(f"{where}: the prompt mentions <Picture {n}> and this piece's media leaves that picture out")
        return f"<Picture {pos[n]}>"

    return re.sub(r"<Picture (\d+)>", sub, text)


def layout(pieces):
    """Start frame of every piece, and the total. A bridge starts 5 frames before its left shot ends and
    ends 5 frames into its right shot."""
    starts, pos = [], 0
    for i, p in enumerate(pieces):
        if i:
            pos = starts[-1] + pieces[i - 1]["frames"] - PIN
        starts.append(pos)
    return starts, starts[-1] + pieces[-1]["frames"]


def frame_label(n, pieces, starts):
    """(text, color) for the top-left label of global frame n. Shot frames win where pieces overlap."""
    for i in range(0, len(pieces), 2):
        s, ln = starts[i], pieces[i]["frames"]
        if s <= n < s + ln:
            k, f = i // 2 + 1, n - s + 1
            if i and f <= PIN:
                return f"SHOT {k}  f {f}/{ln} = BRIDGE {k - 1} end pin", (255, 255, 0)
            if i < len(pieces) - 1 and f > ln - PIN:
                return f"SHOT {k}  f {f}/{ln} = BRIDGE {k} start pin", (255, 255, 0)
            return f"SHOT {k}  f {f}/{ln}", (255, 255, 255)
    for i in range(1, len(pieces), 2):
        s, ln = starts[i], pieces[i]["frames"]
        if s <= n < s + ln:
            return f"BRIDGE {(i + 1) // 2}  f {n - s + 1}/{ln}  FREE", (255, 165, 0)
    return "", (255, 255, 255)


def _font(size):
    from PIL import ImageFont
    for name in ("consolab.ttf", "DejaVuSansMono-Bold.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_label(arr, n, pieces, starts):
    """Burn the piece/frame label (top-left) and the global frame + time (bottom-left) into one uint8 frame."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    h = arr.shape[0]
    text, color = frame_label(n, pieces, starts)
    t = n / FPS
    rows = [(text, color, _font(max(10, h // 26)), 16),
            (f"frame {n}  {int(t // 60)}:{t % 60:05.2f}", (255, 255, 255), _font(max(9, h // 32)), None)]
    for txt, col, font, y in rows:
        if not txt:
            continue
        box = d.textbbox((0, 0), txt, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        if y is None:
            y = h - th - 16 - 8
        d.rectangle([8, y - 8, 16 + tw + 8, y + th + 8], fill=(0, 0, 0))
        d.text((16 - box[0], y - box[1]), txt, font=font, fill=col)
    return np.asarray(img).copy()


# ---------------------------------------------------------------- the run

def run_timeline(model, clip, vae, audio_vae, timeline, prompt, beat, shot_frames, bridge_frames, width, height,
                 steps, sampler_name, scheduler, seed, ref_image_size, pin_mode="mask", soundtrack="generated",
                 upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4,
                 frame_labels=False, pictures=(), ref_video=None, ref_audio=None, sampler=None, sigmas=None):
    import sys

    import comfy.model_management
    import comfy.nested_tensor
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import folder_paths
    import latent_preview
    import node_helpers
    import nodes
    from comfy.ldm.minimax.model import FRAME_RESCALE
    from comfy_extras.nodes_audio import vae_decode_audio
    from comfy_extras.nodes_custom_sampler import Guider_Basic
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    NT = comfy.nested_tensor.NestedTensor
    tag = "[H3 Rolling Timeline]"
    lab = _lab()
    pin_mode = lab.get("pin_mode", pin_mode)
    pieces, notes = parse_timeline(timeline, prompt, beat, shot_frames, bridge_frames)
    P = len(pieces)
    N = (P + 1) // 2
    starts, F = layout(pieces)
    w, h = max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)
    na = round(PIN * FRAME_RESCALE)
    _T[0] = 0.0
    _mark("start")

    def name(i):
        return f"shot {i // 2 + 1}" if i % 2 == 0 else f"bridge {(i + 1) // 2}"

    pics = [p for p in pictures if p is not None]
    for i, p in enumerate(pieces):
        p["pics"], p["video"], p["audio"] = pick_media(p["media"], len(pics), ref_video is not None,
                                                       ref_audio is not None, name(i))
        p["text"] = renumber_pictures(p["text"], p["pics"], len(pics), name(i))
        p["still"] = i > 0 and "{still}" in p["text"]
        if len(p["pics"]) + p["still"] > 9:
            raise ValueError(f"{name(i)}: H3 takes 9 pictures and this piece has {len(p['pics'])} plus the still")
    use_still = any(p["still"] for p in pieces)

    upscaler, up_model = None, None
    if upscale_pass:
        upscaler = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
        if upscaler is None:
            raise ValueError("upscale_pass needs the Minimax H3 Latent Upscaler node pack")
        names = [m for m in sys.modules[upscaler.__module__].scan_models() if not m.startswith("(")]
        if not names:
            raise ValueError("no latent upscaler models found")
        up_model = next((m for m in names if "3d" in m.lower()), names[0])

    def ts(f):
        return f"{f / FPS:.2f}s"

    rows = []
    for i, p in enumerate(pieces):
        s, ln = starts[i], p["frames"]
        row = f"  {name(i)}: frames {s}-{s + ln - 1} ({ts(s)}, {ln} frames)"
        if i % 2:
            row += f", free {s + PIN}-{s + ln - PIN - 1}"
        if p["media"] not in (None, "all"):
            got = [f"picture_{j + 1}" for j in p["pics"]] + ["video"] * p["video"] + ["audio"] * p["audio"]
            row += f" | media: {', '.join(got) or 'none'}"
        rows.append(row)
    shot_lens = sorted({p["frames"] for p in pieces[0::2]})
    bridge_lens = sorted({p["frames"] for p in pieces[1::2]})
    plan = (f"{N} shot(s) of {'/'.join(map(str, shot_lens))} frames"
            + (f" + {N - 1} bridge(s) of {'/'.join(map(str, bridge_lens))} frames" if N > 1 else "")
            + f" at {w}x{h} -> {F} frames ({F / FPS:.2f}s) | pins {PIN}+{PIN} ({pin_mode})"
            + (" | a frame of shot 1 is an extra picture for every later piece whose prompt says {still}"
               if use_still else "")
            + (f" | upscale to {upscale_megapixels} MP with {up_model}, {upscale_steps} steps @ {upscale_denoise}"
               if upscaler else "") + "\n" + "\n".join(rows + ["  note: " + n for n in notes]))
    print(f"{tag} {plan}")

    model_sampling = model.get_model_object("model_sampling")
    if sigmas is None:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()
    if sampler is None:
        sampler = comfy.samplers.sampler_object(sampler_name)
    up_sigmas = None
    if upscaler:
        total = max(upscale_steps, int(upscale_steps / max(upscale_denoise, 1e-3)))
        up_sigmas = comfy.samplers.calculate_sigmas(model_sampling, "linear_quadratic", total).cpu()[-(upscale_steps + 1):]

    def encode(p, still):
        images = [pics[j] for j in p["pics"]]
        still_tag = None
        if p["still"] and still is not None:
            images = images + [still]
            still_tag = f"<Picture {len(images)}>"
        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=with_still(p["text"], still_tag), width=w, height=h, length=p["frames"],
            ref_image_size=ref_image_size, vae=vae, audio_vae=audio_vae,
            ref_images={f"ref_image_{j + 1}": im for j, im in enumerate(images)} or None,
            ref_videos={"ref_video_1": ref_video} if p["video"] else None,
            ref_audios={"ref_audio_1": ref_audio} if p["audio"] else None)
        return out.args

    def sample(cond, latent_samples, sig, noise_seed, mask=None):
        guider = Guider_Basic(model)
        guider.set_conds(cond)
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed)
        callback = latent_preview.prepare_callback(model, sig.shape[-1] - 1, {})
        result = guider.sample(noise, latent_samples, sampler, sig, denoise_mask=mask, callback=callback,
                               disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise_seed)
        return result.to(comfy.model_management.intermediate_device())

    def upscale(samples):
        video_lat, audio_lat = samples.unbind()
        res = upscaler.execute(latent={"samples": video_lat}, model_name=up_model,
                               mode={"mode": "megapixels", "megapixels": upscale_megapixels}, align=32,
                               enable_temporal_chunking=True, force_unload=True, device="cuda", precision="fp16")
        res = getattr(res, "args", None) or res.result
        return NT((res[0]["samples"], audio_lat))

    def held(samples, a_lat, a_len, b_lat, bl):
        """mask pins: the left shot's last 5 frames and the right shot's first 5 written straight into the
        bridge's own starting latent and masked 0, so the sampler keeps them and they decode as the shots'
        own frames (roycho87's Geiru Context / Anchoring method, here into a piece nothing is rendered from)."""
        v, a = [t.clone() for t in samples.unbind()[:2]]
        av, aa = a_lat.unbind()[:2]
        bv, ba = b_lat.unbind()[:2]
        mv, ma = torch.ones_like(v), torch.ones_like(a)
        n = video_latent_t(PIN)
        t = v.shape[2]
        v[:, :, :n] = av[:, :, -n:].to(v)
        mv[:, :, :n] = 0
        v[:, :, t - n:] = bv[:, :, :n].to(v)
        mv[:, :, t - n:] = 0
        a0 = round((a_len - PIN) * FRAME_RESCALE)
        src = aa[..., a0:a0 + na]
        k0 = min(src.shape[-1], a.shape[-1])
        a[..., :k0] = src[..., :k0].to(a)
        ma[..., :k0] = 0
        b0 = round(FRAME_RESCALE * (bl - PIN))
        k1 = max(0, min(na, a.shape[-1] - b0, ba.shape[-1]))
        if k1:
            a[..., b0:b0 + k1] = ba[..., :k1].to(a)
            ma[..., b0:b0 + k1] = 0
        return NT((v, a)), NT((mv, ma))

    def pins(cond, a_lat, a_len, b_lat, bl, bridge_audio_t):
        """keyframe pins: the same latents passed as H3 keyframe guides; the bridge draws its own copy."""
        av, aa = a_lat.unbind()[:2]
        bv, ba = b_lat.unbind()[:2]
        n = video_latent_t(PIN)
        a0 = round((a_len - PIN) * FRAME_RESCALE)
        kfs = list(cond[0][1].get("minimax_keyframes", []))
        kfs.append({"resolved_frame_index": 0, "latent": av[:, :, -n:].clone(),
                    "audio_latent": aa[..., a0:a0 + na].clone()})
        end = {"resolved_frame_index": bl - PIN, "latent": bv[:, :, :n].clone()}
        room = math.floor(bridge_audio_t - FRAME_RESCALE * (bl - PIN))
        if room > 0:
            end["audio_latent"] = ba[..., :min(na, room)].clone()
        kfs.append(end)
        return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": kfs})

    def render_shot(i, enc):
        cond, latent = enc
        k = i // 2
        own = pieces[i]["seed"]
        s_lo = seed + k if own is None else int(own)  # own seed: same prompt + same seed = the same clip
        lo = sample(cond, latent["samples"], sigmas, s_lo)
        _mark(f"{name(i)} sample")
        hi = None
        if upscaler:
            hi = sample(cond, upscale(lo), up_sigmas, s_lo + 100000)
            _mark(f"{name(i)} upscale+polish")
        return {"lo": lo, "hi": hi, "len": pieces[i]["frames"]}

    def render_bridge(i, enc, a, b):
        cond, latent = enc
        j = (i - 1) // 2
        bl = pieces[i]["frames"]
        at = latent["samples"].unbind()[1].shape[-1]
        own = pieces[i]["seed"]
        s_lo = seed + 50000 + j if own is None else int(own)
        s_hi = s_lo + 100000
        if pin_mode == "mask":
            start, mask = held(latent["samples"], a["lo"], a["len"], b["lo"], bl)
            lo = sample(cond, start, sigmas, s_lo, mask)
        else:
            lo = sample(pins(cond, a["lo"], a["len"], b["lo"], bl, at), latent["samples"], sigmas, s_lo)
        _mark(f"{name(i)} sample")
        hi = None
        if upscaler:
            big = upscale(lo)
            if pin_mode == "mask":
                big, mask = held(big, a["hi"], a["len"], b["hi"], bl)
                hi = sample(cond, big, up_sigmas, s_hi, mask)
            else:
                hi = sample(pins(cond, a["hi"], a["len"], b["hi"], bl, at), big, up_sigmas, s_hi)
            _mark(f"{name(i)} upscale+polish")
        return {"lo": lo, "hi": hi, "len": bl}

    def decode(r):
        samples = r["hi"] if r["hi"] is not None else r["lo"]
        imgs = nodes.VAEDecode().decode(vae, {"samples": samples})[0].float().cpu()
        aud = vae_decode_audio(audio_vae, {"samples": samples})
        return imgs, aud["waveform"].float().cpu(), aud["sample_rate"]

    st = {"sink": None, "audio": None, "sr": None, "stamped": 0}
    fade = 0 if pin_mode == "mask" else PIN  # masked pins decode as the same frames, so the joins are plain cuts

    def place(i, r, dec=None):
        nm, s, first = name(i), starts[i], i == 0
        imgs, wf, sr = dec if dec is not None else decode(r)
        _mark(f"{nm} decode")
        if st["sink"] is None:
            hh, ww = imgs.shape[1], imgs.shape[2]
            st["sink"] = FrameSink(F, (hh, ww), _stream_path(lab, F, hh, ww))
            st["sr"] = sr
            st["audio"] = torch.zeros((1, wf.shape[1], round(F / FPS * sr)), dtype=torch.float32)
        seg = imgs[:max(0, min(imgs.shape[0], r["len"], F - s))]
        if not first:
            m = min(PIN, seg.shape[0])
            d = (st["sink"].get(s, s + m) - seg[:m].float()).abs().mean(dim=(1, 2, 3)) * 255
            print(f"{tag} join {nm} @ {s}: mean abs diff over the {m} shared frames "
                  f"{' '.join(f'{x:.2f}' for x in d.tolist())} (0-255)")
        st["sink"].place(0 if first or not fade else 1, s, seg, fade)
        del imgs, seg
        out = st["audio"]
        spf = st["sr"] / FPS
        a0 = round(s * spf)
        want = max(0, min(round(r["len"] * spf), out.shape[-1] - a0))
        seg_a = wf[..., :want]
        if seg_a.shape[-1] < want:
            seg_a = torch.nn.functional.pad(seg_a, (0, want - seg_a.shape[-1]))
        ab = 0 if first else min(max(round(fade * spf), round(0.05 * st["sr"])), want)
        if ab:
            ramp = torch.linspace(0, 1, ab + 2)[1:-1]
            out[..., a0:a0 + ab] = out[..., a0:a0 + ab] * (1 - ramp) + seg_a[..., :ab] * ramp
        out[..., a0 + ab:a0 + want] = seg_a[..., ab:]

    def colorfix(i, br_dec, b_imgs):
        """Ease the bridge's free frames onto the real frames at both ends: a per-channel quantile curve where the
        free part starts (vs the left shot's frames just before it) and one where it ends (vs the right shot's
        first frames), blended linearly across the free part. Takes out the brightness step where free frames
        meet pinned ones."""
        imgs = br_dec[0]
        b0, f0, f1, k = starts[i], PIN, pieces[i]["frames"] - PIN, 3
        qs = torch.linspace(0, 1, 65)

        def curve(src, ref):
            src = src[:, ::4, ::4].reshape(-1, 3).float()
            ref = ref[:, ::4, ::4].reshape(-1, 3).float()
            return [(torch.quantile(src[:, ch], qs).numpy(), torch.quantile(ref[:, ch], qs).numpy()) for ch in range(3)]

        ca = curve(imgs[f0:f0 + k], st["sink"].get(b0 + f0 - k, b0 + f0))
        cb = curve(imgs[f1 - k:f1], b_imgs[:k])

        # The ease is for the small brightness step between frames of one continuous picture. If an end of
        # the bridge shows a different picture from its shot (the bridge cut to another framing), the curve
        # would "correct" one image toward another, so that end is left alone.
        def sane(c, end):
            shift = float(np.mean([np.abs(y - x).mean() for x, y in c])) * 255
            if shift <= 6:
                return c
            print(f"{tag} {name(i)}: its {end} frames differ from the shot's by {shift:.1f} levels, which is a "
                  f"different picture, not a brightness step; no color ease at that end")
            return [(x, x) for x, _ in c]

        ca, cb = sane(ca, "first free"), sane(cb, "last free")
        out = imgs.clone()
        n = f1 - f0
        for f in range(f0, f1):
            wt = (f - f0 + 0.5) / n
            x = imgs[f].numpy()
            for ch in range(3):
                ya = np.interp(x[..., ch], ca[ch][0], ca[ch][1])
                yb = np.interp(x[..., ch], cb[ch][0], cb[ch][1])
                out[f, ..., ch] = torch.from_numpy((1 - wt) * ya + wt * yb)
        a_gain = (out[f0:f0 + k] - imgs[f0:f0 + k]).mean().item() * 255
        b_gain = (out[f1 - k:f1] - imgs[f1 - k:f1]).mean().item() * 255
        print(f"{tag} {name(i)} color eased onto its shots: {a_gain:+.2f} levels where it leaves {name(i - 1)}, "
              f"{b_gain:+.2f} where it reaches {name(i + 1)}")
        return (out.clamp(0, 1),) + tuple(br_dec[1:])

    def settle(upto):
        """Frames before `upto` can no longer change: label them (if asked) and let the sink write them out.
        Labels go on only here, so nothing that compares frames ever sees one."""
        sink = st["sink"]
        upto = min(upto, F)
        if frame_labels:
            for n in range(st["stamped"], upto):
                if sink.path is None:
                    arr = draw_label(sink._u8(sink.buf[n]).numpy(), n, pieces, starts)
                    sink.buf[n] = torch.from_numpy(arr).float() / 255
                else:
                    o = n - sink.p0
                    sink.frames[o] = torch.from_numpy(draw_label(sink.frames[o].numpy(), n, pieces, starts))
            st["stamped"] = max(st["stamped"], upto)
        sink.flush(upto)

    color = bool(lab.get("bridge_color", True))

    # shot 1: from the pictures alone
    prev = render_shot(0, encode(pieces[0], None))
    dec0 = decode(prev)
    still = None
    if use_still:
        mid = dec0[0].shape[0] // 2
        still = dec0[0][mid:mid + 1].clone()
        try:
            from PIL import Image
            d = os.path.join(folder_paths.get_output_directory(), "h3_rolling")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, time.strftime("timeline_%Y%m%d_%H%M%S_still.png"))
            Image.fromarray((still[0].clamp(0, 1) * 255).round().byte().numpy()).save(path)
            plan += f"\n  the still = frame {mid} of shot 1, saved as {path}"
        except Exception as e:
            print(f"{tag} couldn't save the still ({e})")
    place(0, prev, dec0)
    del dec0

    # every later piece is encoded now, in one go, so the text encoder comes back only once
    enc, seen = {}, {}
    for i in range(1, P):
        p = pieces[i]
        key = (p["text"], p["frames"], tuple(p["pics"]), p["video"], p["audio"])
        if key not in seen:  # pieces with the same words, length and media share one encode
            seen[key] = encode(p, still)
            _mark(f"encode piece {i + 1}/{P}")
        enc[i] = seen[key]

    for i in range(2, P, 2):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur = render_shot(i, enc[i])
        br = render_bridge(i - 1, enc[i - 1], prev, cur)
        if color:
            cur_dec = decode(cur)
            br_dec = colorfix(i - 1, decode(br), cur_dec[0])
            place(i - 1, br, br_dec)
            del br_dec
            place(i, cur, cur_dec)
            del cur_dec
        else:
            place(i - 1, br)
            place(i, cur)
        print(f"{tag} rendered {name(i)}/{N} + {name(i - 1)}")
        settle(starts[i + 1] if i < P - 1 else F)
        prev = cur
        del br

    settle(F)
    final_audio = None if soundtrack == "silent" else {"waveform": st["audio"], "sample_rate": st["sr"]}
    frames_out = st["sink"].finish(final_audio)
    if st["sink"].path:
        plan += f"\n  full video: {st['sink'].path}"
    return frames_out, final_audio, plan
