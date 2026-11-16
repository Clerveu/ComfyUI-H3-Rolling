"""H3 Rolling Anchors: a long H3 video from pictures and prompts where nothing is ever more than one
generation from a fresh render.

Short ANCHOR clips are rendered first, each on its own from the pictures and its own line of the
storyboard ("she stands at the window", "she sits on the couch"). Every stretch between two anchors is
then a long BRIDGE: a normal clip pinned at its start to the last 5 frames of one anchor and at its end
to the first 5 frames of the next, with a prompt that says how she gets from one to the other.

Anchors come from the pictures. Bridges come from two anchors. Nothing is rendered from a bridge.
"""
import math
import os
import re
import time

import numpy as np
import torch

from .rolling_core import FPS, FrameSink, _lab, _mark, _stream_path, _T, align, video_latent_t

PIN = 5  # frames a bridge shares with the anchor on each side (one 1-frame latent + one 4-frame latent)


def split_pieces(prompt, beats):
    """Piece prompts in timeline order: anchor, bridge, anchor, bridge ... anchor.

    Two ways to write them:
    - a template with {beat} in it + one beat per line in the beats box (anchor, move, anchor, move ... anchor)
    - whole prompts separated by a line of three or more dashes, in the same order (the beats box is ignored)
    """
    parts = [p.strip() for p in re.split(r"(?m)^\s*-{3,}\s*$", prompt)]
    parts = [p for p in parts if p]
    if len(parts) > 1:
        pieces = parts
    else:
        lines = [ln.strip() for ln in beats.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("the beats box is empty: one line per piece, going anchor, move, anchor, move ... anchor")
        if "{beat}" not in prompt:
            raise ValueError("the prompt needs {beat} where each piece's own action goes "
                             "(or write whole prompts separated by lines of ---)")
        pieces = [prompt.replace("{beat}", ln) for ln in lines]
    if len(pieces) % 2 == 0:
        raise ValueError(f"{len(pieces)} pieces: they go anchor, move, anchor, move ... and must end on an anchor, "
                         "so the count is odd")
    return pieces


def with_still(text, tag):
    """{still} is the tag of the frame taken from anchor 1. Lines that mention it only exist once that
    frame exists: for anchor 1 itself (tag None) they are dropped."""
    if "{still}" not in text:
        return text
    if tag is None:
        return "\n".join(ln for ln in text.splitlines() if "{still}" not in ln)
    return text.replace("{still}", tag)


def run_anchors(model, clip, vae, audio_vae, prompt, beats, width, height, anchor_frames, clip_seconds, steps,
                sampler_name, scheduler, seed, ref_image_size, pin_mode="mask", soundtrack="generated",
                upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4,
                pictures=(), sampler=None, sigmas=None):
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
    lab = _lab()
    pin_mode = lab.get("pin_mode", pin_mode)
    pieces = split_pieces(prompt, beats)
    N = (len(pieces) + 1) // 2
    A = align(int(anchor_frames))
    BL = align(round(clip_seconds * FPS))
    if BL < 2 * PIN + 12:
        BL = align(2 * PIN + 12)
    step = A - 2 * PIN + BL                      # anchor k starts here after anchor k-1
    starts = [k * step for k in range(N)]
    bstarts = [s + A - PIN for s in starts[:-1]]
    F = starts[-1] + A
    w, h = max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)
    na = round(PIN * FRAME_RESCALE)
    _T[0] = 0.0
    _mark("start")

    pics = [p for p in pictures if p is not None]
    use_still = any("{still}" in p for p in pieces) and N > 1
    still_tag = f"<Picture {len(pics) + 1}>" if use_still else None

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
    for k in range(N):
        rows.append(f"  anchor {k + 1}: frames {starts[k]}-{starts[k] + A - 1} ({ts(starts[k])})")
        if k < N - 1:
            rows.append(f"  bridge {k + 1}: frames {bstarts[k]}-{bstarts[k] + BL - 1} ({ts(bstarts[k])}), "
                        f"free {bstarts[k] + PIN}-{bstarts[k] + BL - PIN - 1}")
    plan = (f"{N} anchor(s) of {A} frames + {N - 1} bridge(s) of {BL} frames ({BL / FPS:.2f}s) at {w}x{h} -> "
            f"{F} frames ({F / FPS:.2f}s) | pins {PIN}+{PIN} ({pin_mode})"
            + (f" | a frame of anchor 1 is {still_tag} for every later piece" if use_still else "")
            + (f" | upscale to {upscale_megapixels} MP with {up_model}, {upscale_steps} steps @ {upscale_denoise}"
               if upscaler else "") + "\n" + "\n".join(rows))
    print(f"[H3 Rolling Anchors] {plan}")

    model_sampling = model.get_model_object("model_sampling")
    if sigmas is None:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()
    if sampler is None:
        sampler = comfy.samplers.sampler_object(sampler_name)
    up_sigmas = None
    if upscaler:
        total = max(upscale_steps, int(upscale_steps / max(upscale_denoise, 1e-3)))
        up_sigmas = comfy.samplers.calculate_sigmas(model_sampling, "linear_quadratic", total).cpu()[-(upscale_steps + 1):]

    def encode(text, length, images):
        ref_images = {f"ref_image_{i + 1}": p for i, p in enumerate(images)} or None
        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=text, width=w, height=h, length=length, ref_image_size=ref_image_size,
            vae=vae, audio_vae=audio_vae, ref_images=ref_images)
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

    def held(samples, a_lat, b_lat):
        """mask pins: anchor A's last 5 frames and anchor B's first 5 written straight into the bridge's own
        starting latent and masked 0, so the sampler keeps them and they decode as the anchors' own frames
        (roycho87's Geiru Context / Anchoring method, here into a piece nothing is rendered from)."""
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
        a0 = round((A - PIN) * FRAME_RESCALE)
        src = aa[..., a0:a0 + na]
        k0 = min(src.shape[-1], a.shape[-1])
        a[..., :k0] = src[..., :k0].to(a)
        ma[..., :k0] = 0
        b0 = round(FRAME_RESCALE * (BL - PIN))
        k1 = max(0, min(na, a.shape[-1] - b0, ba.shape[-1]))
        if k1:
            a[..., b0:b0 + k1] = ba[..., :k1].to(a)
            ma[..., b0:b0 + k1] = 0
        return NT((v, a)), NT((mv, ma))

    def pins(cond, a_lat, b_lat, bridge_audio_t):
        """keyframe pins: the same latents passed as H3 keyframe guides; the bridge draws its own copy."""
        av, aa = a_lat.unbind()[:2]
        bv, ba = b_lat.unbind()[:2]
        n = video_latent_t(PIN)
        a0 = round((A - PIN) * FRAME_RESCALE)
        kfs = list(cond[0][1].get("minimax_keyframes", []))
        kfs.append({"resolved_frame_index": 0, "latent": av[:, :, -n:].clone(),
                    "audio_latent": aa[..., a0:a0 + na].clone()})
        end = {"resolved_frame_index": BL - PIN, "latent": bv[:, :, :n].clone()}
        room = math.floor(bridge_audio_t - FRAME_RESCALE * (BL - PIN))
        if room > 0:
            end["audio_latent"] = ba[..., :min(na, room)].clone()
        kfs.append(end)
        return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": kfs})

    def render_anchor(k, enc):
        cond, latent = enc
        lo = sample(cond, latent["samples"], sigmas, seed)
        _mark(f"anchor {k + 1} sample")
        hi = None
        if upscaler:
            hi = sample(cond, upscale(lo), up_sigmas, seed + 100000)
            _mark(f"anchor {k + 1} upscale+polish")
        return {"lo": lo, "hi": hi, "len": A}

    def render_bridge(j, enc, a, b):
        cond, latent = enc
        at = latent["samples"].unbind()[1].shape[-1]
        s_lo, s_hi = seed + 50000 + j, seed + 150000 + j
        if pin_mode == "mask":
            start, mask = held(latent["samples"], a["lo"], b["lo"])
            lo = sample(cond, start, sigmas, s_lo, mask)
        else:
            lo = sample(pins(cond, a["lo"], b["lo"], at), latent["samples"], sigmas, s_lo)
        _mark(f"bridge {j + 1} sample")
        hi = None
        if upscaler:
            big = upscale(lo)
            if pin_mode == "mask":
                big, mask = held(big, a["hi"], b["hi"])
                hi = sample(cond, big, up_sigmas, s_hi, mask)
            else:
                hi = sample(pins(cond, a["hi"], b["hi"], at), big, up_sigmas, s_hi)
            _mark(f"bridge {j + 1} upscale+polish")
        return {"lo": lo, "hi": hi, "len": BL}

    def decode(r):
        samples = r["hi"] if r["hi"] is not None else r["lo"]
        imgs = nodes.VAEDecode().decode(vae, {"samples": samples})[0].float().cpu()
        aud = vae_decode_audio(audio_vae, {"samples": samples})
        return imgs, aud["waveform"].float().cpu(), aud["sample_rate"]

    st = {"sink": None, "audio": None, "sr": None}
    fade = 0 if pin_mode == "mask" else PIN  # masked pins decode as the same frames, so the joins are plain cuts

    def place(name, s, r, first, dec=None):
        imgs, wf, sr = dec if dec is not None else decode(r)
        _mark(f"{name} decode")
        if st["sink"] is None:
            hh, ww = imgs.shape[1], imgs.shape[2]
            st["sink"] = FrameSink(F, (hh, ww), _stream_path(lab, F, hh, ww))
            st["sr"] = sr
            st["audio"] = torch.zeros((1, wf.shape[1], round(F / FPS * sr)), dtype=torch.float32)
        seg = imgs[:max(0, min(imgs.shape[0], r["len"], F - s))]
        if not first:
            m = min(PIN, seg.shape[0])
            d = (st["sink"].get(s, s + m) - seg[:m].float()).abs().mean(dim=(1, 2, 3)) * 255
            print(f"[H3 Rolling Anchors] join {name} @ {s}: mean abs diff over the {m} shared frames "
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

    def colorfix(j, br_dec, b_imgs):
        """Ease the bridge's free frames onto the real frames at both ends: a per-channel quantile curve where the
        free part starts (vs anchor A's frames just before it) and one where it ends (vs anchor B's first frames),
        blended linearly across the free part. Takes out the brightness step where free frames meet pinned ones."""
        imgs = br_dec[0]
        b0, f0, f1, k = bstarts[j], PIN, BL - PIN, 3
        qs = torch.linspace(0, 1, 65)

        def curve(src, ref):
            src = src[:, ::4, ::4].reshape(-1, 3).float()
            ref = ref[:, ::4, ::4].reshape(-1, 3).float()
            return [(torch.quantile(src[:, ch], qs).numpy(), torch.quantile(ref[:, ch], qs).numpy()) for ch in range(3)]

        ca = curve(imgs[f0:f0 + k], st["sink"].get(b0 + f0 - k, b0 + f0))
        cb = curve(imgs[f1 - k:f1], b_imgs[:k])
        out = imgs.clone()
        n = f1 - f0
        for i in range(f0, f1):
            wt = (i - f0 + 0.5) / n
            x = imgs[i].numpy()
            for ch in range(3):
                ya = np.interp(x[..., ch], ca[ch][0], ca[ch][1])
                yb = np.interp(x[..., ch], cb[ch][0], cb[ch][1])
                out[i, ..., ch] = torch.from_numpy((1 - wt) * ya + wt * yb)
        a_gain = (out[f0:f0 + k] - imgs[f0:f0 + k]).mean().item() * 255
        b_gain = (out[f1 - k:f1] - imgs[f1 - k:f1]).mean().item() * 255
        print(f"[H3 Rolling Anchors] bridge {j + 1} color eased onto its anchors: {a_gain:+.2f} levels where it "
              f"leaves anchor {j + 1}, {b_gain:+.2f} where it reaches anchor {j + 2}")
        return (out.clamp(0, 1),) + tuple(br_dec[1:])

    color = bool(lab.get("bridge_color", True))

    # anchor 1: from the pictures alone
    prev = render_anchor(0, encode(with_still(pieces[0], None), A, pics))
    dec0 = decode(prev)
    later = pics
    if use_still:
        mid = dec0[0].shape[0] // 2
        still = dec0[0][mid:mid + 1].clone()
        later = pics + [still]
        try:
            from PIL import Image
            d = os.path.join(folder_paths.get_output_directory(), "h3_rolling")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, time.strftime("anchors_%Y%m%d_%H%M%S_still.png"))
            Image.fromarray((still[0].clamp(0, 1) * 255).round().byte().numpy()).save(p)
            plan += f"\n  {still_tag} = frame {mid} of anchor 1, saved as {p}"
        except Exception as e:
            print(f"[H3 Rolling Anchors] couldn't save the still ({e})")
    place("anchor 1", 0, prev, True, dec0)
    del dec0

    # every later piece is encoded now, in one go, so the text encoder comes back only once
    enc = {}
    for i in range(1, len(pieces)):
        enc[i] = encode(with_still(pieces[i], still_tag), A if i % 2 == 0 else BL, later)
        _mark(f"encode piece {i + 1}/{len(pieces)}")

    for k in range(1, N):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur = render_anchor(k, enc[2 * k])
        br = render_bridge(k - 1, enc[2 * k - 1], prev, cur)
        if color:
            cur_dec = decode(cur)
            br_dec = colorfix(k - 1, decode(br), cur_dec[0])
            place(f"bridge {k}", bstarts[k - 1], br, False, br_dec)
            del br_dec
            place(f"anchor {k + 1}", starts[k], cur, False, cur_dec)
            del cur_dec
        else:
            place(f"bridge {k}", bstarts[k - 1], br, False)
            place(f"anchor {k + 1}", starts[k], cur, False)
        print(f"[H3 Rolling Anchors] rendered anchor {k + 1}/{N} + bridge {k}")
        st["sink"].flush(bstarts[k] if k < N - 1 else F)
        prev = cur
        del br

    final_audio = None if soundtrack == "silent" else {"waveform": st["audio"], "sample_rate": st["sr"]}
    frames_out = st["sink"].finish(final_audio)
    if st["sink"].path:
        plan += f"\n  full video: {st['sink'].path}"
    return frames_out, final_audio, plan
