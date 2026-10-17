"""H3 Rolling Ref2VA: the actual work, reloaded on every run so edits apply without a restart.

load_source: decode a long video to 24 fps frames (+ soundtrack) between two timestamps.
run_rolling: render the source in fixed-length chunks, each with its own slice of the source
as <Video 1>. In the default bridge hand-off every chunk is rendered independently and each
seam gets a short bridge clip pinned to the end of one chunk and the start of the next, so
nothing is ever more than one generation from a fresh render. The older pixel/latent hand-offs
chain each chunk off the last one instead.
"""

import math

import numpy as np
import torch

FPS = 24


def align(n):
    """Snap a frame count up to H3's 17k+5 grid."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


# ---------------------------------------------------------------- source

def load_source(path, start, end, short_edge):
    import av

    start = max(0.0, float(start or 0.0))
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        dur = None
        if c.duration:
            dur = c.duration / av.time_base
        elif vs.duration and vs.time_base:
            dur = float(vs.duration * vs.time_base)
        end_t = float(end) if end and end > start else dur
        src_w, src_h = vs.codec_context.width, vs.codec_context.height
        if short_edge and min(src_w, src_h) > short_edge:
            s = short_edge / min(src_w, src_h)
            w, h = max(2, round(src_w * s / 2) * 2), max(2, round(src_h * s / 2) * 2)
        else:
            w, h = src_w - src_w % 2, src_h - src_h % 2
        n_slots = math.floor((end_t - start) * FPS + 1e-6) if end_t is not None else None
        if start > 0:
            c.seek(int(start * av.time_base), backward=True, any_frame=False)

        def rgb(frame):
            try:
                f = frame.reformat(width=w, height=h, format="rgb24", interpolation="AREA")
            except TypeError:
                f = frame.reformat(width=w, height=h, format="rgb24")
            return f.to_ndarray()

        out = []
        prev, prev_rgb, pending = None, None, 0
        gi = 0
        done = False
        for frame in c.decode(vs):
            ft = frame.time
            if ft is None:
                continue
            # sample-and-hold: every grid slot before this frame shows the previous frame
            while (n_slots is None or gi < n_slots) and start + gi / FPS < ft - 1e-6:
                if prev is None:
                    pending += 1
                else:
                    if prev_rgb is None:
                        prev_rgb = rgb(prev)
                    out.append(prev_rgb)
                gi += 1
            if n_slots is not None and gi >= n_slots:
                done = True
                break
            prev, prev_rgb = frame, None
            if pending:
                prev_rgb = rgb(prev)
                out.extend([prev_rgb] * pending)
                pending = 0
        if prev is not None and not done:
            # video ended before `end`: hold the last frame for the remaining slots
            last = prev_rgb if prev_rgb is not None else rgb(prev)
            remaining = (n_slots - gi) if n_slots is not None else 1
            out.extend([last] * max(0, remaining))

    if len(out) < 5:
        raise ValueError(f"H3 Rolling Source: only {len(out)} frames between {start}s and {end_t}s")
    frames = torch.from_numpy(np.stack(out)).float().div_(255.0)

    audio = _load_audio(path, start, len(out) / FPS)
    info = (f"{len(out)} frames @ {FPS} fps = {len(out) / FPS:.2f}s, {w}x{h} (source {src_w}x{src_h}), "
            f"{start:.2f}s -> {start + len(out) / FPS:.2f}s, "
            f"{'with' if audio else 'no'} audio")
    print(f"[H3 Rolling Source] {info}")
    return frames, audio, info


def _load_audio(path, start, seconds):
    import av

    with av.open(path) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[0]
        sr = a.codec_context.sample_rate or 44100
        res = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
        if start > 0:
            c.seek(int(start * av.time_base), backward=True, any_frame=False)
        chunks, first_t = [], None
        stop_t = start + seconds + 1.0
        for fr in c.decode(a):
            if first_t is None and fr.time is not None:
                first_t = fr.time
            for rf in res.resample(fr):
                chunks.append(rf.to_ndarray())
            if fr.time is not None and fr.time > stop_t:
                break
        for rf in res.resample(None):
            chunks.append(rf.to_ndarray())
    if not chunks:
        return None
    arr = np.concatenate(chunks, axis=1).astype(np.float32)
    off = int(round((start - (first_t or 0.0)) * sr))
    if off > 0:
        arr = arr[:, off:]
    elif off < 0:
        arr = np.pad(arr, ((0, 0), (-off, 0)))
    n = int(round(seconds * sr))
    arr = arr[:, :n]
    if arr.shape[1] < n:
        arr = np.pad(arr, ((0, 0), (0, n - arr.shape[1])))
    return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": sr}


# ---------------------------------------------------------------- rolling render

def plan_chunks(F, L, ov):
    """Chunk starts covering F source frames with length-L chunks overlapping >= ov frames."""
    if F <= L:
        return [0]
    n = math.ceil((F - L) / (L - ov - 1)) + 1
    return [round(k * (F - L) / (n - 1)) for k in range(n)]


def plan_aligned(F, L, ov):
    """Chunks stepping exactly L - ov frames, so each seam pins the previous chunk's last ov frames.
    L and ov are both 17k+5, so the step is a multiple of 17 and the pinned frames are whole
    latent tokens at the end of the previous chunk's latent. The last chunk is shortened to fit."""
    if F <= L:
        return [0], [align(F)]
    starts = [0]
    while starts[-1] + L < F:
        starts.append(starts[-1] + L - ov)
    lens = [L] * len(starts)
    lens[-1] = align(F - starts[-1])
    return starts, lens


def plan_bridged(F, L, Bl, pin=5, pin_end=5):
    """Independent chunks joined by short bridges. A bridge pins chunk A's last `pin` frames at its start
    and chunk B's first `pin_end` frames at its end, so B starts at bridge start + Bl - pin_end.
    If the clip ends inside what would be the next bridge, the last piece is a tail bridge pinned
    at its start only. Returns chunk starts, chunk lengths, and (start, length, end_pinned) per bridge."""
    if F <= L:
        return [0], [align(F)], []
    step = L - pin - pin_end + Bl
    starts, tail = [0], None
    while starts[-1] + L < F:
        nxt = starts[-1] + step
        if nxt + pin_end + 12 > F:
            b0 = starts[-1] + L - pin
            tail = (b0, align(max(F - b0, 22)), False)
            break
        starts.append(nxt)
    lens = [L] * len(starts)
    if starts[-1] + L > F:
        lens[-1] = align(F - starts[-1])
    bridges = [(s0 + L - pin, Bl, True) for s0 in starts[:-1]] + ([tail] if tail else [])
    return starts, lens, bridges


def video_latent_t(frames):
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def _lab():
    """Experimental switches from lab.json next to this file (read every run, no restart needed)."""
    import json
    import os
    try:
        with open(os.path.join(os.path.dirname(__file__), "lab.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _audio_slice(audio, f0, f1):
    sr = audio["sample_rate"]
    wf = audio["waveform"]
    a0, a1 = round(f0 / FPS * sr), round(f1 / FPS * sr)
    seg = wf[..., a0:a1]
    if seg.shape[-1] < sr // 10:
        return None
    return {"waveform": seg, "sample_rate": sr}


class FrameSink:
    """Assembles the output video. Normal mode: one float32 [F,H,W,3] buffer, returned as IMAGE.
    Stream mode (output too big for RAM): frames that can no longer change are encoded straight to
    an mp4 as soon as each chunk lands, only the still-open tail stays in memory, and the node
    returns a small proxy video (240p) so the graph still has something to show.
    In stream mode the tail is a list of uint8 frames: growing it appends frames and flushing drops
    them, so nothing ever copies the whole tail (the old fp16 tensor + torch.cat doubled it)."""

    PROXY_SHORT = 240

    def __init__(self, F, hw, stream_path=None):
        self.F, (self.H, self.W) = F, hw
        self.path = stream_path
        self.p0 = 0
        if stream_path is None:
            self.buf = torch.zeros((F, self.H, self.W, 3), dtype=torch.float32)
        else:
            import subprocess
            import shutil
            self.frames = []
            self.proxy = []
            self.tmp = stream_path[:-4] + ".video.mp4"
            exe = shutil.which("ffmpeg")
            if exe is None:
                raise RuntimeError("streaming the output needs ffmpeg on PATH")
            self.exe = exe
            self.proc = subprocess.Popen(
                [exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}",
                 "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
                 "-pix_fmt", "yuv420p", self.tmp],
                stdin=subprocess.PIPE)

    @staticmethod
    def _u8(x):
        return (x.float().clamp(0, 1) * 255).round().to(torch.uint8)

    def _ensure(self, end):
        if self.path is None:
            return
        end = min(end, self.F)
        while self.p0 + len(self.frames) < end:
            self.frames.append(torch.zeros((self.H, self.W, 3), dtype=torch.uint8))

    def get(self, a, b):
        if self.path is None:
            return self.buf[a:b].float()
        return torch.stack(self.frames[a - self.p0:b - self.p0]).float() / 255

    def place(self, k, s, seg, ov):
        n = min(seg.shape[0], self.F - s)
        b = 0 if k == 0 else min(ov, n)
        wts = torch.linspace(0, 1, b + 2)[1:-1].view(-1, 1, 1, 1)
        if self.path is None:
            if b:
                self.buf[s:s + b] = self.buf[s:s + b] * (1 - wts) + seg[:b].float() * wts
            self.buf[s + b:s + n] = seg[b:n]
            return
        self._ensure(s + n)
        o = s - self.p0
        if b:
            mixed = self.get(s, s + b) * (1 - wts) + seg[:b].float() * wts
            for i in range(b):
                self.frames[o + i] = self._u8(mixed[i])
        for i in range(b, n):
            self.frames[o + i] = self._u8(seg[i])

    def flush(self, upto):
        """Stream mode: frames before `upto` are final; encode them and drop them from RAM."""
        if self.path is None or upto <= self.p0:
            return
        n = upto - self.p0
        s = self.PROXY_SHORT / min(self.H, self.W)
        ph, pw = max(2, round(self.H * s / 2) * 2), max(2, round(self.W * s / 2) * 2)
        for i in range(0, n, 16):
            blk = torch.stack(self.frames[i:min(i + 16, n)])
            self.proc.stdin.write(blk.numpy().tobytes())
            small = torch.nn.functional.interpolate(blk.movedim(-1, 1).float() / 255, size=(ph, pw), mode="area")
            self.proxy.append(self._u8(small.movedim(1, -1)))
        self.frames = self.frames[n:]
        self.p0 = upto

    def finish(self, audio):
        if self.path is None:
            return self.buf
        import subprocess
        import wave
        self.flush(self.F)
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing " + self.tmp)
        if audio is not None:
            wav = self.path[:-4] + ".audio.wav"
            wf = audio["waveform"][0].float().clamp(-1, 1)
            with wave.open(wav, "wb") as w:
                w.setnchannels(wf.shape[0])
                w.setsampwidth(2)
                w.setframerate(int(audio["sample_rate"]))
                w.writeframes((wf.t() * 32767).round().to(torch.int16).numpy().tobytes())
            r = subprocess.run([self.exe, "-v", "error", "-y", "-i", self.tmp, "-i", wav, "-c:v", "copy",
                                "-c:a", "aac", "-b:a", "192k", "-shortest", self.path])
            if r.returncode == 0:
                import os
                os.remove(self.tmp)
                os.remove(wav)
            else:
                print(f"[H3 Rolling] audio mux failed; video-only file kept at {self.tmp}")
        else:
            import os
            os.replace(self.tmp, self.path)
        print(f"[H3 Rolling] full-resolution video written to {self.path}")
        return torch.cat(self.proxy).float() / 255


def _stream_path(lab, F, H, W):
    """Where to stream the output, or None to assemble it in RAM as usual."""
    import os
    import time
    want = lab.get("stream")
    need = F * H * W * 3 * 4
    if want is None:
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            avail = 16 * 2**30
        want = need > min(12 * 2**30, 0.35 * avail)
    if not want:
        return None
    import folder_paths
    d = os.path.join(folder_paths.get_output_directory(), "h3_rolling")
    os.makedirs(d, exist_ok=True)
    print(f"[H3 Rolling] output would need {need / 2**30:.1f} GB of RAM as one image batch; streaming it to disk instead")
    return os.path.join(d, time.strftime("rolling_%Y%m%d_%H%M%S.mp4"))


def _build_control(frames, lock):
    """Structure lock: SAM3 tracks the person being replaced, the rest of the frame becomes a Canny
    edge map, and the person's (grown) silhouette is blacked out of it. Fed to the H3 Fun ControlNet,
    it pins the background, framing and camera to the source while the person is drawn freely."""
    import nodes
    from comfy_extras.nodes_canny import Canny
    from comfy_extras.nodes_mask import GrowMask
    from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack

    def first(out):
        return (getattr(out, "args", None) or out.result)[0]

    sam_ckpt = lock.get("sam_ckpt", "sam3.1_multiplex_fp16.safetensors")
    track_text = lock.get("track", "person")
    threshold = float(lock.get("threshold", 0.4))
    max_objects = int(lock.get("max_objects", 2))
    sam = []  # loaded on the first window that isn't cached

    # SAM3 loses people over long runs and doesn't pick them back up, so track in short windows that
    # each start with a fresh detection; a lost track then only stays lost until the next window.
    # Windows are independent, so each one is cached by its pixels + settings: a preview of the first
    # part of a range is reused by the full run, and a rerun with only grow/strength changed skips SAM3.
    win = int(lock.get("window", 120))
    masks, hits = [], 0
    for a in range(0, frames.shape[0], win):
        clip = frames[a:a + win]
        key = _mask_key(clip, f"{sam_ckpt}|{track_text}|{threshold}|{max_objects}")
        m = _mask_cache_get(key, clip.shape[:3])
        if m is not None:
            hits += 1
        else:
            if not sam:
                sam_model, sam_clip = nodes.CheckpointLoaderSimple().load_checkpoint(sam_ckpt)[:2]
                sam.extend([sam_model, nodes.CLIPTextEncode().encode(sam_clip, track_text)[0]])
            track = first(SAM3_VideoTrack.execute(images=clip, model=sam[0], conditioning=sam[1],
                                                  detection_threshold=threshold, max_objects=max_objects,
                                                  detect_interval=1))
            m = first(SAM3_TrackToMask.execute(track_data=track, object_indices="")).cpu().float()
            if m.shape[0] != clip.shape[0]:
                raise RuntimeError(f"SAM3 returned {m.shape[0]} masks for {clip.shape[0]} frames")
            _mask_cache_put(key, m)
        masks.append(m)
    n_win = len(masks)
    print(f"[H3 Rolling] structure lock: {hits} of {n_win} tracking windows reused from cache, {n_win - hits} tracked")
    mask = torch.cat(masks)
    checks, flagged = _mask_checks(mask, win)
    lost = int((mask.flatten(1).amax(1) == 0).sum())
    if lost:
        print(f"[H3 Rolling] structure lock: no person found in {lost} of {mask.shape[0]} frames")
    mask = first(GrowMask.execute(mask=mask, expand=int(lock.get("grow", 40)), tapered_corners=True))
    lo, hi = lock.get("canny", [0.2, 0.5])
    edges = torch.cat([first(Canny.execute(image=frames[i:i + 64], low_threshold=lo, high_threshold=hi)).cpu()
                       for i in range(0, frames.shape[0], 64)])
    m = mask.cpu().float().clamp(0, 1)
    ctrl = edges * (1.0 - m).unsqueeze(-1)
    covered = float(m.mean())
    print(f"[H3 Rolling] structure lock: tracked '{lock.get('track', 'person')}', person mask covers "
          f"{covered * 100:.1f}% of the frame on average (grown {lock.get('grow', 40)} px)")
    _write_lock_preview(frames, m, ctrl, flagged, checks)
    return ctrl, checks


def _mask_cache_dir():
    import os
    d = os.path.join(os.path.dirname(__file__), "mask_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _mask_key(clip, settings):
    import hashlib
    h = hashlib.sha1(settings.encode())
    h.update(str(tuple(clip.shape)).encode())
    h.update((clip[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _mask_cache_get(key, shape):
    import os
    p = os.path.join(_mask_cache_dir(), key + ".npz")
    try:
        a = np.load(p)["m"]
    except (OSError, KeyError, ValueError):
        return None
    if tuple(a.shape) != tuple(shape):
        return None
    os.utime(p)  # newest-used survives pruning
    return torch.from_numpy(a.astype(np.float32) / 255.0)


def _mask_cache_put(key, m, budget=4 * 2**30):
    """Masks compress to almost nothing (mostly flat 0/1). Oldest-used files go past `budget` bytes."""
    import os
    d = _mask_cache_dir()
    a = (m.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    tmp = os.path.join(d, key + ".tmp.npz")
    np.savez_compressed(tmp, m=a)
    os.replace(tmp, os.path.join(d, key + ".npz"))
    files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")), key=os.path.getmtime)
    total = sum(os.path.getsize(f) for f in files)
    while files and total > budget:
        f = files.pop(0)
        total -= os.path.getsize(f)
        os.remove(f)


def _mask_checks(mask, win):
    """Spots in the tracked mask worth a look: nobody found, the size or position jumping between two
    frames (a cut, an occlusion, or SAM3 grabbing someone else), and window starts where the fresh
    detection doesn't line up with the end of the previous window. Returns (text, flagged frame list)."""
    n = mask.shape[0]
    b = mask > 0.5
    cov = b.flatten(1).float().mean(1)
    h, w = b.shape[1], b.shape[2]
    ys = torch.arange(h).view(1, h, 1).float()
    xs = torch.arange(w).view(1, 1, w).float()
    area = b.flatten(1).sum(1).clamp_min(1).float()
    cx = (b * xs).flatten(1).sum(1) / area / w
    cy = (b * ys).flatten(1).sum(1) / area / h
    reason = {}

    def flag(i, why):
        reason.setdefault(i, why)

    for i in range(n):
        if cov[i] == 0:
            flag(i, "nobody found")
    for i in range(1, n):
        a0, a1 = float(cov[i - 1]), float(cov[i])
        if a0 == 0 or a1 == 0:
            continue
        seam = i % win == 0
        inter = float((b[i - 1] & b[i]).sum())
        iou = inter / max(1.0, float((b[i - 1] | b[i]).sum()))
        jump = max(abs(float(cx[i] - cx[i - 1])), abs(float(cy[i] - cy[i - 1])))
        if seam and iou < 0.5:
            flag(i, "tracking restart doesn't match the previous window")
        elif max(a0, a1) / min(a0, a1) > 1.8 and max(a0, a1) > 0.005:
            flag(i, "mask size jumps")
        elif jump > 0.12:
            flag(i, "mask jumps across the frame")
    flagged = sorted(reason)
    spans = []
    for i in flagged:
        if spans and i - spans[-1][1] <= 12 and reason[spans[-1][0]] == reason[i]:
            spans[-1][1] = i
        else:
            spans.append([i, i])

    def ts(f):
        return f"{int(f / FPS // 60)}:{f / FPS % 60:05.2f}"

    if not spans:
        text = "structure lock check: nothing suspicious in the tracking"
    else:
        lines = [f"  {ts(s)}" + (f"-{ts(e)}" if e > s else "") + f"  {reason[s]}" for s, e in spans[:25]]
        if len(spans) > 25:
            lines.append(f"  ... and {len(spans) - 25} more")
        text = (f"structure lock check: {len(spans)} spot(s) worth a look (times from the start of the selection; "
                f"flagged frames have a yellow border in the preview):\n" + "\n".join(lines))
    print(f"[H3 Rolling] {text}")
    return text, flagged


def _write_lock_preview(frames, mask, ctrl, flagged=(), checks=""):
    """Side-by-side check video: source with the tracked person tinted red | the edge drawing H3 gets.
    Frames flagged by _mask_checks get a yellow border on the left half."""
    import os
    import shutil
    import subprocess
    import time
    import folder_paths
    exe = shutil.which("ffmpeg")
    if exe is None:
        print("[H3 Rolling] no ffmpeg on PATH; skipping the structure-lock preview")
        return
    d = os.path.join(folder_paths.get_output_directory(), "h3_rolling")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, time.strftime("lock_preview_%Y%m%d_%H%M%S.mp4"))
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    proc = subprocess.Popen([exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w * 2}x{h}",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                             "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    red = torch.tensor([1.0, 0.15, 0.15]).view(1, 1, 1, 3)
    yellow = torch.tensor([1.0, 0.85, 0.0])
    flagged = set(flagged)
    bw = max(4, h // 60)
    for i in range(0, n, 32):
        f = frames[i:i + 32, ..., :3].float().cpu()
        mk = mask[i:i + 32].unsqueeze(-1)
        tint = f * (1 - 0.45 * mk) + red * 0.45 * mk
        for j in range(tint.shape[0]):
            if i + j in flagged:
                t = tint[j]
                t[:bw], t[-bw:], t[:, :bw], t[:, -bw:] = yellow, yellow, yellow, yellow
        both = torch.cat([tint, ctrl[i:i + 32].float()], dim=2)
        proc.stdin.write((both.clamp(0, 1) * 255).round().to(torch.uint8).numpy().tobytes())
    proc.stdin.close()
    proc.wait()
    if checks:
        with open(path[:-4] + ".txt", "w", encoding="utf-8") as fh:
            fh.write(checks + "\n")
    print(f"[H3 Rolling] structure-lock preview written to {path}")


_T = [0.0]


def _mark(label):
    """Phase timing + VRAM snapshot, for benchmarking (shows up in Comfy's log)."""
    import time
    now = time.perf_counter()
    dt = now - _T[0] if _T[0] else 0.0
    _T[0] = now
    try:
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated() / 2**30
        print(f"[H3 Rolling][t] {label}: {dt:.2f}s | torch alloc {alloc:.1f}G, card free {free / 2**30:.1f}G")
    except Exception:
        print(f"[H3 Rolling][t] {label}: {dt:.2f}s")


def run_rolling(model, clip, vae, audio_vae, frames, prompt, megapixels, chunk_seconds, overlap_frames,
                steps, sampler_name, scheduler, seed, ref_image_size, soundtrack, source_audio_as_reference,
                audio=None, pictures=(), sampler=None, sigmas=None, refmod_conditioning=None,
                upscale_pass=False, upscale_megapixels=1.5, upscale_steps=2, upscale_denoise=0.4, options=None):
    import sys

    import comfy.model_management
    import comfy.nested_tensor
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
    import nodes
    from comfy_extras.nodes_audio import vae_decode_audio
    from comfy_extras.nodes_custom_sampler import Guider_Basic
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo

    F, H0, W0 = frames.shape[0], frames.shape[1], frames.shape[2]
    ar = W0 / H0
    area = megapixels * 1e6
    w = max(32, round(math.sqrt(area * ar) / 32) * 32)
    h = max(32, round(math.sqrt(area / ar) / 32) * 32)
    ov = int(overlap_frames)
    L = align(round(chunk_seconds * FPS))
    if F <= L:
        L = align(F)
    if ov >= L:
        raise ValueError(f"overlap ({ov}) must be shorter than the chunk ({L} frames)")
    # node widgets first, lab.json on top: any key in lab.json wins over the widget
    opts = options or {}
    lab = _lab()
    handoff = lab.get("handoff", opts.get("handoff", "pixel"))
    bridges = []
    BL = align(int(lab.get("bridge_frames", opts.get("bridge_frames", 39))))
    PE = align(int(lab.get("bridge_pin_end", 5)))
    if handoff == "bridge":
        if BL < PE + 17:
            BL = align(PE + 17)
        starts, lens, bridges = plan_bridged(F, L, BL, 5, PE)
    elif handoff in ("latent", "none") or lab.get("planner") == "aligned":
        starts, lens = plan_aligned(F, L, ov)
    else:
        starts = plan_chunks(F, L, ov)
        lens = [L] * len(starts)
    _T[0] = 0.0
    _mark("start")

    upscaler, up_model = None, None
    if upscale_pass:
        upscaler = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
        if upscaler is None:
            raise ValueError("upscale_pass needs the Minimax H3 Latent Upscaler node pack")
        names = [m for m in sys.modules[upscaler.__module__].scan_models() if not m.startswith("(")]
        if not names:
            raise ValueError("no latent upscaler models found")
        up_model = next((m for m in names if "3d" in m.lower()), names[0])

    refmod_blocks = []
    for entry in refmod_conditioning or []:
        for b in entry[1].get("minimax_refs", []):
            if b.get("refmod") and not any(b is x for x in refmod_blocks):
                refmod_blocks.append(b)

    ref_images = {f"ref_image_{i + 1}": p for i, p in enumerate(p for p in pictures if p is not None)} or None
    plan = (f"{F} frames ({F / FPS:.2f}s) -> {len(starts)} chunk(s) of {L} frames ({L / FPS:.2f}s) at {w}x{h}, "
            f"overlap {ov}, starts {starts}")
    if lens[-1] != L:
        plan += f", last chunk {lens[-1]} frames"
    if handoff == "latent":
        plan += " | latent hand-off"
    elif handoff == "none":
        plan += " | independent chunks (nothing inherited, crossfaded seams)"
    elif handoff == "bridge":
        plan += (f" | independent chunks + {len(bridges)} bridge(s) of {BL} frames, end pin {PE},"
                 f" at {[b for b, _, _ in bridges]}")
    if upscaler:
        plan += f" | upscale to {upscale_megapixels} MP with {up_model}, {upscale_steps} steps @ {upscale_denoise}"
    if refmod_blocks:
        plan += f" | {len(refmod_blocks)} RefMod block(s) on every chunk"
    lock = None
    if opts.get("structure_lock"):
        lock = {"track": opts.get("lock_track") or "person", "grow": opts.get("lock_grow", 40),
                "strength": opts.get("lock_strength", 1.0), "preview_only": opts.get("lock_preview_only", False),
                "max_objects": opts.get("lock_max_people", 2)}
    if "lock" in lab:  # a lock dict in lab.json turns it on and overrides fields; false/null turns it off
        lock = {**(lock or {}), **lab["lock"]} if lab["lock"] else None
    job_model = lambda s0, n: model
    if lock:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlNetApply
        from comfy_extras.nodes_model_patch import ModelPatchLoader
        ctrl, checks = _build_control(frames, lock)
        _mark("structure lock control video")
        if lock.get("preview_only"):
            print("[H3 Rolling] preview_only is on: stopping after the structure-lock preview, nothing rendered")
            return frames, audio, plan + " | structure-lock preview only (see output/h3_rolling)\n" + checks
        mpatch = ModelPatchLoader().load_model_patch(
            lock.get("patch", "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"))[0]
        lock_strength = float(lock.get("strength", 1.0))
        plan += f" | structure lock on '{lock.get('track', 'person')}' @ {lock_strength}\n{checks}"

        def job_model(s0, n):
            seg = ctrl[s0:s0 + n]
            if seg.shape[0] == 0:
                seg = ctrl[-1:]
            if seg.shape[0] < n:
                seg = torch.cat([seg, seg[-1:].expand(n - seg.shape[0], -1, -1, -1)])
            out = MiniMaxH3FunControlNetApply.execute(model=model, model_patch=mpatch, vae=vae, strength=lock_strength,
                                                      start_percent=0.0, end_percent=1.0, control_video=seg)
            return (getattr(out, "args", None) or out.result)[0]
    print(f"[H3 Rolling] {plan}")

    # 1) encode every chunk's conditioning up front so the text encoder loads once
    def encode(s, Lk):
        ref_audio = None
        if audio is not None and source_audio_as_reference:
            ref_audio = _audio_slice(audio, s, s + Lk)
        ref_vid = frames[s:s + Lk]
        if ref_vid.shape[0] == 0:  # a bridge plan can put the last chunk's start right at the end
            ref_vid = frames[-1:]
        if ref_vid.shape[0] < Lk:  # short last chunk rounded up to 17k+5: hold the last frame
            ref_vid = torch.cat([ref_vid, ref_vid[-1:].expand(Lk - ref_vid.shape[0], -1, -1, -1)])
        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=prompt, width=w, height=h, length=Lk, ref_image_size=ref_image_size,
            vae=vae, audio_vae=audio_vae, ref_images=ref_images,
            ref_videos={"ref_video_1": ref_vid},
            ref_video_audios={"ref_video_audio_1": ref_audio} if ref_audio else None)
        cond, latent = out.args
        if refmod_blocks:
            cond = [[c[0], dict(c[1], minimax_refs=list(c[1].get("minimax_refs", [])) + refmod_blocks)]
                    for c in cond]
        return cond, latent

    conds = []
    for k, s in enumerate(starts):
        conds.append(encode(s, lens[k]))
        print(f"[H3 Rolling] encoded chunk {k + 1}/{len(starts)}")
        _mark(f"encode chunk {k + 1}")
    bconds = []
    for j, (b0, bl, _) in enumerate(bridges):
        bconds.append(encode(b0, bl))
        _mark(f"encode bridge {j + 1}")

    # 2) render chunk by chunk, each pinned to the tail of what's already rendered
    model_sampling = model.get_model_object("model_sampling")
    if sigmas is None:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()
    if sampler is None:
        sampler = comfy.samplers.sampler_object(sampler_name)
    up_sigmas = None
    if upscaler:
        total = max(upscale_steps, int(upscale_steps / max(upscale_denoise, 1e-3)))
        up_sigmas = comfy.samplers.calculate_sigmas(model_sampling, "linear_quadratic", total).cpu()[-(upscale_steps + 1):]

    def sample(cond, latent_samples, sig, noise_seed, mdl=None):
        guider = Guider_Basic(mdl if mdl is not None else model)
        guider.set_conds(cond)
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed)
        callback = latent_preview.prepare_callback(model, sig.shape[-1] - 1, {})
        result = guider.sample(noise, latent_samples, sampler, sig, denoise_mask=None, callback=callback,
                               disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise_seed)
        return result.to(comfy.model_management.intermediate_device())

    state = {"sink": None, "audio": None, "sr": None, "lat": None, "lat_up": None, "len": None}
    import node_helpers
    from comfy.ldm.minimax.model import FRAME_RESCALE

    def pinned_latent(cond, prev):
        """Pin the previous chunk's last ov frames straight from its latent: the tail tokens of the
        video stream plus the matching span of the audio stream, no decode/re-encode."""
        vid, aud = prev.unbind()[:2]
        kf = {"resolved_frame_index": 0, "latent": vid[:, :, -video_latent_t(ov):].clone()}
        a0 = round((state["len"] - ov) * FRAME_RESCALE)
        a = aud[..., a0:a0 + round(ov * FRAME_RESCALE)]
        if a.shape[-1] > 0:
            kf["audio_latent"] = a.clone()
        kfs = list(cond[0][1].get("minimax_keyframes", [])) + [kf]
        return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": kfs})

    def pinned(cond, latent, s):
        """Pin what's already rendered at [s, s+ov) as this chunk's opening frames (and audio)."""
        guide_audio = None
        if state["audio"] is not None:
            spf = state["sr"] / FPS
            guide_audio = {"waveform": state["audio"][..., round(s * spf):round((s + ov) * spf)].clone(),
                           "sample_rate": state["sr"]}
        return MiniMaxH3AddGuide.execute(
            positive=cond, latent=latent, frame_idx=0, vae=vae,
            audio_vae=audio_vae if guide_audio is not None else None,
            image=state["sink"].get(s, s + ov), audio=guide_audio).args[0]

    if handoff == "bridge":
        return _render_bridged(locals())

    for k, s in enumerate(starts):
        comfy.model_management.throw_exception_if_processing_interrupted()
        base_cond, latent = conds[k]
        if k == 0 or handoff == "none":
            cond = base_cond
        elif handoff == "latent":
            cond = pinned_latent(base_cond, state["lat"])
        else:
            cond = pinned(base_cond, latent, s)
        _mark(f"chunk {k + 1} guide")
        jm = job_model(s, lens[k])
        samples = sample(cond, latent["samples"], sigmas, seed if handoff == "none" else seed + k, jm)
        _mark(f"chunk {k + 1} sample")
        state["lat"] = samples

        if upscaler:
            video_lat, audio_lat = samples.unbind()
            res = upscaler.execute(latent={"samples": video_lat}, model_name=up_model,
                                   mode={"mode": "megapixels", "megapixels": upscale_megapixels}, align=32,
                                   enable_temporal_chunking=True, force_unload=True, device="cuda", precision="fp16")
            res = getattr(res, "args", None) or res.result
            big = {"samples": comfy.nested_tensor.NestedTensor((res[0]["samples"], audio_lat))}
            if k == 0 or handoff == "none":
                cond = base_cond
            elif handoff == "latent":
                cond = pinned_latent(base_cond, state["lat_up"])
            else:
                cond = pinned(base_cond, big, s)
            samples = sample(cond, big["samples"], up_sigmas, seed + 100000 + (0 if handoff == "none" else k), jm)
            _mark(f"chunk {k + 1} upscale+polish")
            state["lat_up"] = samples
        state["len"] = lens[k]

        imgs = nodes.VAEDecode().decode(vae, {"samples": samples})[0].float().cpu()
        _mark(f"chunk {k + 1} vae decode")
        if state["sink"] is None:
            Hh, Ww = imgs.shape[1], imgs.shape[2]
            state["sink"] = FrameSink(F, (Hh, Ww), _stream_path(lab, F, Hh, Ww))
        seg = imgs[:min(imgs.shape[0], lens[k], F - s)]
        state["sink"].place(k, s, seg, ov)
        del imgs
        # frames before the next chunk's start can't change any more
        state["sink"].flush(starts[k + 1] if k + 1 < len(starts) else F)

        aud = vae_decode_audio(audio_vae, {"samples": samples})
        _mark(f"chunk {k + 1} audio decode")
        wf = aud["waveform"].float().cpu()
        if state["audio"] is None:
            state["sr"] = aud["sample_rate"]
            state["audio"] = torch.zeros((1, wf.shape[1], round(F / FPS * state["sr"])), dtype=torch.float32)
        out_audio = state["audio"]
        spf = state["sr"] / FPS
        a0 = round(s * spf)
        want = max(0, min(round(lens[k] * spf), out_audio.shape[-1] - a0))
        seg_a = wf[..., :want]
        if seg_a.shape[-1] < want:
            seg_a = torch.nn.functional.pad(seg_a, (0, want - seg_a.shape[-1]))
        if k == 0:
            out_audio[..., a0:a0 + want] = seg_a
        else:
            ab = min(round(ov * spf), want)
            ramp = torch.linspace(0, 1, ab + 2)[1:-1]
            out_audio[..., a0:a0 + ab] = out_audio[..., a0:a0 + ab] * (1 - ramp) + seg_a[..., :ab] * ramp
            out_audio[..., a0 + ab:a0 + want] = seg_a[..., ab:]
        print(f"[H3 Rolling] rendered chunk {k + 1}/{len(starts)} (source frames {s}-{s + seg.shape[0]}, "
              f"{state['sink'].W}x{state['sink'].H})")

    if soundtrack == "source" and audio is not None:
        final_audio = audio
    elif soundtrack == "silent":
        final_audio = None
    else:
        final_audio = {"waveform": state["audio"], "sample_rate": state["sr"]}
    frames_out = state["sink"].finish(final_audio)
    if state["sink"].path:
        plan += f" | full video: {state['sink'].path}"
    return frames_out, final_audio, plan


def _render_bridged(c):
    """Render independent chunks, then for each seam a short bridge pinned (in latent space) to the
    previous chunk's last 5 frames and the next chunk's first 5. Nothing is ever more than one
    generation from a fresh render, so nothing compounds."""
    import node_helpers
    import nodes
    import comfy.model_management
    import comfy.nested_tensor
    from comfy.ldm.minimax.model import FRAME_RESCALE
    from comfy_extras.nodes_audio import vae_decode_audio

    starts, lens, bridges, conds, bconds = c["starts"], c["lens"], c["bridges"], c["conds"], c["bconds"]
    F, seed, sample, sigmas, up_sigmas = c["F"], c["seed"], c["sample"], c["sigmas"], c["up_sigmas"]
    upscaler, up_model, vae, audio_vae, lab = c["upscaler"], c["up_model"], c["vae"], c["audio_vae"], c["lab"]
    upscale_megapixels = c["upscale_megapixels"]
    job_model = c["job_model"]
    PIN = 5
    PE = c["PE"]
    na = round(PIN * FRAME_RESCALE)

    def pins(cond, a_lat, a_len, b_lat, bl, bridge_audio_t):
        av, aa = a_lat.unbind()[:2]
        a0 = round((a_len - PIN) * FRAME_RESCALE)
        kfs = list(cond[0][1].get("minimax_keyframes", []))
        kfs.append({"resolved_frame_index": 0, "latent": av[:, :, -video_latent_t(PIN):].clone(),
                    "audio_latent": aa[..., a0:a0 + na].clone()})
        if b_lat is not None:
            bv, ba = b_lat.unbind()[:2]
            room = math.floor(bridge_audio_t - FRAME_RESCALE * (bl - PE))
            k_end = {"resolved_frame_index": bl - PE, "latent": bv[:, :, :video_latent_t(PE)].clone()}
            if room > 0:
                k_end["audio_latent"] = ba[..., :min(round(PE * FRAME_RESCALE), room)].clone()
            kfs.append(k_end)
        return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": kfs})

    def upscale(samples):
        video_lat, audio_lat = samples.unbind()
        res = upscaler.execute(latent={"samples": video_lat}, model_name=up_model,
                               mode={"mode": "megapixels", "megapixels": upscale_megapixels}, align=32,
                               enable_temporal_chunking=True, force_unload=True, device="cuda", precision="fp16")
        res = getattr(res, "args", None) or res.result
        return comfy.nested_tensor.NestedTensor((res[0]["samples"], audio_lat))

    def render_chunk(k):
        cond, latent = conds[k]
        jm = job_model(starts[k], lens[k])
        lo = sample(cond, latent["samples"], sigmas, seed, jm)
        _mark(f"chunk {k + 1} sample")
        hi = None
        if upscaler:
            hi = sample(cond, upscale(lo), up_sigmas, seed + 100000, jm)
            _mark(f"chunk {k + 1} upscale+polish")
        return {"lo": lo, "hi": hi, "len": lens[k]}

    def render_bridge(j, A, B):
        cond, latent = bconds[j]
        bl = bridges[j][1]
        at = latent["samples"].unbind()[1].shape[-1]
        jm = job_model(bridges[j][0], bl)
        lo = sample(pins(cond, A["lo"], A["len"], B["lo"] if B else None, bl, at), latent["samples"], sigmas,
                    seed + 50000 + j, jm)
        _mark(f"bridge {j + 1} sample")
        hi = None
        if upscaler:
            big = upscale(lo)
            hi = sample(pins(cond, A["hi"], A["len"], B["hi"] if B else None, bl, at), big, up_sigmas,
                        seed + 150000 + j, jm)
            _mark(f"bridge {j + 1} upscale+polish")
        return {"lo": lo, "hi": hi, "len": bl}

    def decode(r):
        samples = r["hi"] if r["hi"] is not None else r["lo"]
        imgs = nodes.VAEDecode().decode(vae, {"samples": samples})[0].float().cpu()
        aud = vae_decode_audio(audio_vae, {"samples": samples})
        return imgs, aud["waveform"].float().cpu(), aud["sample_rate"]

    st = {"sink": None, "audio": None, "sr": None}

    def place(idx, s, r, blend, fade=PIN):
        imgs, wf, sr = decode(r)
        _mark(f"{idx} decode")
        if st["sink"] is None:
            hh, ww = imgs.shape[1], imgs.shape[2]
            st["sink"] = FrameSink(F, (hh, ww), _stream_path(lab, F, hh, ww))
            st["sr"] = sr
            st["audio"] = torch.zeros((1, wf.shape[1], round(F / FPS * sr)), dtype=torch.float32)
        seg = imgs[:max(0, min(imgs.shape[0], r["len"], F - s))]
        st["sink"].place(1 if blend else 0, s, seg, fade)
        del imgs, seg  # the decoded chunk is the biggest thing in RAM; drop it before the audio work
        out = st["audio"]
        spf = st["sr"] / FPS
        a0 = round(s * spf)
        want = max(0, min(round(r["len"] * spf), out.shape[-1] - a0))
        seg_a = wf[..., :want]
        if seg_a.shape[-1] < want:
            seg_a = torch.nn.functional.pad(seg_a, (0, want - seg_a.shape[-1]))
        if not blend:
            out[..., a0:a0 + want] = seg_a
        else:
            ab = min(round(fade * spf), want)
            ramp = torch.linspace(0, 1, ab + 2)[1:-1]
            out[..., a0:a0 + ab] = out[..., a0:a0 + ab] * (1 - ramp) + seg_a[..., :ab] * ramp
            out[..., a0 + ab:a0 + want] = seg_a[..., ab:]

    prev = render_chunk(0)
    place("chunk 1", starts[0], prev, False)
    for k in range(1, len(starts)):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur = render_chunk(k)
        br = render_bridge(k - 1, prev, cur)
        place(f"bridge {k}", bridges[k - 1][0], br, True)
        place(f"chunk {k + 1}", starts[k], cur, True, PE)
        print(f"[H3 Rolling] rendered chunk {k + 1}/{len(starts)} + bridge {k}")
        st["sink"].flush(bridges[k][0] if k < len(bridges) else F)
        prev = cur
    if len(bridges) == len(starts):  # tail bridge: pinned to the last chunk at its start only
        j = len(bridges) - 1
        br = render_bridge(j, prev, None)
        place("tail bridge", bridges[j][0], br, True)
        print("[H3 Rolling] rendered tail bridge")
    st["sink"].flush(F)

    if c["soundtrack"] == "source" and c["audio"] is not None:
        final_audio = c["audio"]
    elif c["soundtrack"] == "silent":
        final_audio = None
    else:
        final_audio = {"waveform": st["audio"], "sample_rate": st["sr"]}
    frames_out = st["sink"].finish(final_audio)
    plan = c["plan"]
    if st["sink"].path:
        plan += f" | full video: {st['sink'].path}"
    return frames_out, final_audio, plan
