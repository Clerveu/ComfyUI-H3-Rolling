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
            tail = (b0, align(max(F - b0, pin + 17)), False)
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


# ---------------------------------------------------------------- per-chunk dialogue

LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese",
             "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "ru": "Russian", "nl": "Dutch", "pl": "Polish"}


def _whisper_python():
    """The Python that has whisperx: H3_WHISPER_PY, else ComfyUI's own if whisperx is importable there."""
    import importlib.util
    import os
    import sys
    py = os.environ.get("H3_WHISPER_PY")
    if py:
        if not os.path.exists(py):
            raise ValueError(f"H3_WHISPER_PY points at {py}, which doesn't exist")
        return py
    if importlib.util.find_spec("whisperx") is not None:
        return sys.executable
    raise ValueError("dialogue needs WhisperX: install whisperx into ComfyUI's Python, or set the H3_WHISPER_PY "
                     "environment variable to a Python that has it (and HF_TOKEN for diarization), then restart ComfyUI")


def transcribe(audio, language="en"):
    """Diarized, word-level transcript of the selection's soundtrack. Times are seconds from the start of
    the selection, so they line up with the frames. Cached by the audio itself in transcript_cache/."""
    import hashlib
    import json
    import os
    import subprocess
    import tempfile
    import wave

    wf = audio["waveform"][0].float().mean(0)
    sr = int(audio["sample_rate"])
    try:
        import torchaudio
        mono = torchaudio.functional.resample(wf, sr, 16000) if sr != 16000 else wf
    except ImportError:
        mono = torch.nn.functional.interpolate(wf.view(1, 1, -1), size=int(wf.shape[-1] * 16000 / sr),
                                               mode="linear", align_corners=False).view(-1)
    n = mono.shape[-1]
    pcm = (mono.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    key = hashlib.sha1(pcm + language.encode()).hexdigest()[:20]
    d = os.path.join(os.path.dirname(__file__), "transcript_cache")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, key + ".json")
    if os.path.exists(path):
        print(f"[H3 Rolling] dialogue: transcript reused from cache ({key})")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    py = _whisper_python()
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    tmp = tempfile.mkdtemp(prefix="h3_whisper_")
    wav, out = os.path.join(tmp, "audio.wav"), os.path.join(tmp, "out.json")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    print(f"[H3 Rolling] dialogue: transcribing {n / 16000:.1f}s with WhisperX ({py})")
    r = subprocess.run([py, os.path.join(os.path.dirname(__file__), "whisper_worker.py"), wav, out, language],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=0x08000000 if os.name == "nt" else 0)
    if r.returncode != 0 or not os.path.exists(out):
        tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(f"WhisperX failed (exit {r.returncode}):\n{tail}")
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    with open(path + ".tmp", "w", encoding="utf-8") as f:  # temp dir may be on another drive: write, don't move
        json.dump(data, f)
    os.replace(path + ".tmp", path)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[H3 Rolling] dialogue: {len(data['words'])} words, "
          f"{len({w['speaker'] for w in data['words']})} speaker label(s)")
    return data


def _secs(t):
    """'12.5', '0:12.50' or '1:02:03' -> seconds."""
    s = 0.0
    for part in t.strip().split(":"):
        s = s * 60 + float(part)
    return s


def parse_speakers(text):
    """`SPEAKER_00 = <Subject 1>` per line. Add (voiceover) for an off-screen voice, or map a label to
    `ignore` to leave it out (a TV in the background, crowd noise). A time range fixes diarization by
    hand: `0:08.30-0:08.70 = SPEAKER_00` (or straight to a description) re-labels the words in it.
    Returns (label -> description, [(t0, t1, label or description)])."""
    import re
    names, fixes = {}, []
    rng = re.compile(r"^\s*([\d:.]+)\s*-\s*([\d:.]+)\s*=\s*(.+)$")
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = rng.match(line)
        if m:
            try:
                fixes.append((_secs(m.group(1)), _secs(m.group(2)), m.group(3).strip()))
            except ValueError:
                pass
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            names[k.strip().upper()] = v.strip()
    return names, fixes


def clean_speakers(words, fixes, names):
    """Diarization flickers between voices inside one sentence ("not / drawing / much"): give every
    sentence the voice that speaks most of it, then apply the hand-written time-range fixes. A fix that
    names a description instead of a label becomes its own speaker. Returns new word dicts + names."""
    import re
    out = [dict(w) for w in words]
    names = dict(names)
    sent = []
    for i, w in enumerate(out):
        sent.append(i)
        end = w["w"].rstrip("\"')]")[-1:] in ".?!"
        gap = i + 1 < len(out) and out[i + 1]["start"] - w["end"] > 1.0
        if end or gap or i + 1 == len(out):
            dur = {}
            for j in sent:
                dur[out[j]["speaker"]] = dur.get(out[j]["speaker"], 0) + max(0.05, out[j]["end"] - out[j]["start"])
            best = max(dur, key=dur.get)
            for j in sent:
                out[j]["speaker"] = best
            sent = []
    label = re.compile(r"^SPEAKER_\d+$", re.I)
    for t0, t1, who in fixes:
        if label.match(who):
            tag = who.upper()
        else:
            tag = "FIX:" + who
            names.setdefault(tag.upper(), who)
        for w in out:
            if t0 <= (w["start"] + w["end"]) / 2 < t1:
                w["speaker"] = tag
    return out, names


def _who(label, speakers):
    """(phrase, voiceover?, identity) for a diarization label, or None to leave its words out. Labels
    mapped to the same phrase are one speaker (diarization often splits one voice in two); unmapped
    labels stay apart from each other."""
    v = speakers.get(label.upper())
    if v is None:
        return "a person", False, label
    if v.lower() in ("ignore", "skip", "none", "-"):
        return None
    vo = False
    for tag in ("(voiceover)", "(off-screen)", "(offscreen)"):
        if v.lower().endswith(tag):
            v, vo = v[:-len(tag)].strip(), True
    return v, vo, v


def _mid_sentence(a, b, join):
    """Word b carries on word a's sentence: same voice, a doesn't end one, and b follows soon."""
    return (a["speaker"] == b["speaker"] and a["w"].rstrip("\"')]")[-1:] not in ".?!"
            and b["start"] - a["end"] < join)


def dialogue_lines(words, t0, t1, speakers, join=1.5):
    """The lines spoken inside [t0, t1): a word belongs to the piece its midpoint falls in. A line whose
    sentence began before t0 'continues'; one still mid-sentence at t1 ends in <cutoff>. Returns dicts
    (who, vo, key, text, cont, cut, start, end) in order."""
    keep = [i for i, w in enumerate(words) if t0 <= (w["start"] + w["end"]) / 2 < t1]
    lines = []
    for i in keep:
        w = words[i]
        who = _who(w["speaker"], speakers)
        if who is None:
            continue
        cur = lines[-1] if lines else None
        if cur and cur["who"][2] == who[2]:
            cur["words"].append(w["w"])
            cur["end"], cur["last"] = w["end"], i
            continue
        cont = i > 0 and (words[i - 1]["start"] + words[i - 1]["end"]) / 2 < t0 and _mid_sentence(words[i - 1], w, join)
        lines.append({"who": who, "words": [w["w"]], "start": w["start"], "end": w["end"], "last": i, "cont": cont})
    for ln in lines:
        j = ln["last"]
        nxt = words[j + 1] if j + 1 < len(words) else None
        if nxt is None:  # the selection ends here: a sentence still open was cut off by it
            ln["cut"] = words[j]["w"].rstrip("\"')]")[-1:] not in ".?!"
        else:
            ln["cut"] = (nxt["start"] + nxt["end"]) / 2 >= t1 and _mid_sentence(words[j], nxt, join)
        ln["text"] = " ".join(ln.pop("words"))
        ln["who"], ln["vo"], ln["key"] = ln["who"]
        del ln["last"]
    return lines


def dialogue_block(lines, language="English"):
    """Prompt sentences for one piece, speakers numbered (S1), (S2)... by first vocal event in it."""
    ids, out = {}, []
    for ln in lines:
        sid = ids.setdefault(ln["key"], len(ids) + 1)
        text = ln["text"].strip()
        if ln["cut"]:
            text = text.rstrip(",;:-") + " <cutoff>"
        elif text and text[-1] not in ".?!":
            text = text.rstrip(",;:-") + "."
        verb = "continues" if ln["cont"] else "says"
        if ln["vo"]:
            out.append(f"{ln['who']} (S{sid}) {verb} in an off-screen voiceover: <d>[{language}] {text}</d> "
                       f"while their lips remain completely closed.")
        else:
            out.append(f"{ln['who']} (S{sid}) {verb}, <d>[{language}] {text}</d>")
    return " ".join(out)


def prompt_with_dialogue(prompt, block):
    """Put a piece's dialogue where {dialogue} is in the prompt, else at the end of detailed_description
    (just before overall_soundscape), else at the very end."""
    import re
    if "{dialogue}" in prompt:
        out = prompt.replace("{dialogue}", block)
        return re.sub(r"[ \t]{2,}", " ", out).replace(" \n", "\n")
    if not block:
        return prompt
    m = re.search(r"\n\s*overall_soundscape:", prompt)
    if m:
        head = prompt[:m.start()].rstrip()
        return f"{head} {block}{prompt[m.start():]}"
    return prompt.rstrip() + " " + block


def speaker_summary(words, speakers):
    """One line per diarization label: how much it says, its first words, and what it's mapped to."""
    by = {}
    for w in words:
        s = by.setdefault(w["speaker"], {"n": 0, "first": w["start"], "sample": []})
        s["n"] += 1
        if len(s["sample"]) < 12:
            s["sample"].append(w["w"])
    rows = []
    for lab, s in sorted(by.items(), key=lambda kv: kv[1]["first"]):
        m = speakers.get(lab.upper())
        tgt = f"-> {m}" if m else "-> NOT MAPPED (written as 'a person')"
        rows.append(f"  {lab}: {s['n']} words, first at {_ts(s['first'])}: \"{' '.join(s['sample'])}...\"  {tgt}")
    return "\n".join(rows)


def _ts(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


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
    # Handoff: each window after the first starts one frame early, on the last frame of the window
    # before, with that frame's per-person masks as SAM3's initial masks, so it keeps following the
    # same people instead of choosing again from everything the text matches (crowds).
    # A track that dies (a cut, someone walking out) keeps its slot, so SAM3 can't pick anyone up
    # again inside that window: when a window ends with half a second or more of nobody, that stretch
    # is tracked again fresh from the frame the person was lost, and fills in wherever it finds someone.
    # Windows are cached by their pixels + settings + the incoming handoff.
    win = int(lock.get("window", 120))
    carry_on = bool(lock.get("carry", True))
    settings = f"{sam_ckpt}|{track_text}|{threshold}|{max_objects}|refill"
    refill_min = int(lock.get("refill_frames", 12))

    def track(clip, init=None):
        if not sam:
            sam_model, sam_clip = nodes.CheckpointLoaderSimple().load_checkpoint(sam_ckpt)[:2]
            sam.extend([sam_model, nodes.CLIPTextEncode().encode(sam_clip, track_text)[0]])
        t = first(SAM3_VideoTrack.execute(images=clip, model=sam[0], conditioning=sam[1], initial_mask=init,
                                          detection_threshold=threshold, max_objects=max_objects,
                                          detect_interval=1))
        m = first(SAM3_TrackToMask.execute(track_data=t, object_indices="")).cpu().float()
        if m.shape[0] != clip.shape[0]:
            raise RuntimeError(f"SAM3 returned {m.shape[0]} masks for {clip.shape[0]} frames")
        return m, _last_people(t)

    masks, hits, carried, refilled = [], 0, 0, 0
    carry = None
    for a in range(0, frames.shape[0], win):
        clip = frames[a:a + win]
        handoff = carry_on and a > 0 and carry is not None and carry.shape[0] > 0
        if handoff:
            import hashlib
            ckey = hashlib.sha1(carry.numpy().tobytes()).hexdigest()[:16]
            key = _mask_key(frames[a - 1:a + win], f"{settings}|carry {ckey}")
        else:
            key = _mask_key(clip, settings)
        m = _mask_cache_get(key, clip.shape[:3])
        nxt = _carry_cache_get(key) if m is not None else None
        if m is not None and (nxt is not None or not carry_on):
            hits += 1
        else:
            if handoff:
                m, nxt = track(frames[a - 1:a + win], carry)
                m = m[1:]
                carried += 1
            else:
                m, nxt = track(clip)
            empty = m.flatten(1).amax(1) <= 0.5
            e = m.shape[0]
            while e > 0 and bool(empty[e - 1]):
                e -= 1
            if m.shape[0] - e >= refill_min and e > 0:
                fm, fnxt = track(clip[e:])
                m[e:] = torch.where(empty[e:].view(-1, 1, 1), fm, m[e:])
                nxt = fnxt
                refilled += 1
            _mask_cache_put(key, m)
            _carry_cache_put(key, nxt)
        carry = nxt
        masks.append(m)
    n_win = len(masks)
    print(f"[H3 Rolling] structure lock: {hits} of {n_win} tracking windows reused from cache, {n_win - hits} tracked"
          + ((f" ({carried} handed off from the window before" if carried else " (")
             + (f", {refilled} re-tracked after losing the person)" if refilled else ")")
             if carried or refilled else ""))
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
    hole = float(lock.get("hole", 0.0))  # share of the source's own edges left inside the silhouette (0 = blank)
    ctrl = edges * (1.0 - m * (1.0 - hole)).unsqueeze(-1)
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


def _last_people(track_data):
    """Per-person masks on the last frame of a SAM3 track, [K, h, w] uint8 (people present only)."""
    from comfy.ldm.sam3.tracker import unpack_masks
    packed = track_data.get("packed_masks")
    if packed is None:
        return torch.zeros((0, 1, 1), dtype=torch.uint8)
    last = unpack_masks(packed[-1]).cpu()
    keep = last.flatten(1).any(1)
    return last[keep].to(torch.uint8)


def _carry_cache_get(key):
    import os
    try:
        return torch.from_numpy(np.load(os.path.join(_mask_cache_dir(), key + ".carry.npz"))["c"])
    except (OSError, KeyError, ValueError):
        return None


def _carry_cache_put(key, c):
    import os
    d = _mask_cache_dir()
    tmp = os.path.join(d, key + ".carry.tmp.npz")
    np.savez_compressed(tmp, c=c.numpy())
    os.replace(tmp, os.path.join(d, key + ".carry.npz"))


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
    # how well each window restart lines up with the frame before it (1.0 = same silhouette)
    seam_iou = []
    for i in range(win, n, win):
        if cov[i - 1] > 0 and cov[i] > 0:
            seam_iou.append((float((b[i - 1] & b[i]).sum()) / max(1.0, float((b[i - 1] | b[i]).sum())), i))
    if seam_iou:
        vals = sorted(v for v, _ in seam_iou)
        worst = min(seam_iou)
        print(f"[H3 Rolling] structure lock: {len(seam_iou)} window seams, overlap median {vals[len(vals) // 2]:.3f}, "
              f"worst {worst[0]:.3f} at {int(worst[1] / FPS // 60)}:{worst[1] / FPS % 60:05.2f}; "
              f"all: {' '.join(f'{v:.2f}' for v, _ in seam_iou)}")
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
    PS = align(int(lab.get("bridge_pin", 5)))  # frames of chunk A held at a bridge's start
    if handoff == "bridge":
        if BL < PS + PE + 12:
            BL = align(PS + PE + 12)
        starts, lens, bridges = plan_bridged(F, L, BL, PS, PE)
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
        plan += (f" | independent chunks + {len(bridges)} bridge(s) of {BL} frames, pins {PS}+{PE},"
                 f" at {[b for b, _, _ in bridges]}, pins {lab.get('pin_mode', opts.get('pin_mode', 'keyframe'))}")
    if upscaler:
        plan += f" | upscale to {upscale_megapixels} MP with {up_model}, {upscale_steps} steps @ {upscale_denoise}"
    if refmod_blocks:
        plan += f" | {len(refmod_blocks)} RefMod block(s) on every chunk"

    # per-piece dialogue: each chunk and bridge gets only the words spoken inside its own frames
    dialogue = lab.get("dialogue", opts.get("dialogue", "off"))
    chunk_prompts, bridge_prompts = [prompt] * len(starts), [prompt] * len(bridges)
    if dialogue != "off":
        if audio is None:
            raise ValueError("dialogue needs the source audio (connect H3 Rolling Source's audio output)")
        lang_code = lab.get("dialogue_language", "en")
        tr = transcribe(audio, lang_code)
        lang = LANGUAGES.get(tr.get("language", lang_code), "English")
        speakers, fixes = parse_speakers(opts.get("speakers", ""))
        words, speakers = clean_speakers(tr["words"], fixes, speakers)
        rows = []

        def piece(name, f0, n):
            f1 = min(f0 + n, F)
            block = dialogue_block(dialogue_lines(words, f0 / FPS, f1 / FPS, speakers), lang)
            rows.append(f"{name} {_ts(f0 / FPS)}-{_ts(f1 / FPS)}: {block or '(no dialogue)'}")
            return prompt_with_dialogue(prompt, block)

        chunk_prompts = [piece(f"chunk {k + 1}", s, lens[k]) for k, s in enumerate(starts)]
        bridge_prompts = [piece(f"bridge {j + 1}", b0, bl) for j, (b0, bl, _) in enumerate(bridges)]
        order = sorted([(s, r) for s, r in zip(starts, rows[:len(starts)])]
                       + [(b0, r) for (b0, _, _), r in zip(bridges, rows[len(starts):])])
        report = ("dialogue: speaker labels (map them in the speakers box, e.g. SPEAKER_00 = <Subject 1>):\n"
                  + speaker_summary(words, speakers) + "\n\ndialogue per piece (times from the start of the selection):\n"
                  + "\n".join(r for _, r in order))
        print(f"[H3 Rolling] {report}")
        if dialogue == "preview only":
            import os
            import time
            import folder_paths
            d = os.path.join(folder_paths.get_output_directory(), "h3_rolling")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, time.strftime("dialogue_%Y%m%d_%H%M%S.txt"))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(plan + "\n\n" + report + "\n\nfirst chunk's full prompt:\n" + chunk_prompts[0] + "\n")
            print(f"[H3 Rolling] dialogue preview only: nothing rendered, written to {path}")
            return frames, audio, plan + "\n" + report
        unmapped = sorted({w["speaker"].upper() for w in words} - set(speakers))
        plan += f" | per-piece dialogue ({len(words)} words)" + (f", unmapped: {', '.join(unmapped)}" if unmapped else "")

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
    def encode(s, Lk, piece_prompt):
        ref_audio = None
        if audio is not None and source_audio_as_reference:
            ref_audio = _audio_slice(audio, s, s + Lk)
        ref_vid = frames[s:s + Lk]
        if ref_vid.shape[0] == 0:  # a bridge plan can put the last chunk's start right at the end
            ref_vid = frames[-1:]
        if ref_vid.shape[0] < Lk:  # short last chunk rounded up to 17k+5: hold the last frame
            ref_vid = torch.cat([ref_vid, ref_vid[-1:].expand(Lk - ref_vid.shape[0], -1, -1, -1)])
        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, prompt=piece_prompt, width=w, height=h, length=Lk, ref_image_size=ref_image_size,
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
        conds.append(encode(s, lens[k], chunk_prompts[k]))
        print(f"[H3 Rolling] encoded chunk {k + 1}/{len(starts)}")
        _mark(f"encode chunk {k + 1}")
    bconds = []
    for j, (b0, bl, _) in enumerate(bridges):
        bconds.append(encode(b0, bl, bridge_prompts[j]))
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

    def sample(cond, latent_samples, sig, noise_seed, mdl=None, mask=None):
        guider = Guider_Basic(mdl if mdl is not None else model)
        guider.set_conds(cond)
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed)
        callback = latent_preview.prepare_callback(model, sig.shape[-1] - 1, {})
        result = guider.sample(noise, latent_samples, sampler, sig, denoise_mask=mask, callback=callback,
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
    PIN = c["PS"]
    PE = c["PE"]
    na = round(PIN * FRAME_RESCALE)
    pin_mode = lab.get("pin_mode", c["opts"].get("pin_mode", "keyframe"))
    fade_a = int(lab.get("bridge_fade", PIN))  # frames cross-faded where a bridge meets chunk A (0 = hard join)
    fade_b = int(lab.get("bridge_fade", PE))  # ... and where chunk B takes over from the bridge
    hold_audio = bool(lab.get("hold_audio", True))
    same_seed = bool(lab.get("bridge_same_seed", False))  # bridges sample with the chunks' seed
    print(f"[H3 Rolling] bridge pins: {pin_mode}, fades {fade_a}/{fade_b}"
          + ("" if pin_mode == "keyframe" or hold_audio else ", audio not held"))

    def held(samples, a_lat, a_len, b_lat, bl):
        """mask pin_mode: write A's last PIN frames and B's first PE frames straight into the bridge's own
        starting latent and mask them 0, so the sampler keeps them and they decode as the originals
        (roycho87's Clown Context / Clown Anchoring). Positions only line up on H3's 17-frame blocks."""
        if a_len % 17 != 5 or bl % 17 != 5 or (bl - PE) % 17:
            raise ValueError(f"masked pins need 17k+5 lengths (chunk {a_len}, bridge {bl}, end pin {PE})")
        NT = comfy.nested_tensor.NestedTensor
        v, a = [t.clone() for t in samples.unbind()[:2]]
        av, aa = a_lat.unbind()[:2]
        mv, ma = torch.ones_like(v), torch.ones_like(a)
        n0 = video_latent_t(PIN)
        v[:, :, :n0] = av[:, :, -n0:].to(v)
        mv[:, :, :n0] = 0
        note = f"latents 0..{n0 - 1} <- A's last {PIN} frames"
        if hold_audio:
            a0 = round((a_len - PIN) * FRAME_RESCALE)
            src = aa[..., a0:a0 + na]
            k = min(src.shape[-1], a.shape[-1])
            a[..., :k] = src[..., :k].to(a)
            ma[..., :k] = 0
            note += f" (+{k} audio)"
        if b_lat is not None:
            bv, ba = b_lat.unbind()[:2]
            n1 = video_latent_t(PE)
            t = v.shape[2]
            v[:, :, t - n1:] = bv[:, :, :n1].to(v)
            mv[:, :, t - n1:] = 0
            note += f", latents {t - n1}..{t - 1} <- B's first {PE} frames"
            if hold_audio:
                b0 = round(FRAME_RESCALE * (bl - PE))
                k = max(0, min(round(PE * FRAME_RESCALE), a.shape[-1] - b0, ba.shape[-1]))
                if k:
                    a[..., b0:b0 + k] = ba[..., :k].to(a)
                    ma[..., b0:b0 + k] = 0
                    note += f" (+{k} audio)"
        print(f"[H3 Rolling] held {note}")
        return NT((v, a)), NT((mv, ma))

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
        s_lo, s_hi = (seed, seed + 100000) if same_seed else (seed + 50000 + j, seed + 150000 + j)
        if pin_mode == "mask":
            start, mask = held(latent["samples"], A["lo"], A["len"], B["lo"] if B else None, bl)
            lo = sample(cond, start, sigmas, s_lo, jm, mask)
        else:
            lo = sample(pins(cond, A["lo"], A["len"], B["lo"] if B else None, bl, at), latent["samples"], sigmas,
                        s_lo, jm)
        _mark(f"bridge {j + 1} sample")
        hi = None
        if upscaler:
            big = upscale(lo)
            if pin_mode == "mask":
                big, mask = held(big, A["hi"], A["len"], B["hi"] if B else None, bl)
                hi = sample(cond, big, up_sigmas, s_hi, jm, mask)
            else:
                hi = sample(pins(cond, A["hi"], A["len"], B["hi"] if B else None, bl, at), big, up_sigmas,
                            s_hi, jm)
            _mark(f"bridge {j + 1} upscale+polish")
        return {"lo": lo, "hi": hi, "len": bl}

    def decode(r):
        samples = r["hi"] if r["hi"] is not None else r["lo"]
        imgs = nodes.VAEDecode().decode(vae, {"samples": samples})[0].float().cpu()
        aud = vae_decode_audio(audio_vae, {"samples": samples})
        return imgs, aud["waveform"].float().cpu(), aud["sample_rate"]

    st = {"sink": None, "audio": None, "sr": None}

    def place(idx, s, r, blend, fade=PIN, check=0, dec=None):
        imgs, wf, sr = dec if dec is not None else decode(r)
        dec = None
        _mark(f"{idx} decode")
        if st["sink"] is None:
            hh, ww = imgs.shape[1], imgs.shape[2]
            st["sink"] = FrameSink(F, (hh, ww), _stream_path(lab, F, hh, ww))
            st["sr"] = sr
            st["audio"] = torch.zeros((1, wf.shape[1], round(F / FPS * sr)), dtype=torch.float32)
        seg = imgs[:max(0, min(imgs.shape[0], r["len"], F - s))]
        if blend and check:  # how far the incoming piece's shared frames are from what's already placed
            m = min(check, seg.shape[0])
            d = (st["sink"].get(s, s + m) - seg[:m].float()).abs().mean(dim=(1, 2, 3)) * 255
            print(f"[H3 Rolling] join {idx} @ {s}: mean abs diff over the {m} shared frames "
                  f"{' '.join(f'{x:.2f}' for x in d.tolist())} (0-255)")
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

    def colorfix(j, br_dec, b_imgs):
        """Match the bridge's free middle to the real frames on either side: a per-channel quantile curve where the
        free part starts (vs chunk A's last frames) and where it ends (vs chunk B's first frames), eased across."""
        imgs = br_dec[0]
        b0, bl = bridges[j][0], bridges[j][1]
        f0, f1 = PIN, (bl - PE) if b_imgs is not None else imgs.shape[0]
        if f1 - f0 < 2:
            return br_dec
        k = 3
        qs = torch.linspace(0, 1, 65)

        def curve(src, ref):
            src = src[:, ::4, ::4].reshape(-1, 3)
            ref = ref[:, ::4, ::4].reshape(-1, 3).float()
            return [(torch.quantile(src[:, ch], qs).numpy(), torch.quantile(ref[:, ch], qs).numpy()) for ch in range(3)]

        ca = curve(imgs[f0:f0 + k], st["sink"].get(b0 + f0 - k, b0 + f0))
        cb = curve(imgs[f1 - k:f1], b_imgs[:k]) if b_imgs is not None else ca
        out = imgs.clone()
        n = f1 - f0
        for i in range(f0, f1):
            w = (i - f0 + 0.5) / n
            x = imgs[i].numpy()
            for ch in range(3):
                ya = np.interp(x[..., ch], ca[ch][0], ca[ch][1])
                yb = np.interp(x[..., ch], cb[ch][0], cb[ch][1])
                out[i, ..., ch] = torch.from_numpy((1 - w) * ya + w * yb)
        gain = (out[f0:f1] - imgs[f0:f1]).mean().item() * 255
        print(f"[H3 Rolling] bridge {j + 1} color matched to its chunks: free frames {f0}-{f1 - 1} moved {gain:+.2f} levels on average")
        return (out.clamp(0, 1),) + tuple(br_dec[1:])

    color = bool(lab.get("bridge_color", False))
    prev = render_chunk(0)
    place("chunk 1", starts[0], prev, False)
    for k in range(1, len(starts)):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur = render_chunk(k)
        br = render_bridge(k - 1, prev, cur)
        if color:
            cur_dec = decode(cur)
            br_dec = colorfix(k - 1, decode(br), cur_dec[0])
            place(f"bridge {k}", bridges[k - 1][0], br, True, fade_a, PIN, br_dec)
            del br_dec
            place(f"chunk {k + 1}", starts[k], cur, True, fade_b, PE, cur_dec)
            del cur_dec
        else:
            place(f"bridge {k}", bridges[k - 1][0], br, True, fade_a, PIN)
            place(f"chunk {k + 1}", starts[k], cur, True, fade_b, PE)
        print(f"[H3 Rolling] rendered chunk {k + 1}/{len(starts)} + bridge {k}")
        st["sink"].flush(bridges[k][0] if k < len(bridges) else F)
        prev = cur
    if len(bridges) == len(starts):  # tail bridge: pinned to the last chunk at its start only
        j = len(bridges) - 1
        br = render_bridge(j, prev, None)
        place("tail bridge", bridges[j][0], br, True, fade_a, 0, colorfix(j, decode(br), None) if color else None)
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
