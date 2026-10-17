"""WhisperX transcribe + align + diarize -> word-level JSON, for H3 Rolling's per-chunk dialogue.

Runs in whatever Python has whisperx installed (H3_WHISPER_PY points the node at it); needs HF_TOKEN
for the pyannote diarization model.

Usage: python whisper_worker.py <audio.wav> <out.json> [language] [voices]
  voices = how many different people speak (0 = let diarization guess). Knowing it helps a lot.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

audio_path, out_path = sys.argv[1], sys.argv[2]
language = sys.argv[3] if len(sys.argv) > 3 else "en"
voices = int(sys.argv[4]) if len(sys.argv) > 4 else 0
device = "cuda"
token = os.environ.get("HF_TOKEN", "")
if not token:
    sys.exit("ERROR: HF_TOKEN not set (needed for pyannote diarization)")

audio = whisperx.load_audio(audio_path)
model = whisperx.load_model("large-v3", device, compute_type="float16")
result = model.transcribe(audio, batch_size=16, language=language)
language = result.get("language", language)
del model
model_a, meta = whisperx.load_align_model(language_code=language, device=device)
result = whisperx.align(result["segments"], model_a, meta, audio, device, return_char_alignments=False)
del model_a
diar = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1", token=token, device=device)
result = assign_word_speakers(diar(audio, num_speakers=voices) if voices > 0 else diar(audio), result)

segments, words = [], []
for seg in result["segments"]:
    sw = seg.get("words", [])
    # words the aligner couldn't place (numbers, symbols) get times interpolated from their neighbours,
    # so they still show up in the dialogue instead of silently vanishing
    timed = [i for i, w in enumerate(sw) if "start" in w and "end" in w]
    if not timed:
        continue
    out = []
    for i, w in enumerate(sw):
        if "start" in w and "end" in w:
            s, e = w["start"], w["end"]
        else:
            before = [j for j in timed if j < i]
            after = [j for j in timed if j > i]
            s = sw[before[-1]]["end"] if before else sw[after[0]]["start"]
            e = sw[after[0]]["start"] if after else sw[before[-1]]["end"]
            if e < s:
                e = s
        out.append({"w": w["word"].strip(), "start": round(s, 3), "end": round(e, 3),
                    "speaker": w.get("speaker", seg.get("speaker", "UNKNOWN"))})
    out = [w for w in out if w["w"]]
    words.extend(out)
    segments.append({"start": round(seg["start"], 3), "end": round(seg["end"], 3),
                     "speaker": seg.get("speaker", "UNKNOWN"), "text": seg["text"].strip()})
json.dump({"language": language, "segments": segments, "words": words},
          open(out_path, "w", encoding="utf-8"), indent=1)
print(f"{len(segments)} segments, {len(words)} words -> {out_path}")
