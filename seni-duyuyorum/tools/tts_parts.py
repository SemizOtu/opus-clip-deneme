"""Voice-over with breathing room: every sentence part is voiced on its own (several takes, Whisper picks
the clearest), then the parts of a scene are joined with a natural pause.

usage: python3 tts_parts.py <voice_dir> <whisper_dir> <out_dir> [takes] [pause_s] [only_ids]
takes=0 keeps the saved part takes and only re-joins them with the given pause.
only_ids (comma separated) re-voices just those scenes and keeps the rest of parts.json.
Writes <out_dir>/<id>.wav per scene, <out_dir>/parts/<id>_<k>.wav and <out_dir>/parts.json (part offsets).
"""
import glob, json, os, re, sys
import numpy as np
import sherpa_onnx
import soundfile as sf

voice_dir, wdir, out_dir = sys.argv[1:4]
takes = int(sys.argv[4]) if len(sys.argv) > 4 else 6
pause = float(sys.argv[5]) if len(sys.argv) > 5 else .45
only = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else None
model = glob.glob(os.path.join(voice_dir, "*.onnx"))[0]
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model, lexicon="", tokens=os.path.join(voice_dir, "tokens.txt"),
            data_dir=os.path.join(voice_dir, "espeak-ng-data")),
        provider="cpu", num_threads=4)))
rec = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder=os.path.join(wdir, "small-encoder.int8.onnx"), decoder=os.path.join(wdir, "small-decoder.int8.onnx"),
    tokens=os.path.join(wdir, "small-tokens.txt"), language="tr", task="transcribe", num_threads=4)

def norm(s):
    s = re.sub(r"\b20\b", "yirmi", s)
    s = s.replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"[^a-zçğıöşü]", "", s)

def cer(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1] / max(len(a), 1)

def trim(x, sr):
    thr = 0.004 * np.abs(x).max(); idx = np.where(np.abs(x) > thr)[0]
    return x[max(idx[0] - 220, 0): idx[-1] + int(0.12 * sr)]

def transcribe(x, sr):
    n = int(len(x) * 16000 / sr)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    y = np.concatenate([np.zeros(4000, np.float32), y, np.zeros(12000, np.float32)])
    s = rec.create_stream(); s.accept_waveform(16000, y); rec.decode_stream(s)
    return s.result.text.strip()

os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
speeds = [.95, .98, 1.0]
mpath = os.path.join(out_dir, "parts.json")
meta = json.load(open(mpath, encoding="utf-8")) if only and os.path.exists(mpath) else {}
for ln in json.load(open(os.path.join(os.path.dirname(__file__), "narration.json"), encoding="utf-8")):
    if only and ln["id"] not in only:
        continue
    says = [p.strip() for p in ln.get("say", ln["text"]).split("|")]
    texts = [p.strip() for p in ln["text"].split("|")]
    pieces, offs, t, sr0 = [], [], 0.0, 22050
    for k, (say, txt) in enumerate(zip(says, texts)):
        best = None
        if takes == 0:  # re-join the saved parts with a new pause, no new voicing
            x, sr0 = sf.read(os.path.join(out_dir, "parts", f'{ln["id"]}_{k}.wav'), dtype="float32")
            best = (0.0, len(x) / sr0, x, sr0, "(kept)")
        for j in range(takes):
            a = tts.generate(say, sid=0, speed=speeds[j % len(speeds)])
            x = trim(np.asarray(a.samples, np.float32), a.sample_rate)
            hyp = transcribe(x, a.sample_rate)
            score = cer(norm(say), norm(hyp)); dur = len(x) / a.sample_rate
            if best is None or (score, -dur) < (best[0], -best[1]):
                best = (score, dur, x, a.sample_rate, hyp)
        score, dur, x, sr0, hyp = best
        sf.write(os.path.join(out_dir, "parts", f'{ln["id"]}_{k}.wav'), x, sr0)
        if pieces:
            pieces.append(np.zeros(int(pause * sr0), np.float32)); t += pause
        offs.append({"text": txt, "start": round(t, 3), "dur": round(dur, 3)})
        pieces.append(x); t += dur
        print(f'{ln["id"]}.{k}  CER={score:.3f}  {dur:.2f}s | {hyp}', flush=True)
    sf.write(os.path.join(out_dir, ln["id"] + ".wav"), np.concatenate(pieces), sr0)
    meta[ln["id"]] = offs
json.dump(meta, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
