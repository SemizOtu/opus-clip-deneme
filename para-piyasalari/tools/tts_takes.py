"""Render several voice-over takes per line, transcribe each with Whisper and keep the clearest.

usage: python3 tts_takes.py <voice_dir> <whisper_dir> <out_dir> [takes]
Clarity = character error rate between script and transcript (case, spacing and punctuation ignored).
"""
import glob, json, os, re, sys
import numpy as np
import sherpa_onnx
import soundfile as sf

voice_dir, wdir, out_dir = sys.argv[1:4]
takes = int(sys.argv[4]) if len(sys.argv) > 4 else 6
model = glob.glob(os.path.join(voice_dir, "*.onnx"))[0]
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model, lexicon="", tokens=os.path.join(voice_dir, "tokens.txt"),
            data_dir=os.path.join(voice_dir, "espeak-ng-data")),
        provider="cpu", num_threads=4)))
rec = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder=os.path.join(wdir, "small-encoder.int8.onnx"),
    decoder=os.path.join(wdir, "small-decoder.int8.onnx"),
    tokens=os.path.join(wdir, "small-tokens.txt"), language="tr", task="transcribe", num_threads=4)

NUM = {"12": "on iki", "1": "bir"}
def norm(s):
    for k, v in NUM.items():
        s = re.sub(rf"\b{k}\b", v, s)
    s = s.replace("I", "ı").replace("İ", "i").lower()
    return re.sub(r"[^a-zçğıöşü]", "", s)

def cer(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev, d[j] = d[j], cur
    return d[-1] / max(len(a), 1)

def trim(x, sr):
    thr = 0.004 * np.abs(x).max()
    idx = np.where(np.abs(x) > thr)[0]
    return x[max(idx[0] - 220, 0): idx[-1] + int(0.12 * sr)]

def transcribe(x, sr):
    n = int(len(x) * 16000 / sr)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    y = np.concatenate([np.zeros(4000, np.float32), y, np.zeros(12000, np.float32)])
    s = rec.create_stream(); s.accept_waveform(16000, y); rec.decode_stream(s)
    return s.result.text.strip()

os.makedirs(out_dir, exist_ok=True)
speeds = [1.0, 1.04, 1.08]
report = []
for ln in json.load(open(os.path.join(os.path.dirname(__file__), "narration.json"))):
    best = None
    for k in range(takes):
        sp = speeds[k % len(speeds)]
        a = tts.generate(ln["text"], sid=0, speed=sp)
        x = trim(np.asarray(a.samples, np.float32), a.sample_rate)
        hyp = transcribe(x, a.sample_rate)
        score = cer(norm(ln["text"]), norm(hyp))
        dur = len(x) / a.sample_rate
        if best is None or (score, dur) < (best[0], best[1]):
            best = (score, dur, x, a.sample_rate, hyp, sp)
    score, dur, x, sr, hyp, sp = best
    sf.write(os.path.join(out_dir, ln["id"] + ".wav"), x, sr)
    report.append({"id": ln["id"], "cer": round(score, 3), "dur": round(dur, 3), "speed": sp, "asr": hyp})
    print(f'{ln["id"]}  CER={score:.3f}  {dur:.2f}s  x{sp}  | {hyp}', flush=True)
json.dump(report, open(os.path.join(out_dir, "asr_report.json"), "w"), ensure_ascii=False, indent=2)
