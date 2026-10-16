"""Transcribe generated voice-over with Whisper (sherpa-onnx) and report word error rate.

usage: python3 asr_check.py <whisper_dir> <vo_dir>
"""
import json, os, re, sys
import numpy as np
import sherpa_onnx
import soundfile as sf

wdir, vo = sys.argv[1], sys.argv[2]
rec = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder=os.path.join(wdir, "small-encoder.int8.onnx"),
    decoder=os.path.join(wdir, "small-decoder.int8.onnx"),
    tokens=os.path.join(wdir, "small-tokens.txt"),
    language="tr", task="transcribe", num_threads=4,
)
norm = lambda s: re.sub(r"\b12\b", "on iki", re.sub(r"\b1\b", "bir", s)) if False else _n(s)
_n = lambda s: re.sub(r"[^\wçğıöşüâî ]", " ", s.replace("I", "ı").replace("İ", "i").lower()).split()

def wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), int)
    d[:, 0] = range(len(ref) + 1); d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return d[-1, -1] / len(ref)

lines = json.load(open(os.path.join(os.path.dirname(__file__), "narration.json")))
tot = 0
for ln in lines:
    x, sr = sf.read(os.path.join(vo, ln["id"] + ".wav"), dtype="float32")
    if sr != 16000:
        n = int(len(x) * 16000 / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    x = np.concatenate([np.zeros(4000, np.float32), x, np.zeros(12000, np.float32)])
    s = rec.create_stream(); s.accept_waveform(16000, x); rec.decode_stream(s)
    hyp = s.result.text.strip().replace("12", "on iki").replace("1 ", "bir ")
    w = wer(norm(ln["text"]), norm(hyp)); tot += w
    print(f'{ln["id"]} WER={w:.2f} | {hyp}')
print(f"mean WER={tot/len(lines):.3f}")
