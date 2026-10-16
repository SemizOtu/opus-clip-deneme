"""Generate Turkish voice-over lines with a Piper voice via sherpa-onnx.

usage: python3 tts.py <voice_dir> <out_dir> [speed]
"""
import glob, json, os, sys
import numpy as np
import sherpa_onnx
import soundfile as sf

voice_dir, out_dir = sys.argv[1], sys.argv[2]
speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
model = glob.glob(os.path.join(voice_dir, "*.onnx"))[0]
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=model,
            lexicon="",
            data_dir=os.path.join(voice_dir, "espeak-ng-data"),
            tokens=os.path.join(voice_dir, "tokens.txt"),
        ),
        provider="cpu",
        num_threads=4,
    ),
)
tts = sherpa_onnx.OfflineTts(cfg)
lines = json.load(open(os.path.join(os.path.dirname(__file__), "narration.json")))
os.makedirs(out_dir, exist_ok=True)
for ln in lines:
    a = tts.generate(ln["text"], sid=0, speed=speed)
    x = np.asarray(a.samples, dtype=np.float32)
    # trim leading/trailing silence
    thr = 0.004 * np.abs(x).max()
    idx = np.where(np.abs(x) > thr)[0]
    x = x[max(idx[0] - 220, 0): idx[-1] + int(0.12 * a.sample_rate)]
    sf.write(os.path.join(out_dir, ln["id"] + ".wav"), x, a.sample_rate)
    print(f'{ln["id"]}\t{len(x)/a.sample_rate:.2f}s\t{a.sample_rate}Hz')
