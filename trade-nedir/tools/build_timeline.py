"""Build the shared timeline (scene times, word timings, cue times) from the voice-over takes.

usage: python3 build_timeline.py <vo_dir> <out_json>
Both the canvas renderer and the sound design read this file, so picture and sound share one clock.
"""
import json, os, re, sys
import numpy as np
import soundfile as sf

vo_dir, out = sys.argv[1], sys.argv[2]
lines = json.load(open(os.path.join(os.path.dirname(__file__), "narration.json")))
META = {
    "s0": ("tanim", "TANIM", 0.60, 0.30),
    "s1": ("defter", "EMİR DEFTERİ", 0.18, 0.36),
    "s2": ("mum", "MUM", 0.32, 0.50),
    "s3": ("trend", "TREND", 0.22, 0.90),
    "s4": ("plan", "İŞLEM PLANI", 0.22, 1.10),
    "s5": ("risk", "RİSK", 0.26, 0.55),
    "s6": ("sistem", "SİSTEM", 0.30, 1.35),
}
VOWELS = set("aeıioöuüAEIİOÖUÜ")

def syllables(w):
    return max(1, sum(c in VOWELS for c in w))

def word_times(x, sr, words):
    """Split the take at its longest pauses (one per punctuation break), then share each phrase by syllables."""
    hop = int(sr * 0.01)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))[::hop]
    quiet = rms < 0.07 * rms.max()
    gaps, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if i > 5 and j < len(quiet) - 5 and j - i >= 7:
                gaps.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    breaks = [k for k, w in enumerate(words[:-1]) if re.search(r"[,;:.]$", w)]
    total = len(x) / sr
    # match each punctuation break to the pause nearest its syllable-proportional position
    wts = [syllables(w) + 0.6 for w in words]
    chosen, after = [], -1.0
    for k in breaks:
        expect = total * sum(wts[: k + 1]) / sum(wts)
        cand = [g for g in gaps if g[0] > after]
        if not cand:
            chosen = []
            break
        g = min(cand, key=lambda g: abs((g[0] + g[1]) / 2 - expect) - 1.5 * (g[1] - g[0]))
        chosen.append(g); after = g[1]
    gaps = chosen
    bounds, start, phrases = [], 0.0, []
    cut = 0
    for k in range(len(words)):
        if k in breaks and gaps:
            phrases.append((cut, k + 1, start, gaps[len(phrases)][0]))
            start, cut = gaps[len(phrases) - 1][1], k + 1
    phrases.append((cut, len(words), start, total))
    res = []
    for a, b, t0, t1 in phrases:
        wts = [syllables(w) + 0.6 for w in words[a:b]]
        acc = t0
        for w, wt in zip(words[a:b], wts):
            d = (t1 - t0) * wt / sum(wts)
            res.append({"w": w, "s": round(acc, 3), "e": round(acc + d, 3)})
            acc += d
    return res

scenes, t = [], 0.0
for ln in lines:
    key, label, lead, tail = META[ln["id"]]
    x, sr = sf.read(os.path.join(vo_dir, ln["id"] + ".wav"), dtype="float32")
    dur = len(x) / sr
    words = ln["text"].split()
    wt = word_times(x, sr, words)
    vs = t + lead
    for w in wt:
        w["s"] = round(w["s"] + vs, 3); w["e"] = round(w["e"] + vs, 3)
    end = vs + dur + tail
    scenes.append({"id": ln["id"], "key": key, "label": label, "start": round(t, 3), "end": round(end, 3),
                   "vo": {"start": round(vs, 3), "dur": round(dur, 3), "text": ln["text"], "words": wt}})
    t = end

S = {s["id"]: s for s in scenes}
r = lambda v: round(v, 3)

def W(sid, prefix, k="s"):
    return next(w for w in S[sid]["vo"]["words"] if w["w"].lower().startswith(prefix))[k]

def win(sid, before, after):
    return [r(S[sid]["end"] - before), r(S[sid]["end"] + after)]

last = lambda sid: S[sid]["vo"]["words"][-1]["e"]
cues = {
    "tr": {"t01": win("s0", .32, .32), "t12": win("s1", .38, .42), "t23": win("s2", .52, .38),
           "t34": win("s3", .32, .5), "t45": win("s4", .32, .32), "t56": win("s5", .38, .42)},
    "s0": {"title": .3, "line": [.15, 1.75], "gap": r(W("s0", "fiyat")), "buy": r(W("s0", "alıp")), "sell": r(W("s0", "satmak"))},
    "s1": {"match": r(W("s1", "buluşur")), "arz": r(W("s1", "arz")), "talep": r(W("s1", "talep"))},
    "s2": {"form": [r(S["s1"]["end"] + .4), r(W("s2", "açılış") - .05)], "open": r(W("s2", "açılış")), "high": r(W("s2", "yüksek")),
           "low": r(W("s2", "düşük")), "close": r(W("s2", "kapanış")), "legend": r(last("s2") + .05)},
    "s3": {"build": [r(S["s3"]["start"] + .12), r(W("s3", "trend") + .05)], "trend": r(W("s3", "trend")), "lows": r(W("s3", "yükselen")),
           "sup": r(W("s3", "destek")), "res": r(W("s3", "direnç")), "brk": r(last("s3") + .08)},
    "s4": {"entry": r(W("s4", "giriş")), "stop": r(W("s4", "zarar")), "tp": r(W("s4", "kâr")), "run": [r(last("s4") - .05), r(S["s4"]["end"] - .5)]},
    "s5": {"rule": r(W("s5", "altın")), "cell": r(W("s5", "yüzde")), "sim": [r(W("s5", "fazla")), r(S["s5"]["end"] - .5)]},
    "s6": {"ring": [r(S["s6"]["start"] + .1), r(S["s6"]["start"] + 1.4)], "title": r(W("s6", "trade")), "sub": r(W("s6", "kurallar")),
           "disc": r(last("s6") + .1), "fade": r(S["s6"]["end"] - .45)},
}
tl = {"duration": r(scenes[-1]["end"]), "fps": 60, "scenes": scenes, "cues": cues}
json.dump(tl, open(out, "w"), ensure_ascii=False, indent=1)
for s in scenes:
    print(f'{s["id"]} {s["label"]:13s} {s["start"]:6.2f} → {s["end"]:6.2f}  ' + " ".join(f'{w["w"]}@{w["s"]:.2f}' for w in s["vo"]["words"]))
print("duration", tl["duration"])
print(json.dumps(cues))
