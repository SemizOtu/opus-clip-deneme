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
    "s0": ("tanim", "TANIM", 0.55, 0.25),
    "s1": ("vade", "VADE", 0.10, 0.22),
    "s2": ("aktorler", "AKTÖRLER", 0.12, 0.25),
    "s3": ("araclar", "ARAÇLAR", 0.12, 0.28),
    "s4": ("faiz", "FAİZ", 0.10, 0.38),
    "s5": ("ozet", "ÖZET", 0.12, 0.85),
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

def word(sid, prefix):
    sc = next(s for s in scenes if s["id"] == sid)
    return next(w for w in sc["vo"]["words"] if w["w"].lower().startswith(prefix))["s"]

S = {s["id"]: s for s in scenes}
r = lambda v: round(v, 3)
cues = {
    "s0": {"spark": 0.0, "title": 0.95, "sub": 1.5, "flip": r(S["s0"]["end"] - 0.72), "stretch": r(S["s0"]["end"] - 0.3)},
    "s1": {},
    "s2": {"cb": r(S["s2"]["start"] + 0.02), "nodes": [r(S["s2"]["start"] + 0.16 + i * 0.09) for i in range(5)],
           "hlA": r(word("s2", "nakit")), "hlB": r(word("s2", "ihtiyac")), "flow": r(word("s2", "borç")),
           "cbPulse": r(word("s2", "merkez")), "all": r(word("s2", "yönetir")),
           "zoom": r(S["s2"]["end"] - 0.6), "iris": r(S["s2"]["end"] - 0.26), "irisEnd": r(S["s2"]["end"] + 0.42)},
    "s3": {"cards": [r(word("s3", p) - 0.14) for p in ("hazine", "repo", "bankalar", "finansman")],
           "exit": r(S["s3"]["end"] - 0.62)},
    "s4": {"draw": r(S["s4"]["start"] + 0.12), "hike": r(word("s4", "politika")), "drawEnd": r(min(word("s4", "politika") + 0.55, S["s4"]["end"] - 0.78)),
           "curl": r(S["s4"]["end"] - 0.72)},
    "s5": {"beats": [r(S["s5"]["start"] + d) for d in (0.32, 0.55, 1.42, 1.65)], "fade": r(S["s5"]["end"] - 0.45)},
}
s1 = S["s1"]
t_on, t_12 = word("s1", "gecelik"), word("s1", "aya") + 0.05
mid = [t_on + (t_12 - t_on) * f for f in (0.2, 0.42, 0.62, 0.82)]
cues["s1"] = {"ticks": [r(s1["start"] + 0.12)] + [r(t_on)] + [r(m) for m in mid] + [r(t_12)],
              "go": r(t_on - 0.3), "limit": r(word("s1", "bir")), "wipe": r(s1["end"] - 0.36), "wipeEnd": r(s1["end"] + 0.46)}
tl = {"duration": r(scenes[-1]["end"]), "fps": 60, "scenes": scenes, "cues": cues}
json.dump(tl, open(out, "w"), ensure_ascii=False, indent=1)
for s in scenes:
    print(f'{s["id"]} {s["label"]:9s} {s["start"]:6.2f} → {s["end"]:6.2f}  vo {s["vo"]["start"]:.2f}+{s["vo"]["dur"]:.2f}  ' +
          " ".join(f'{w["w"]}@{w["s"]:.2f}' for w in s["vo"]["words"]))
print("duration", tl["duration"])
