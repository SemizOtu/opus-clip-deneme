"""Build the shared timeline (scene times, word timings, cue times) from the voice-over takes.

usage: python3 build_timeline.py <vo_dir> <out_json>   (needs <vo_dir>/parts.json from tts_parts.py)
Both the canvas renderer and the sound design read this file, so picture and sound share one clock.
"""
import json, os, re, sys
import numpy as np
import soundfile as sf

vo_dir, out = sys.argv[1], sys.argv[2]
lines = json.load(open(os.path.join(os.path.dirname(__file__), "narration.json")))
META = {
    "s0": ("tanidik", "TANIDIK BİR AN", 1.25, 0.30),
    "s1": ("dongu", "SORUN · DÖNGÜ", 0.32, 0.30),
    "s2": ("derin", "SORUNUN ALTINDA", 0.40, 0.35),
    "s3": ("donum", "DÖNÜM NOKTASI", 0.42, 0.35),
    "s4": ("adim1", "ADIM 1 · YUMUŞAK BAŞLANGIÇ", 0.32, 0.35),
    "s5": ("adim2", "ADIM 2 · MOLA", 0.32, 0.30),
    "s6": ("adim3", "ADIM 3 · DİNLE VE YANSIT", 0.32, 0.35),
    "s7": ("adim4", "ADIM 4 · ONARIM", 0.32, 0.35),
    "s8": ("kapanis", "KAPANIŞ", 0.36, 2.60),
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

PARTS = json.load(open(os.path.join(vo_dir, "parts.json"), encoding="utf-8"))
scenes, t = [], 0.0
for ln in lines:
    key, label, lead, tail = META[ln["id"]]
    x, sr = sf.read(os.path.join(vo_dir, ln["id"] + ".wav"), dtype="float32")
    dur = len(x) / sr
    vs = t + lead
    wt = []
    for k, part in enumerate(PARTS[ln["id"]]):
        px, psr = sf.read(os.path.join(vo_dir, "parts", f'{ln["id"]}_{k}.wav'), dtype="float32")
        for w in word_times(px, psr, part["text"].split()):
            wt.append({"w": w["w"], "s": round(w["s"] + part["start"] + vs, 3), "e": round(w["e"] + part["start"] + vs, 3)})
    end = vs + dur + tail
    scenes.append({"id": ln["id"], "key": key, "label": label, "start": round(t, 3), "end": round(end, 3),
                   "vo": {"start": round(vs, 3), "dur": round(dur, 3), "text": ln["text"].replace(" | ", " "), "words": wt,
                          "parts": [{"text": p["text"], "s": round(p["start"] + vs, 3), "e": round(p["start"] + p["dur"] + vs, 3)} for p in PARTS[ln["id"]]]}})
    t = end

S = {s["id"]: s for s in scenes}
r = lambda v: round(v, 3)
clean = lambda w: w.lower().strip("“”\"'.,;:!?")

def W(sid, prefix, k="s", nth=0):
    hits = [w for w in S[sid]["vo"]["words"] if clean(w["w"]).startswith(prefix)]
    return hits[nth][k]

last = lambda sid: S[sid]["vo"]["words"][-1]["e"]
st = lambda sid: S[sid]["start"]
cues = {
    "title": [0.0, r(S["s0"]["vo"]["start"] + .8)],
    "s0": {"chars": r(S["s0"]["vo"]["start"] - .35), "bubble": r(W("s0", "tartışma")), "rew": [r(W("s0", "tekrar")), r(W("s0", "tekrar", nth=1))],
           "topics": r(W("s0", "konu")), "turn": r(W("s0", "sonu"))},
    "s1": {"blame": r(W("s1", "suçlar")), "quiet": r(W("s1", "susar")), "chase": r(W("s1", "yaklaş")), "away": r(W("s1", "uzaklaş")), "ring": r(W("s1", "yaklaş") + .1)},
    "s2": {"flood": [r(st("s2") - .35), r(st("s2") + .7)], "words": r(W("s2", "bulaşık")), "pan": [r(W("s2", "altında") - .5), r(W("s2", "beni") - .1)],
           "ask": r(W("s2", "beni"))},
    "s3": {"rise": [r(st("s3") - .2), r(st("s3") + 1.0)], "ring": r(W("s3", "döngü")), "brk": r(W("s3", "kırmak")), "path": r(W("s3", "birkaç")), "stones": r(W("s3", "adım"))},
    "s4": {"soft": r(W("s4", "yumuşak")), "harsh": r(W("s4", "sen")), "strike": r(W("s4", "yerine")), "ben": r(W("s4", "ben"))},
    "s5": {"race": r(W("s5", "kalbin")), "stop": r(W("s5", "dur")), "timer": [r(W("s5", "yirmi")), r(W("s5", "ama"))], "back": r(W("s5", "geri"))},
    "s6": {"speak": r(st("s6") + .15), "listen": r(W("s6", "dinle")), "ask": r(W("s6", "seni"))},
    "s7": {"smile": r(W("s7", "gülümseme")), "touch": r(W("s7", "dokunuş")), "right": r(W("s7", "haklısın"))},
    "s8": {"heart": r(W("s8", "sevgi")), "line2": r(W("s8", "tartıştıktan")), "card": r(last("s8") + .6), "fade": r(S["s8"]["end"] - .5)},
}
tl = {"duration": r(scenes[-1]["end"]), "fps": 60, "scenes": scenes, "cues": cues}
json.dump(tl, open(out, "w"), ensure_ascii=False, indent=1)
for s in scenes:
    print(f'{s["id"]} {s["label"][:16]:16s} {s["start"]:6.2f} → {s["end"]:6.2f}  ' + " ".join(f'{w["w"]}@{w["s"]:.2f}' for w in s["vo"]["words"]))
print("duration", tl["duration"])
print(json.dumps(cues, ensure_ascii=False))
