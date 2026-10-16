"""Synthesize the score and sound design from the shared timeline, then mix in the voice-over.

usage: python3 build_audio.py <timeline_json> <vo_dir> <out_wav>
Everything is generated here (no samples): FM electric piano, lounge groove, bells, pop-art hits, whooshes.
"""
import json, os, sys
import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000
tl_path, vo_dir, out = sys.argv[1:4]
TL = json.load(open(tl_path, encoding="utf-8"))
S = {s["id"]: s for s in TL["scenes"]}
CU = TL["cues"]
DUR = TL["duration"]
N = int(DUR * SR) + 1
rng = np.random.default_rng(21)

music = np.zeros((N, 2)); sfx = np.zeros((N, 2)); vo = np.zeros(N)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def hz(name):
    return midi(12 * (int(name[-1]) + 1) + NOTE[name[:-1]])


def stereo(x, pan=0.0):
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], 1) * np.sqrt(2)


def add(buf, x, at, gain=1.0, pan=0.0):
    if x.ndim == 1:
        x = stereo(x, pan)
    i = int(round(at * SR))
    if i < 0:
        x, i = x[-i:], 0
    j = min(len(buf), i + len(x))
    if j > i:
        buf[i:j] += x[: j - i] * gain


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def adsr(n, a=0.005, r=0.05):
    e = np.ones(n)
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] *= np.linspace(1, 0, nr)
    return e


# ---------- instruments ----------
def bell(f, dur=1.6, idx=2.4, ratio=1.41, decay=3.0):
    t = tt(dur)
    I = idx * np.exp(-t * 5)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t)) * np.exp(-t * decay)
    y += .25 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * decay * 2.2)
    return y * adsr(len(t), .002, .05)


def pluck(f, dur=.9, bright=1.0):
    t = tt(dur)
    y = sum((.6 ** (k - 1)) * bright ** (k - 1) * np.sin(2 * np.pi * f * k * t) * np.exp(-t * (5 + 4 * k)) for k in range(1, 6))
    return y * adsr(len(t), .003, .08)


def kick(dur=.42, f0=120, f1=46, amp=1.0):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 32)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8)
    y[:120] += rng.normal(0, .25, 120) * np.linspace(1, 0, 120)
    return y * amp


def boom(dur=1.8, f0=90, f1=36):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 7)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) * adsr(len(t), .004, .2)


def svf_noise(dur, f0, f1, q=.7, shape="bell", seed=0, curve=1.0):
    """White noise through a state-variable band-pass whose centre sweeps from f0 to f1."""
    n = int(dur * SR)
    x = np.random.default_rng(seed).normal(0, 1, n)
    p = (np.arange(n) / n) ** curve
    fc = f0 * (f1 / f0) ** p
    F = 2 * np.sin(np.pi * np.minimum(fc, SR / 6) / SR)
    lo = bp = 0.0
    y = np.empty(n)
    damp = 1 / q
    for i in range(n):
        hi = x[i] - lo - damp * bp
        bp += F[i] * hi
        lo += F[i] * bp
        y[i] = bp
    u = np.arange(n) / n
    if shape == "bell":
        env = np.sin(np.pi * u) ** 2
    elif shape == "rise":
        env = u ** 2.2 * (1 - np.clip((u - .96) / .04, 0, 1))
    else:
        env = np.exp(-u * 5) * (1 - np.exp(-u * 60))
    return y * env / (np.abs(y).max() + 1e-9)


def tick(f=2600, dur=.045, amp=1.0):
    t = tt(dur)
    y = (np.sin(2 * np.pi * f * t) + .4 * np.sin(2 * np.pi * f * 2.3 * t)) * np.exp(-t * 90)
    return y * amp


def flick(seed=0):
    t = tt(.12)
    y = svf_noise(.12, 5200, 2600, q=1.2, shape="hit", seed=seed) * .9
    y += np.sin(2 * np.pi * 140 * t) * np.exp(-t * 40) * .5
    return y


def heartbeat(f=56, amp=1.0):
    t = tt(.32)
    fr = f + 30 * np.exp(-t * 40)
    y = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 14) * (1 - np.exp(-t * 300))
    return y * amp


def saw(f, t, nh=10, fc=1500.0):
    y = np.zeros_like(t)
    for k in range(1, nh + 1):
        if f * k > 9000:
            break
        y += np.sin(2 * np.pi * f * k * t + k * .7) / k ** 1.25 / (1 + (f * k / fc) ** 2)
    return y


# ---------- extra instruments ----------
def epiano(f, dur=1.4, vel=1.0):
    """Two-operator FM electric piano: soft tine bark that decays into a warm sine."""
    t = tt(dur)
    I = (1.6 * vel) * np.exp(-t * 6) + .25
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * t))
    y += .18 * vel * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t * 40)
    return y * np.exp(-t * 1.6) * adsr(len(t), .004, .15)


def snap(seed=0):
    t = tt(.18)
    n = np.random.default_rng(seed).normal(0, 1, len(t))
    b_, a_ = signal.butter(2, [1400 / (SR / 2), 5200 / (SR / 2)], "band")
    return signal.lfilter(b_, a_, n) * np.exp(-t * 38) + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 60) * .4


def shaker(seed=0):
    t = tt(.07)
    n = np.random.default_rng(seed).normal(0, 1, len(t))
    b_, a_ = signal.butter(2, 6500 / (SR / 2), "high")
    return signal.lfilter(b_, a_, n) * np.exp(-t * 70) * (1 - np.exp(-t * 900))


def crash(dur=1.6, seed=0):
    t = tt(dur)
    n = np.random.default_rng(seed).normal(0, 1, (len(t), 2))
    b_, a_ = signal.butter(2, 3500 / (SR / 2), "high")
    return signal.lfilter(b_, a_, n, axis=0) * (np.exp(-t * 2.6) * (1 - np.exp(-t * 300)))[:, None]


def zap(dur=.35, f0=2400, f1=180):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 14)
    return np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * .5 * np.exp(-t * 7) * adsr(len(t), .002, .05)


def bubble(f0=300, f1=1200, dur=.12):
    t = tt(dur)
    f = f0 + (f1 - f0) * (t / dur) ** .6
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / dur) ** 1.5


def scratch(dur=.32, seed=0):
    t = tt(dur)
    n = np.random.default_rng(seed).normal(0, 1, len(t))
    lfo = np.abs(np.sin(2 * np.pi * 7 * t))
    fc = 400 + 2600 * lfo
    y = np.zeros_like(n); lo = bp = 0.0
    for i in range(len(n)):
        Fq = 2 * np.sin(np.pi * fc[i] / SR); hi = n[i] - lo - 1.2 * bp; bp += Fq * hi; lo += Fq * bp; y[i] = bp
    return y / (np.abs(y).max() + 1e-9) * np.sin(np.pi * t / dur) ** .5


# ---------- score: lounge in A minor, 100 BPM ----------
BEAT = 60 / 100
CH = {
    "s0": ["A2", "C4", "E4", "G4", "B4"], "s1": ["F2", "A3", "C4", "E4", "G4"], "s2": ["D2", "F3", "A3", "C4", "E4"],
    "s3": ["G2", "B3", "D4", "E4", "A4"], "s4": ["C2", "E3", "G3", "B3", "D4"], "s5": ["E2", "D3", "G#3", "B3", "F4"],
    "s6": ["A2", "C#4", "E4", "G#4", "B4"],
}
for sid, notes in CH.items():
    a, b2 = S[sid]["start"], S[sid]["end"] + (1.2 if sid == "s6" else 0)
    # sustained warm pad underneath
    t = tt(b2 - a + .8)
    env = np.minimum(1, t / .6) * np.clip((b2 - a + .8 - t) / .8, 0, 1)
    pad = np.zeros((len(t), 2))
    for j, nm in enumerate(notes[1:]):
        for d, pn in ((-6, -.6), (6, .6)):
            pad += stereo(saw(hz(nm) * 2 ** (d / 1200), t + j * .01, nh=6, fc=900) * .16, pn)
    add(music, pad * env[:, None], a, .1)
    # electric piano comping: on the beat and on the "and" of two
    k, x = 0, a + .05
    while x < b2 - .1:
        pos = k % 4
        if pos in (0, 1.5) or k % 8 == 3:
            for j, nm in enumerate(notes[1:]):
                add(music, epiano(hz(nm), 1.3, .8 if pos == 0 else .55), x + j * .012, .055, (-.3, -.1, .1, .3)[j])
        if k % 8 == 5:
            add(music, epiano(hz(notes[-1]) * 2, .9, .6), x + BEAT / 2, .04, .5)
        k += 1; x += BEAT
    # bass: root on 1, fifth on 3
    root = hz(notes[0])
    x, k = a, 0
    while x < b2 - .05:
        f = root if k % 2 == 0 else root * 1.5
        bt = tt(BEAT * .95)
        bass = (np.sin(2 * np.pi * f * bt) + .25 * np.sin(4 * np.pi * f * bt)) * np.exp(-bt * 2.2) * adsr(len(bt), .006, .06)
        add(music, np.tanh(bass * 1.6), x, .3)
        x += BEAT * 2; k += 1

# groove: from the order book to the end of the risk scene
g0, g1 = S["s1"]["start"], S["s5"]["end"] - .2
side = np.ones(N); x, k = g0, 0
while x < g1:
    if k % 2 == 0:
        add(music, kick(.4, 110, 48, 1), x, .36)
        i = int(x * SR); n = int(.35 * SR)
        side[i:i + n] = np.minimum(side[i:i + n], 1 - .4 * np.exp(-np.arange(min(n, N - i)) / SR * 10))
    else:
        add(music, snap(k), x, .16, .15)
    for s16 in range(4):
        add(music, shaker(k * 4 + s16), x + s16 * BEAT / 4, .035 if s16 % 2 else .02, .4)
    x += BEAT; k += 1
music *= side[:, None]

# ---------- sound design ----------
c0, c1, c2, c3, c4, c5, c6 = (CU[k] for k in ("s0", "s1", "s2", "s3", "s4", "s5", "s6"))
TRW = CU["tr"]
# opening: low swell and a gilded chime as the title rises
add(sfx, boom(1.6, 70, 38), .05, .3)
for i, nm in enumerate(["A5", "E6", "B6"]):
    add(sfx, bell(hz(nm), 1.8, 1.2, 1.41, 2.2), c0["title"] + i * .11, .06, -.3 + i * .3)
t = tt(c0["line"][1] - c0["line"][0])
add(sfx, svf_noise(len(t) / SR, 900, 2400, 2.5, "bell", 1) * .5, c0["line"][0], .05)
add(sfx, bell(hz("E6"), 1.0, 1.2, 1.41, 3.5), c0["gap"] + .1, .06)
add(sfx, pluck(hz("A4"), .8, .8), c0["buy"], .16, -.3)
add(sfx, pluck(hz("E5"), .8, .8), c0["sell"], .16, .3)
add(sfx, bell(hz("A6"), 1.2, 1.6, 1.41, 3), c0["sell"] + .32, .05)

# pop transitions
def halftone_sfx(win, seed):
    a, b = win; mid = (a + b) / 2
    r = np.random.default_rng(seed)
    for i in range(26):
        x = a + (mid - a) * (i / 26) ** .7
        add(sfx, tick(r.uniform(2500, 6000), .015, 1), x, .05, r.uniform(-.8, .8))
    add(sfx, bubble(250, 1400, .14), mid - .07, .22)
    add(sfx, svf_noise(b - a, 300, 6000, .8, "bell", seed), a, .18)
halftone_sfx(TRW["t01"], 11)
halftone_sfx(TRW["t45"], 12)

def burst_sfx(win, seed, high=True):
    a, b = win
    add(sfx, svf_noise(.2, 800, 5000, .8, "rise", seed), a - .05, .25)
    hit = a + .12
    add(sfx, boom(1.2, 140, 45), hit, .55)
    add(sfx, crash(.9, seed), hit, .09)
    add(sfx, zap(.3, 2600, 220), hit + .02, .05)
    add(sfx, svf_noise(.3, 5000, 1200, .7, "bell", seed + 1), b - .36, .09)
burst_sfx(TRW["t12"], 21)
add(sfx, bubble(400, 1000, .1), TRW["t12"][0] + .2, .18, -.5)
add(sfx, bubble(300, 800, .1), TRW["t12"][0] + .27, .18, .5)
burst_sfx(TRW["t56"], 31)
add(sfx, scratch(.34, 33), TRW["t56"][0] + .14, .16)

a, b = TRW["t23"]
for i in range(4):
    add(sfx, bubble(300 + i * 120, 1100 + i * 200, .1), a + .02 + i * .05 * (b - a), .2, (-.6, .6, -.6, .6)[i])
    add(sfx, tick(1800, .03, 1), a + .02 + i * .05 * (b - a), .1)
add(sfx, svf_noise(.12, 7000, 3000, 1, "hit", 40), a + .3, .2)
for i in range(4):
    add(sfx, svf_noise(.25, 3000, 7000, .8, "bell", 41 + i), a + .64 * (b - a) + i * .025, .05, (-.7, .7, -.7, .7)[i])

a, b = TRW["t34"]
for j in range(3):
    t0 = a + (j * .08 + .3) * (b - a)
    add(sfx, svf_noise(.22, 600, 3200, .8, "bell", 50 + j), t0 - .18, .2, (-.7, 0, .7)[j])
    add(sfx, kick(.3, 160, 60, 1), t0, .25, (-.5, 0, .5)[j])
add(sfx, zap(.4, 3000, 150), a + .24 * (b - a), .07)

# order book: rows tick in, market order whoosh, match ding, last price uptick
for k in range(5):
    add(sfx, tick(1600 + k * 140, .03, 1), S["s1"]["start"] + .12 + k * .06, .05, -.3)
add(sfx, svf_noise(.45, 600, 3500, .9, "rise", 60), c1["match"] - .05, .2, -.4)
add(sfx, bell(hz("E6"), 1.4, 2.4, 1.0, 2.6), c1["match"] + .35, .16)
add(sfx, bell(hz("B6"), 1.2, 1.6, 1.0, 3.0), c1["match"] + .38, .08)
add(sfx, pluck(hz("A5"), .5, .6), c1["match"] + .72, .1)
add(sfx, svf_noise(.5, 2000, 900, 1.5, "bell", 61), c1["arz"], .05, .6)
add(sfx, svf_noise(.5, 900, 2000, 1.5, "bell", 62), c1["talep"], .05, -.6)

# candle: the price itself is sonified while the candle forms
f0, f1 = c2["form"]
t = tt(f1 - f0)
u = np.clip(t / (f1 - f0), 0, 1)
keys = [[0, 100], [.12, 98.4], [.22, 96.5], [.34, 98.8], [.5, 103.2], [.64, 106.4], [.72, 108], [.8, 105.6], [.9, 104.2], [1, 105]]
price = np.interp(u, [k[0] for k in keys], [k[1] for k in keys])
fr = 440 * 2 ** ((price - 100) / 12 * 1.5)
tone = np.sin(2 * np.pi * np.cumsum(fr) / SR) * (np.minimum(1, t / .08)) * np.clip((t[-1] - t) / .1, 0, 1)
add(sfx, tone * .5, f0, .05)
for nm_t, nm in ((c2["open"], "A4"), (c2["high"], "E5"), (c2["low"], "C4"), (c2["close"], "A5")):
    add(sfx, pluck(hz(nm), .7, .8), nm_t, .14)
add(sfx, bell(hz("C#6"), 1.2, 1.0, 1.41, 3.5), c2["legend"], .05)

# trend: candles tick as they print, the trendline glides, the breakout lifts
for i in range(43):
    x = c3["build"][0] + (c3["build"][1] - c3["build"][0]) * i / 42
    add(sfx, tick(5200 + (i % 5) * 220, .01, 1), x, .012, -.7 + i * .033)
add(sfx, svf_noise(.7, 3000, 7000, 1.2, "bell", 70), c3["trend"], .04, .2)
for k in range(3):
    add(sfx, pluck(hz(["A4", "C5", "E5"][k]), .6, .7), c3["lows"] + k * .14, .12, -.5 + k * .5)
add(sfx, bell(hz("E5"), 1.2, 1.0, 1.41, 3), c3["sup"], .07, -.3)
add(sfx, bell(hz("F5"), 1.2, 1.4, 1.41, 3), c3["res"], .07, .3)
add(sfx, svf_noise(.5, 400, 5000, .8, "rise", 71), c3["brk"] - .25, .2)
add(sfx, kick(.5, 120, 46, 1), c3["brk"] + .2, .3)
add(sfx, bell(hz("A6"), 1.4, 1.8, 1.41, 2.4), c3["brk"] + .22, .1)

# plan: three levels, then the target is hit
add(sfx, pluck(hz("A4"), .9, .9), c4["entry"], .15)
add(sfx, pluck(hz("E4"), .9, .6), c4["stop"], .15, -.3)
add(sfx, pluck(hz("E5"), .9, 1.0), c4["tp"], .15, .3)
add(sfx, bell(hz("B5"), 1.0, 1.0, 1.41, 3.5), c4["tp"] + .45, .05)
for i in range(5):
    add(sfx, tick(2400 + i * 260, .02, 1), c4["run"][0] + (c4["run"][1] - c4["run"][0]) * i / 5, .05, .3)
for i, nm in enumerate(["A5", "C#6", "E6", "A6"]):
    add(sfx, bell(hz(nm), 1.6, 1.8, 1.0, 2.2), c4["run"][1] + i * .05, .07, -.3 + i * .2)

# risk: rule chime, the gold cell, and one soft tick per loss
for i, nm in enumerate(["E5", "B5", "E6"]):
    add(sfx, bell(hz(nm), 1.8, 1.2, 1.41, 2.0), c5["rule"] + i * .1, .06)
add(sfx, bell(hz("G#6"), 1.6, 2.0, 1.41, 2.4), c5["cell"], .1)
for i in range(10):
    add(sfx, tick(1500 - i * 60, .03, 1), c5["sim"][0] + (c5["sim"][1] - c5["sim"][0]) * (i + 1) / 10, .06, .2)

# system: ring nodes rise through the scale, the title lands on a full chord
for k, nm in enumerate(["A4", "C#5", "E5", "G#5", "B5"]):
    add(sfx, pluck(hz(nm), 1.0, .9), c6["ring"][0] + k * .24, .13, -.6 + k * .3)
add(sfx, boom(2.0, 60, 32), c6["title"], .14)
for j, nm in enumerate(["A3", "E4", "A4", "C#5", "E5", "G#5"]):
    add(sfx, epiano(hz(nm), 3.0, .9), c6["disc"] - .05 + j * .02, .07, -.4 + j * .16)
add(sfx, boom(2.0, 60, 32), c6["disc"] - .05, .3)
for i, nm in enumerate(["E6", "A6", "C#7", "E7"]):
    add(sfx, bell(hz(nm), 2.4, 1.0, 2.0, 1.4), c6["disc"] + .15 + i * .14, .045, -.6 + i * .4)

# ---------- reverb ----------
def make_ir(dur=2.4, decay=.42, seed=50):
    t = tt(dur)
    r = np.random.default_rng(seed)
    ir = r.normal(0, 1, (len(t), 2)) * np.exp(-t / decay)[:, None]
    b_, a_ = signal.butter(1, 5500 / (SR / 2))
    ir = signal.lfilter(b_, a_, ir, axis=0)
    ir[: int(.012 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(0))


IR = make_ir()
send = music * .4 + sfx * .45
wet = np.stack([signal.fftconvolve(send[:, ch], IR[:, ch])[:N] for ch in range(2)], 1)

# ---------- voice-over ----------
hp_b, hp_a = signal.butter(2, 75 / (SR / 2), "high")
for sc in TL["scenes"]:
    x, sr = sf.read(os.path.join(vo_dir, sc["id"] + ".wav"), dtype="float64")
    x = signal.resample_poly(x, 320, 147) if sr == 22050 else x
    x = signal.lfilter(hp_b, hp_a, x)
    env = np.sqrt(np.convolve(x ** 2, np.ones(480) / 480, "same")) + 1e-6
    thr = np.percentile(env, 80) * .7
    g = np.where(env > thr, (thr / env) ** (1 - 1 / 2.5), 1.0)
    g = np.convolve(g, np.ones(240) / 240, "same")
    x = x * g
    x *= 10 ** (-18 / 20) / np.sqrt(np.mean(x[np.abs(x) > .01 * np.abs(x).max()] ** 2))
    i = int(sc["vo"]["start"] * SR)
    j = min(N, i + len(x))
    vo[i:j] += x[: j - i]
vo_room = np.stack([signal.fftconvolve(vo, make_ir(.9, .18, 60)[:, ch])[:N] for ch in range(2)], 1)

# ducking under the voice
venv = np.abs(vo)
k_att, k_rel = np.exp(-1 / (.01 * SR)), np.exp(-1 / (.28 * SR))
duck = np.empty(N); level = 0.0
for i in range(N):
    v = venv[i]
    level = k_att * level + (1 - k_att) * v if v > level else k_rel * level + (1 - k_rel) * v
    duck[i] = level
duck = np.clip(duck / (np.percentile(duck[duck > 1e-4], 60) + 1e-9), 0, 1)

# carve a pocket for the narrator: zero-phase dip of the bed's speech band while the voice is active
bed = music + wet * .6
bb, ba = signal.butter(2, [900 / (SR / 2), 4200 / (SR / 2)], "band")
bed = bed - signal.filtfilt(bb, ba, bed, axis=0) * (.65 * duck)[:, None]
mixL = bed * (1 - .62 * duck)[:, None] * .62 + sfx * (1 - .45 * duck)[:, None] * .8
mixL += stereo(vo, 0) * .62 + vo_room * .05
fade = np.clip((DUR - np.arange(N) / SR) / .6, 0, 1)
fade_in = np.clip(np.arange(N) / (.004 * SR), 0, 1)
mixL *= (fade * fade_in)[:, None]
pk = np.abs(mixL).max()
mixL = mixL / pk * .98
mixL = np.tanh(mixL * 1.25) / np.tanh(1.25) * .89
sf.write(out, mixL.astype(np.float32), SR, subtype="PCM_24")
vo_rms = 20 * np.log10(np.sqrt(np.mean((mixL[np.abs(vo) > .005]) ** 2)) + 1e-9)
print(f"wrote {out}: {N / SR:.2f}s, voice-section RMS {vo_rms:.1f} dBFS, peak {20*np.log10(np.abs(mixL).max()):.1f} dBFS")
