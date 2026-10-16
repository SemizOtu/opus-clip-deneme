"""Synthesize the score and sound design from the shared timeline, then mix in the voice-over.

usage: python3 build_audio.py <timeline_json> <vo_dir> <out_wav>
Everything is generated here (no samples): pads, pulse, plucks, bells, whooshes, heartbeat.
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


NOTE = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


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


# ---------- score ----------
CHORDS = [
    ("s0", ["D2", "D3", "F3", "A3", "C4", "E4"]),
    ("s1", ["Bb1", "Bb2", "D3", "F3", "A3", "C4"]),
    ("s2", ["G1", "G2", "Bb2", "D3", "F3", "A3"]),
    ("s3", ["F1", "F2", "A2", "C3", "E3", "G3"]),
    ("s4a", ["A1", "A2", "D3", "E3", "G3", "B3"]),
    ("s4b", ["A1", "A2", "C#3", "E3", "G3", "B3"]),
    ("s5", ["D2", "D3", "F3", "A3", "C4", "E4"]),
    ("end", ["D2", "D3", "F#3", "A3", "C#4", "E4"]),
]
bounds = {
    "s0": (0.0, S["s0"]["end"]), "s1": (S["s1"]["start"], S["s1"]["end"]), "s2": (S["s2"]["start"], S["s2"]["end"]),
    "s3": (S["s3"]["start"], S["s3"]["end"]), "s4a": (S["s4"]["start"], CU["s4"]["hike"]),
    "s4b": (CU["s4"]["hike"], S["s4"]["end"]), "s5": (S["s5"]["start"], S["s5"]["end"] - 1.25), "end": (S["s5"]["end"] - 1.25, DUR + .5),
}
for key, notes in CHORDS:
    a, b = bounds[key]
    a0, b0 = max(0, a - .25), b + .7
    t = tt(b0 - a0)
    env = np.minimum(1, t / (.9 if key == "s0" else .45)) * np.clip((b0 - a0 - t) / .7, 0, 1)
    seg = np.zeros((len(t), 2))
    for j, nm in enumerate(notes):
        f = hz(nm)
        if j == 0:
            seg += stereo(np.sin(2 * np.pi * f * t) * .55 + .12 * np.sin(4 * np.pi * f * t), 0)
            continue
        for d, pan in ((-7, -.7), (0, 0), (7, .7)):
            ff = f * 2 ** (d / 1200)
            v = saw(ff, t + j * .013, fc=1100 + 500 * np.sin(t * .6 + j)) * (1 + .12 * np.sin(2 * np.pi * (.21 + j * .05) * t))
            seg += stereo(v * .22, pan * (.5 + .1 * j))
    add(music, seg * env[:, None], a0, gain=.16)

# pulse: soft half-time kick and ticking hats from the maturity scene until the curl
beat0, beat1 = S["s1"]["start"], CU["s4"]["curl"]
side = np.ones(N)
b = beat0
k = 0
while b < beat1:
    if k % 2 == 0:
        add(music, kick(amp=.9), b, gain=.32)
        i = int(b * SR); n = int(.42 * SR)
        side[i:i + n] = np.minimum(side[i:i + n], 1 - .55 * np.exp(-np.arange(min(n, N - i)) / SR * 9))
    add(music, tick(5200 + 300 * (k % 2), .03, .6), b + .25, gain=.05, pan=.35)
    add(music, tick(6000, .02, .5), b, gain=.03, pan=-.35)
    b += .5; k += 1
music *= side[:, None]

# arpeggio across the actors and instruments scenes
arp_notes = {"s2": ["G4", "Bb4", "D5", "F5", "A5", "D5"], "s3": ["F4", "A4", "C5", "E5", "G5", "C5"]}
for sid, ns in arp_notes.items():
    a, b2 = S[sid]["start"], S[sid]["end"]
    x, k = a + .02, 0
    while x < b2 - .1:
        add(music, pluck(hz(ns[k % len(ns)]), .7, .8), x, gain=.07, pan=(-.45, .45)[k % 2])
        x += .25; k += 1

# ---------- sound design ----------
c0, c1, c2, c3, c4, c5 = (CU[k] for k in ("s0", "s1", "s2", "s3", "s4", "s5"))
add(sfx, boom(1.4), 0.0, .5)
add(sfx, bell(hz("C6"), 1.6, 2.0, 1.41, 2.6), .0, .16, -.1)
add(sfx, bell(hz("G6"), 1.4, 1.6, 1.41, 3.0), .06, .09, .2)
add(sfx, svf_noise(.8, 400, 5200, .9, "bell", 1, .7), .04, .06)
for i, nm in enumerate(["D5", "F5", "A5", "C6", "E6"]):
    add(sfx, bell(hz(nm), 1.0, 1.0, 2.0, 4.5), c0["title"] + i * .07, .045, -.4 + i * .2)
sub_n = 29
for i in range(sub_n):
    add(sfx, tick(5600 + rng.uniform(-500, 500), .015, 1), c0["sub"] + i * .62 / sub_n, .02, .1)
add(sfx, svf_noise(.5, 500, 3800, .8, "bell", 2), c0["flip"] + .02, .28)
add(sfx, bell(hz("A6"), .8, 1.0, 1.5, 6), c0["flip"] + .1, .06)
t = tt(.28)
add(sfx, np.sin(2 * np.pi * np.cumsum(300 * (6 ** (t / .28))) / SR) * np.sin(np.pi * t / .28) ** 2, c0["stretch"], .05)
add(sfx, svf_noise(.32, 1500, 7000, 1.0, "bell", 3), c0["stretch"], .12)

# maturity: tick pops, odometer clicks, limit thud, year accent
pent = ["D5", "E5", "G5", "A5", "B5", "D6", "E6"]
for i, tk in enumerate(c1["ticks"]):
    add(sfx, pluck(hz(pent[i]), .5, .5), tk, .14, -.6 + i * .2)
    add(sfx, tick(hz(pent[i]) * 2, .03, .5), tk, .07, -.6 + i * .2)


def head_days(x):
    T = list(c1["ticks"]); T[0] = c1["go"]
    D = [0, 1, 7, 30, 91, 182, 365]
    if x <= T[0]:
        return 0.0
    lx = lambda d: np.log1p(d) / np.log(366)
    for i in range(6):
        if x < T[i + 1]:
            u = (x - T[i]) / (T[i + 1] - T[i]); e = 4 * u ** 3 if u < .5 else 1 - (-2 * u + 2) ** 3 / 2
            p = lx(D[i]) + (lx(D[i + 1]) - lx(D[i])) * e
            return float(np.exp(p * np.log(366)) - 1)
    return 365.0


prev, x = 0, c1["go"]
while x < c1["ticks"][6] + .05:
    d = int(head_days(x))
    step = 1 if d < 40 else 5 if d < 120 else 12
    if d // step != prev // step:
        add(sfx, tick(4200, .012, 1), x, .035, .3)
        prev = d
    x += .002
add(sfx, kick(.5, 90, 40, 1), c1["limit"], .35)
add(sfx, bell(hz("D4"), 1.4, .8, 1.0, 2.5), c1["limit"], .1)
add(sfx, bell(hz("D6"), 1.6, 1.8, 1.41, 2.2), c1["ticks"][6] + .02, .16)
w0, w1 = c1["wipe"], c1["wipeEnd"]
wh = svf_noise(w1 - w0 + .1, 380, 5200, .85, "bell", 4, .8)
pan = np.linspace(-.95, .95, len(wh))
add(sfx, np.stack([wh * np.cos((pan + 1) * np.pi / 4), wh * np.sin((pan + 1) * np.pi / 4)], 1) * 1.4, w0, .3)
for i, nm in enumerate(["A6", "D7", "E7"]):
    add(sfx, bell(hz(nm), .9, .8, 1.41, 5), w0 + .25 + i * .09, .03, -.5 + i * .5)

# actors
add(sfx, kick(.6, 80, 42, 1), c2["cb"], .25)
add(sfx, bell(hz("D4"), 1.6, 1.0, 1.0, 2.0), c2["cb"] + .05, .12)
for i, (tn, nm) in enumerate(zip(c2["nodes"], ["A4", "C5", "D5", "F5", "G5"])):
    add(sfx, pluck(hz(nm), .6, .7), tn + .05, .13, [-.7, .7, -.8, .8, 0][i])
hlA = max(c2["hlA"], c2["nodes"][0] + .3)
for i, nm in enumerate(["D5", "A5"]):
    add(sfx, bell(hz(nm), 1.0, 1.2, 2.0, 3), hlA + i * .08, .1, -.6)
for i, nm in enumerate(["A4", "F4"]):
    add(sfx, bell(hz(nm), 1.0, 1.4, 1.5, 3), c2["hlB"] + i * .09, .1, .6)
for i in range(12):
    at = c2["flow"] + i * .1 + rng.uniform(0, .05)
    add(sfx, bell(rng.uniform(2600, 4200), .25, 1.5, 2.3, 18), at, .045, -.8 + i * .14)
add(sfx, boom(1.2, 70, 34), c2["cbPulse"], .45)
add(sfx, bell(hz("D3"), 2.0, 1.2, 1.0, 1.5), c2["cbPulse"], .1)
rs = c2["zoom"] - .05
rise = svf_noise(c2["iris"] + .5 - rs, 220, 6500, .7, "rise", 5, 1.3)
add(sfx, rise, rs, .32)
t = tt(c2["iris"] + .5 - rs)
add(sfx, np.sin(2 * np.pi * np.cumsum(180 * 5 ** (t / t[-1]) ** 2) / SR) * (t / t[-1]) ** 2, rs, .05)
hit = c2["iris"] + .45
add(sfx, boom(1.6, 110, 40), hit, .6)
add(sfx, svf_noise(1.0, 6000, 900, .8, "hit", 6), hit, .2)
add(sfx, bell(hz("F5"), 1.8, 2.2, 1.41, 1.8), hit, .12)

# instruments: banknote flicks
for i, tc in enumerate(c3["cards"]):
    add(sfx, flick(10 + i), tc + .04, .4, (-.4, .4, -.4, .4)[i])
    add(sfx, pluck(hz(["D5", "F5", "A5", "C6"][i]), .8, .9), tc + .06, .13, (-.4, .4, -.4, .4)[i])
    add(sfx, svf_noise(.6, 3000, 9000, 1.2, "bell", 20 + i), tc + .3, .04)
ex = c3["exit"]
add(sfx, svf_noise(.5, 2400, 500, .8, "bell", 30), ex, .22, -.3)
add(sfx, svf_noise(S["s3"]["end"] - ex + .1, 300, 3000, .8, "rise", 31), ex + .02, .25, .2)
add(sfx, kick(.5, 100, 45, 1), S["s3"]["end"], .3)

# rates
add(sfx, svf_noise(c4["drawEnd"] - c4["draw"], 1800, 2600, 2.5, "bell", 40), c4["draw"], .03)
t = tt(.35)
add(sfx, np.sin(2 * np.pi * np.cumsum(hz("A4") * (4 / 3) ** np.minimum(1, t / .25)) / SR) * np.exp(-t * 4) * adsr(len(t), .01, .05), c4["hike"], .14)
add(sfx, kick(.5, 110, 44, 1), c4["hike"], .35)
add(sfx, bell(hz("D6"), 1.4, 1.6, 1.41, 2.5), c4["hike"] + .05, .12)
cs = c4["curl"]
sw = svf_noise(S["s4"]["end"] - cs + .15, 600, 2400, 1.5, "bell", 41)
ph = np.linspace(0, 2 * np.pi * 1.5, len(sw))
add(sfx, np.stack([sw * (.5 + .5 * np.sin(ph)), sw * (.5 - .5 * np.sin(ph))], 1) * 1.6, cs, .28)
add(sfx, bell(hz("D6"), 2.4, 2.0, 1.41, 1.5), S["s4"]["end"], .18)
add(sfx, bell(hz("A5"), 2.4, 1.6, 1.41, 1.7), S["s4"]["end"] + .03, .1, .3)

# summary: heartbeat and shimmer
for i, bt in enumerate(c5["beats"]):
    add(sfx, heartbeat(56 if i % 2 == 0 else 62, 1 if i % 2 == 0 else .7), bt, .9)
words = S["s5"]["vo"]["words"]
for i, nm in enumerate(["D6", "A6", "E7", "F#7"]):
    add(sfx, bell(hz(nm), 2.2, 1.0, 2.0, 1.4), words[-1]["s"] + .1 + i * .12, .05, -.6 + i * .4)

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
send = music * .35 + sfx * .5
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
