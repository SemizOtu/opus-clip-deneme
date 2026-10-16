"""Synthesize the score and sound design from the shared timeline, then mix in the voice-over.

usage: python3 build_audio.py <timeline_json> <vo_dir> <out_wav>
Everything is generated here (no samples): felt piano, plucked guitar, strings, storm, water, heartbeat, breath, birds.
The music follows the emotional arc: warm dusk, storm, underwater loneliness, dawn, resolution.
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


NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "Gb": 6, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


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


# ---------- instruments for this piece ----------
def piano(f, dur=3.0, vel=1.0):
    """Felt piano: slightly inharmonic partials, soft hammer, faster decay for upper partials."""
    t = tt(dur)
    y = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + .0004 * k * k)
        if fk > 12000:
            break
        y += np.sin(2 * np.pi * fk * t + k) * (vel ** (k * .35)) / k ** 1.3 * np.exp(-t * (.9 + .55 * k))
    y *= 1 - np.exp(-t * 400)
    th = np.random.default_rng(int(f)).normal(0, 1, min(len(t), 600)) * np.exp(-np.arange(min(len(t), 600)) / 90)
    y[:len(th)] += th * .02
    return y * adsr(len(t), .004, .4)


def guitar(f, dur=2.2, bright=.5, seed=0):
    """Karplus-Strong string."""
    n = int(SR / f)
    ex = np.random.default_rng(seed).normal(0, 1, n)
    ex = signal.lfilter([bright, 1 - bright], [1], ex)
    x = np.zeros(int(dur * SR)); x[:n] = ex
    a = np.zeros(n + 2); a[0] = 1; a[n] = -.4985; a[n + 1] = -.4985
    y = signal.lfilter([1], a, x)
    return y / (np.abs(y).max() + 1e-9) * adsr(len(y), .002, .3)


def strings(freqs, dur, attack=.9, trem=0.0, fc=1500.0):
    t = tt(dur)
    out = np.zeros((len(t), 2))
    for j, f in enumerate(freqs):
        for d, pn in ((-9, -.8), (-3, -.3), (4, .3), (10, .8)):
            vib = 1 + .003 * np.sin(2 * np.pi * (5 + j * .3) * t + d)
            ph = 2 * np.pi * np.cumsum(f * 2 ** (d / 1200) * vib) / SR
            v = sum(np.sin(k * ph + k * d) / k ** 1.2 / (1 + (f * k / fc) ** 2) for k in range(1, 8))
            out += stereo(v * .12, pn)
    env = np.minimum(1, t / attack) * np.clip((dur - t) / 1.0, 0, 1)
    if trem:
        env = env * (1 - trem * (.5 + .5 * np.sin(2 * np.pi * 7 * t)))
    return out * env[:, None]


def noise_band(dur, lo, hi, seed=0):
    n = np.random.default_rng(seed).normal(0, 1, int(dur * SR))
    b_, a_ = signal.butter(2, [lo / (SR / 2), hi / (SR / 2)], "band")
    return signal.lfilter(b_, a_, n)


def breath(dur=1.1, seed=0):
    t = tt(dur)
    y = noise_band(dur, 400, 1800, seed)
    return y * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2 / (np.abs(y).max() + 1e-9)


def chirp_bird(seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros(int(.5 * SR)); x = 0
    for _ in range(r.integers(2, 5)):
        d = r.uniform(.05, .09); t = tt(d)
        f0, f1 = r.uniform(2800, 3600), r.uniform(4200, 5600)
        f = f0 + (f1 - f0) * np.sin(np.pi * t / d)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2
        i = int(x * SR); out[i:i + len(s)] += s[: len(out) - i]; x += d + r.uniform(.02, .06)
    return out


def plop(seed=0):
    r = np.random.default_rng(seed); d = r.uniform(.03, .06); t = tt(d)
    f = r.uniform(250, 500) * (1 + 3 * t / d)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d)


def thunder(dur=3.0, seed=0):
    t = tt(dur)
    n = np.random.default_rng(seed).normal(0, 1, len(t))
    b_, a_ = signal.butter(2, 180 / (SR / 2))
    y = signal.lfilter(b_, a_, n) * (np.exp(-t * 1.2) * (1 - np.exp(-t * 12))) * (1 + .5 * np.sin(2 * np.pi * 3 * t))
    return y / (np.abs(y).max() + 1e-9)


def crackle(dur=.35, seed=0):
    y = noise_band(dur, 1500, 4500, seed)
    t = tt(dur)
    gate = (np.random.default_rng(seed + 1).random(len(t)) > .7).astype(float)
    gate = np.convolve(gate, np.ones(60) / 60, "same")
    return np.tanh(y * 3) * gate * np.exp(-t * 4) / 1.0


def chord_arp(notes, start, end, step, vel=.8, gain=.1, spread=.5, inst=piano):
    x, k = start, 0
    while x < end - .05:
        nm = notes[k % len(notes)]
        add(music, inst(hz(nm), 2.6, vel), x, gain, (-spread, spread)[k % 2])
        x += step; k += 1


def heartbeats(t0, t1, bpm0, bpm1, gain):
    x = t0
    while x < t1:
        u = (x - t0) / max(t1 - t0, 1e-3); bpm = bpm0 + (bpm1 - bpm0) * u
        add(sfx, heartbeat(56, 1), x, gain); add(sfx, heartbeat(62, .7), x + .22 * 72 / bpm, gain)
        x += 60 / bpm


S_ = {k: v for k, v in S.items()}
st = lambda k: S_[k]["start"]; en = lambda k: S_[k]["end"]
c0, c1, c2, c3, c4, c5, c6, c7, c8 = (CU[k] for k in ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"))

# ---------- warm dusk: felt piano, soft wind ----------
chord_arp(["F3", "C4", "A4", "E5", "A4", "C4"], 0.25, 2.6, .42, .7, .12)
chord_arp(["D3", "A3", "F4", "C5", "F4", "A3"], 2.6, 5.1, .42, .7, .11)
chord_arp(["Bb2", "F3", "D4", "A4", "D4", "F3"], 5.1, c0["turn"] + .1, .42, .65, .1)
add(music, stereo(noise_band(en("s0"), 200, 900, 3) * .05, 0) * np.minimum(1, tt(en("s0")) / 1.5)[:, None], 0, 1)
for x in np.arange(c0["bubble"], c0["turn"], .5):
    add(sfx, tick(2600, .02, 1), x, .02, .5)
for r_ in c0["rew"]:
    for i in range(9):
        t = tt(.04); add(sfx, np.sin(2 * np.pi * np.cumsum(900 + 2600 * t / .04) / SR) * np.sin(np.pi * t / .04), r_ + i * .042, .05, .3)
    t = tt(.4); add(sfx, np.sin(2 * np.pi * np.cumsum(700 * np.exp(-t * 4)) / SR) * np.exp(-t * 5), r_ + .38, .08)
for i in range(3):
    add(sfx, bell(hz(["A5", "C6", "E6"][i]), .5, .6, 2.0, 7), c0["topics"] + i * .38, .03)
add(music, piano(hz("E2"), 3.5, .9) + piano(hz("Bb2"), 3.5, .9), c0["turn"], .14)
add(sfx, boom(1.2, 70, 40), c0["turn"] + .05, .18)

# ---------- storm: drone, dissonant tremolo strings, rain, thunder on the flashes, rising heartbeat ----------
s1a, s1b = st("s1") - .2, c2["flood"][0] + .4
t = tt(s1b - s1a)
drone = (np.sin(2 * np.pi * hz("D2") * t) + .6 * np.sin(2 * np.pi * hz("Eb2") * t) + .3 * np.sin(2 * np.pi * hz("D1") * t)) * np.minimum(1, t / .8) * np.clip((t[-1] - t) / .5, 0, 1)
add(music, stereo(drone * .5, 0), s1a, .22)
add(music, strings([hz("D3"), hz("F3"), hz("Ab3"), hz("Db4")], s1b - s1a, attack=1.2, trem=.5, fc=1000), s1a, .36)
rain = noise_band(s1b - s1a + .5, 2500, 9000, 9)
add(sfx, stereo(rain * .03 * np.minimum(1, tt(s1b - s1a + .5) / .8), .2), s1a, 1)
for k, tf in enumerate((c1["blame"] + .05, c1["blame"] + .5, c1["chase"] + .1)):
    add(sfx, thunder(3.0, int(tf * 10)), tf + .08, (.3, .22, .14)[k], -.3)
for k in range(3):
    add(sfx, crackle(.3, 40 + k), c1["blame"] + k * .33, .08, .3)
heartbeats(st("s1"), c2["flood"][0] + .3, 76, 112, .4)
add(sfx, kick(.5, 70, 40, 1), c1["quiet"], .2)
add(sfx, svf_noise(1.3, 300, 2200, 1.4, "rise", 44), c1["chase"], .05)
sw = svf_noise(1.6, 500, 1600, 1.6, "bell", 45); ph = np.linspace(0, 4 * np.pi, len(sw))
add(sfx, np.stack([sw * (.5 + .5 * np.sin(ph)), sw * (.5 - .5 * np.sin(ph))], 1), c1["ring"], .06)

# ---------- beneath the surface: splash, everything muffled, lonely high notes, bubbles, sonar ----------
add(sfx, svf_noise(1.1, 4000, 300, .8, "hit", 50), c2["flood"][0] + .1, .35)
add(sfx, boom(2.0, 60, 30), c2["flood"][0] + .4, .3)
u0, u1 = c2["flood"][1], st("s3") + .5
t = tt(u1 - u0); rum = noise_band(u1 - u0, 30, 120, 51)
add(music, stereo(rum / (np.abs(rum).max() + 1e-9) * .25 * np.minimum(1, t / 1) * np.clip((t[-1] - t) / .8, 0, 1), 0), u0, 1)
for x, nm in zip(np.arange(c2["flood"][1] + .3, st("s3"), 1.25), ["C6", "G5", "E6", "D6", "A5"]):
    add(music, piano(hz(nm), 4.0, .5), x, .1, .4 if int(x) % 2 else -.4)
r = np.random.default_rng(5)
for x in r.uniform(c2["flood"][1], st("s3") + .6, 26):
    add(sfx, plop(int(x * 100)), x, .05, r.uniform(-.8, .8))
for k, d in enumerate((0, .38, .76)):
    t = tt(1.6); add(sfx, np.sin(2 * np.pi * 1180 * t) * np.exp(-t * 3) * (1 - np.exp(-t * 200)), c2["ask"] + .05 + d, .12 * (.55 ** k))

# ---------- turning point: a breath, then warmth ----------
add(sfx, breath(1.2, 60), c3["rise"][0] + .25, .18)
add(music, strings([hz("F2"), hz("C3"), hz("A3"), hz("G4"), hz("C5")], en("s3") - c3["rise"][0] + 1.0, attack=1.6, fc=1800), c3["rise"][0] + .5, .5)
add(sfx, bell(hz("G5"), 2.6, 1.8, 1.41, 1.6), c3["brk"], .1)
add(sfx, svf_noise(.8, 3000, 9000, 1, "bell", 61), c3["brk"], .06)
for k, nm in enumerate(["C5", "D5", "F5", "A5"]):
    add(music, guitar(hz(nm), 2.0, .5, k), c3["stones"] + k * .14, .14, -.5 + k * .33)

# ---------- step 1: piano returns; harsh scratch, eraser, warm bloom ----------
prog = [("Bb2", ["Bb3", "D4", "F4", "A4"]), ("A2", ["A3", "C4", "F4", "C5"]), ("G2", ["G3", "Bb3", "D4", "F4"]), ("C3", ["C4", "E4", "G4", "Bb4"])]
def piano_bars(t0, t1, bars, step=.36, vel=.75, gain=.1):
    span = (t1 - t0) / len(bars)
    for i, (bass, ns) in enumerate(bars):
        a = t0 + i * span
        add(music, piano(hz(bass), 3.0, .8), a, gain * 1.2, -.1)
        chord_arp(ns, a + step / 2, a + span, step, vel, gain)
piano_bars(st("s4"), en("s4"), prog)
add(sfx, crackle(.35, 70), c4["harsh"] + .05, .05)
t = tt(.42); er = noise_band(.42, 1500, 4500, 71) * (.6 + .4 * np.sin(2 * np.pi * 26 * t)) * np.sin(np.pi * t / .42)
add(sfx, er / (np.abs(er).max() + 1e-9), c4["strike"] + .05, .07)
for j, nm in enumerate(["F4", "A4", "C5", "E5"]):
    add(music, piano(hz(nm), 3.5, .7), c4["ben"] + j * .05, .08, -.3 + j * .2)
add(sfx, bell(hz("E6"), 2.0, 1.0, 2.0, 2.2), c4["ben"] + .25, .04)

# ---------- step 2: racing heart, a held breath, then slow breathing pad as the pulse settles ----------
heartbeats(c5["race"] - .3, c5["stop"], 98, 108, .55)
add(sfx, tick(1500, .05, 1), c5["stop"], .15)
heartbeats(c5["timer"][0] + .2, c5["back"] + .6, 92, 64, .4)
t = tt(en("s5") - c5["timer"][0])
pad = strings([hz("F3"), hz("A3"), hz("C4"), hz("E4")], en("s5") - c5["timer"][0] + .6, attack=1.0, fc=1100)
br_env = .65 + .35 * np.sin(2 * np.pi * .28 * tt(len(pad) / SR))
add(music, pad * br_env[:, None], c5["timer"][0], .4)
for x in np.arange(c5["timer"][0], c5["timer"][1] + .4, .5):
    add(sfx, tick(3200, .015, 1), x, .02, .3)
add(sfx, bell(hz("A5"), 2.2, 1.0, 2.0, 2.0), c5["back"], .06)
piano_bars(c5["back"] - .1, en("s5") + .2, [("F2", ["F3", "A3", "C4", "E4"])], .38, .7, .09)

# ---------- step 3: guitar joins; a melody and its mirror ----------
chord_arp(["D3", "A3", "D4", "F4", "A4", "F4"], st("s6"), en("s6"), .36, .6, .1, .6, inst=lambda f, d, v: guitar(f, d, .45, int(f)))
for j, nm in enumerate(["A4", "C5", "E5"]):
    add(music, piano(hz(nm), 2.5, .7), c6["speak"] + .2 + j * .34, .09, .4)
for j, nm in enumerate(["E5", "C5", "A4"]):
    add(music, piano(hz(nm), 2.5, .7), c6["ask"] + .1 + j * .34, .09, -.4)
add(music, strings([hz("D3"), hz("A3"), hz("F4")], en("s6") - st("s6") + .5, attack=1.4, fc=1300), st("s6"), .28)
for k in range(2):
    add(sfx, chirp_bird(80 + k), st("s6") + 1.0 + k * 1.9, .03, (-.7, .7)[k])

# ---------- step 4: warmth fills in; a chime for each small repair ----------
piano_bars(st("s7"), en("s7"), [("Bb2", ["Bb3", "D4", "F4", "A4"]), ("C3", ["C4", "E4", "G4", "C5"]), ("F2", ["F3", "A3", "C4", "E4"])], .34, .8, .1)
add(music, strings([hz("Bb2"), hz("F3"), hz("D4"), hz("A4")], en("s7") - st("s7") + .6, attack=1.0, fc=1600), st("s7"), .4)
for (tc, nm) in ((c7["smile"], "C6"), (c7["touch"], "E6"), (c7["right"], "G6")):
    add(sfx, bell(hz(nm), 2.2, 1.4, 2.0, 2.0), tc + .32, .06)
    add(music, guitar(hz(nm) / 2, 1.8, .6, int(tc * 10)), tc + .34, .08)
for k in range(3):
    add(sfx, chirp_bird(90 + k), st("s7") + .6 + k * 1.4, .028, (-.6, .6, 0)[k])

# ---------- closing: full resolution in F ----------
cl = [("F2", ["F3", "A3", "C4", "E4", "A4"]), ("E2", ["C4", "E4", "G4", "C5"]), ("D2", ["D4", "F4", "A4", "C5"]), ("Bb1", ["Bb3", "D4", "F4", "A4"])]
piano_bars(st("s8"), c8["card"], cl, .34, .85, .1)
add(music, strings([hz("F2"), hz("C3"), hz("A3"), hz("E4"), hz("A4")], c8["card"] - st("s8") + 1.4, attack=1.4, fc=1900), st("s8"), .55)
for k, nm in enumerate(["C5", "F5", "A5", "C6", "E6", "F6", "A6"]):
    add(music, guitar(hz(nm), 2.0, .55, 100 + k), c8["heart"] + k * .07, .09, -.6 + k * .2)
for j, nm in enumerate(["F2", "C3", "F3", "A3", "C4", "E4", "A4"]):
    add(music, piano(hz(nm), 5.0, .8), c8["card"] - .1 + j * .03, .1, -.3 + j * .1)
add(sfx, bell(hz("C6"), 3.0, 1.0, 2.0, 1.2), c8["card"] + .3, .05)
add(sfx, bell(hz("F6"), 3.0, 1.0, 2.0, 1.2), c8["card"] + .45, .04, .4)
for k in range(2):
    add(sfx, chirp_bird(120 + k), st("s8") + 1.2 + k * 2.2, .025, (-.5, .5)[k])

# ---------- time-varying filtering: muffled under water, a short muffle when the wall rises ----------
MUF = np.zeros(N)
def env_window(a, b, ra, rb, depth):
    t = np.arange(N) / SR
    return depth * np.clip((t - a) / ra, 0, 1) * np.clip((b - t) / rb, 0, 1)
MUF = np.maximum(MUF, env_window(c2["flood"][0] + .3, st("s3") + .6, .6, .9, 1.0))
MUF = np.maximum(MUF, env_window(c1["quiet"], c1["quiet"] + .9, .08, .5, .65))
_lb, _la = signal.butter(4, 520 / (SR / 2))
def muffle(x):
    lp = signal.filtfilt(_lb, _la, x, axis=0)
    return x * (1 - MUF)[:, None] + lp * MUF[:, None] * 1.25

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
bed = muffle(bed); sfx = muffle(sfx)
mixL = bed * (1 - .6 * duck)[:, None] * .66 + sfx * (1 - .4 * duck)[:, None] * .85
mixL += stereo(vo, 0) * .62 + vo_room * .085
fade = np.clip((DUR - np.arange(N) / SR) / .6, 0, 1)
fade_in = np.clip(np.arange(N) / (.004 * SR), 0, 1)
mixL *= (fade * fade_in)[:, None]
pk = np.abs(mixL).max()
mixL = mixL / pk * .98
mixL = np.tanh(mixL * 1.25) / np.tanh(1.25) * .89
sf.write(out, mixL.astype(np.float32), SR, subtype="PCM_24")
vo_rms = 20 * np.log10(np.sqrt(np.mean((mixL[np.abs(vo) > .005]) ** 2)) + 1e-9)
print(f"wrote {out}: {N / SR:.2f}s, voice-section RMS {vo_rms:.1f} dBFS, peak {20*np.log10(np.abs(mixL).max()):.1f} dBFS")
