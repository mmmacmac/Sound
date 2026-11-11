import math
import os

import numpy as np
from PIL import Image, ImageDraw

SIZE = 1080
FPS = 30
DURATION = 10.0
FRAMES = int(FPS * DURATION)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")
os.makedirs(OUT_DIR, exist_ok=True)

CENTER = SIZE / 2
CIRCLE_R = 170
CIRCLE_W = 6
BAR_COUNT = 96
BAR_START = CIRCLE_R + 26
BAR_MAX = 190
BAR_MIN = 12
BAR_W = 7

rng = np.random.default_rng(7)
BAR_PHASE = rng.uniform(0, 2 * math.pi, BAR_COUNT)
BAR_FREQ = rng.uniform(0.6, 2.2, BAR_COUNT)
NOISE = rng.normal(0, 1, (FRAMES, BAR_COUNT))
NOISE = np.cumsum(NOISE, axis=0) / np.sqrt(np.arange(1, FRAMES + 1))[:, None]

BPM = 120
BEAT = 60.0 / BPM


def bar_signal(t):
    n = np.arange(BAR_COUNT)
    base = (
        0.45
        + 0.28 * np.sin(2 * math.pi * BAR_FREQ * n * t + BAR_PHASE)
        + 0.17 * np.sin(2 * math.pi * (BAR_FREQ * 2.7 + 0.3) * n * t + BAR_PHASE * 1.7)
    )
    beat_pos = (t % BEAT) / BEAT
    kick = np.exp(-beat_pos * 6.0) * np.clip(np.cos(2 * math.pi * n / BAR_COUNT * 3), -1, 1)
    bass = np.exp(-((n - BAR_COUNT / 2) ** 2) / (2 * (BAR_COUNT / 6) ** 2))
    f = int(t * FPS)
    noise = NOISE[min(f, FRAMES - 1)] * 0.12
    v = base + 0.35 * kick * (0.4 + bass) + noise
    return np.clip(v, 0.0, 1.0)


def draw_frame(t):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse(
        [CENTER - CIRCLE_R, CENTER - CIRCLE_R, CENTER + CIRCLE_R, CENTER + CIRCLE_R],
        outline=(255, 255, 255, 255),
        width=CIRCLE_W,
    )
    sig = bar_signal(t)
    for i in range(BAR_COUNT):
        ang = 2 * math.pi * i / BAR_COUNT - math.pi / 2
        length = BAR_MIN + sig[i] * (BAR_MAX - BAR_MIN)
        ca, sa = math.cos(ang), math.sin(ang)
        px, py = -sa, ca
        hw = BAR_W / 2
        r0, r1 = BAR_START, BAR_START + length
        pts = [
            (CENTER + r0 * ca - hw * px, CENTER + r0 * sa - hw * py),
            (CENTER + r1 * ca - hw * px, CENTER + r1 * sa - hw * py),
            (CENTER + r1 * ca + hw * px, CENTER + r1 * sa + hw * py),
            (CENTER + r0 * ca + hw * px, CENTER + r0 * sa + hw * py),
        ]
        d.polygon(pts, fill=(255, 255, 255, 255))
    return img


for f in range(FRAMES):
    t = f / FPS
    draw_frame(t).save(os.path.join(OUT_DIR, f"frame_{f:04d}.png"))
    if f % 30 == 0:
        print(f"frame {f}/{FRAMES}", flush=True)
print("done")
