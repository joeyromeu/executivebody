#!/usr/bin/env python3
"""Render the "Blade / 80% zone" reel.

Layers, back to front:
  1. background  - a photo plate (Ken Burns zoom, faded to black at the bottom), an animated
                   graphic, or plain black
  2. Joey        - the talking-head video, cut out with the RVM alpha matte so photos sit behind him
  3. overlays    - text labels, highlights and stamps that pop in on the spoken word

Usage:
  render.py video OUT.mp4            full render (1080x1920, 30fps, original audio)
  render.py stills OUTDIR t1 t2 ...  render single frames at the given seconds (for checking)

Paths are set by the env vars SRC (source video), ALPHA (matte video from matte.py),
ASSETS (folder with user/ and commons/full/ images).
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SRC = os.environ["SRC"]
ALPHA = os.environ["ALPHA"]
ASSETS = os.environ["ASSETS"]

W, H, FPS = 1080, 1920, 30
SW, SH = 720, 1280                 # source size
PW, PHH = 1080, 1350               # photo region (top of frame)
K = 1.1                            # plate oversize for Ken Burns headroom
FADE_START, FADE_END = 1020, 1350  # photo fades to black over Joey's shoulders

FONT_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
FONT_XB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 214, 0)
RED = (229, 31, 38)
GREEN = (38, 201, 106)
GREY = (110, 110, 110)
DARK = (28, 28, 28)

_fonts = {}


def font(size, path=FONT_BLACK):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


# ----------------------------------------------------------------------------- images

def load(rel):
    p = os.path.join(ASSETS, rel)
    return Image.open(p).convert("RGB") if os.path.exists(p) else None


def cover(img, w, h, cx=0.5, cy=0.5):
    s = max(w / img.width, h / img.height)
    im = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    x = min(max(0, round(im.width * cx - w / 2)), im.width - w)
    y = min(max(0, round(im.height * cy - h / 2)), im.height - h)
    return im.crop((x, y, x + w, y + h))


def fit(img, w, h):
    s = min(w / img.width, h / img.height)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


def rounded_mask(size, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], r, fill=255)
    return m


def darken(img, f):
    return Image.eval(img, lambda v: int(v * f))


def plate_cover(img, cx=0.5, cy=0.5, dim=1.0):
    p = cover(img, round(PW * K), round(PHH * K), cx, cy)
    return darken(p, dim) if dim < 1 else p


def plate_card(img, card_h, cy=0.45, dim_bg=0.35, sharpen=False):
    """Low-res photo: blurred, darkened fill behind a sharp rounded card."""
    pw, ph = round(PW * K), round(PHH * K)
    base = darken(cover(img, pw, ph).filter(ImageFilter.GaussianBlur(45)), dim_bg)
    card = fit(img, round(pw * 0.9), round(card_h * K))
    if sharpen:
        card = card.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    x, y = (pw - card.width) // 2, round(ph * cy - card.height / 2)
    shadow = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(shadow).rounded_rectangle([x + 10, y + 18, x + card.width + 10, y + card.height + 18], 34, fill=170)
    shadow = shadow.filter(ImageFilter.GaussianBlur(24))
    base = Image.composite(Image.new("RGB", (pw, ph), BLACK), base, shadow)
    base.paste(card, (x, y), rounded_mask(card.size, 30))
    return base


def plate_split(left, right, gap=12):
    pw, ph = round(PW * K), round(PHH * K)
    half = (pw - gap) // 2
    p = Image.new("RGB", (pw, ph), BLACK)
    p.paste(cover(left, half, ph), (0, 0))
    p.paste(cover(right, pw - half - gap, ph), (half + gap, 0))
    return p


# vertical fade of the photo region into black
_grad = np.ones((PHH,), np.float32)
_grad[FADE_START:FADE_END] = np.linspace(1, 0, FADE_END - FADE_START) ** 1.4
_grad[:170] *= np.linspace(0.72, 1, 170)  # slight darkening under the Reels header
FADE_MASK = Image.fromarray((np.repeat(_grad[:, None], PW, 1) * 255).astype(np.uint8), "L")
BLACK_REGION = Image.new("RGB", (PW, PHH), BLACK)


def ken_burns(plate, p, z0=1.0, z1=1.07):
    z = z0 + (z1 - z0) * p
    cw, ch = plate.width / z, plate.height / z
    x0, y0 = (plate.width - cw) / 2, (plate.height - ch) / 2
    return plate.resize((PW, PHH), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))


def dark_bg():
    g = np.linspace(26, 4, H).astype(np.uint8)
    return Image.fromarray(np.repeat(np.repeat(g[:, None], W, 1)[:, :, None], 3, 2), "RGB")


DARK_BG = dark_bg()


# ----------------------------------------------------------------------------- sprites

def text_sprite(text, size, fill=WHITE, box=None, stroke=0, stroke_fill=BLACK, pad=(34, 18),
                radius=20, path=FONT_BLACK, align="center", spacing=8, shadow=True):
    f = font(size, path)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = tmp.multiline_textbbox((0, 0), text, font=f, stroke_width=stroke, spacing=spacing, align=align)
    tw, th = r - l, b - t
    px, py = pad if box else (stroke + 6, stroke + 6)
    sw, sh = tw + 2 * px, th + 2 * py
    extra = 14 if shadow else 0
    im = Image.new("RGBA", (sw + extra, sh + extra), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if box:
        if shadow:
            sh_im = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh_im).rounded_rectangle([extra // 2, extra, sw + extra // 2 - 1, sh + extra - 1], radius, fill=(0, 0, 0, 150))
            im = Image.alpha_composite(im, sh_im.filter(ImageFilter.GaussianBlur(6)))
            d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, sw - 1, sh - 1], radius, fill=box)
    elif shadow:
        sh_im = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh_im).multiline_text((px - l + 4, py - t + 6), text, font=f, fill=(0, 0, 0, 200),
                                             stroke_width=stroke, stroke_fill=(0, 0, 0, 200), spacing=spacing, align=align)
        im = Image.alpha_composite(im, sh_im.filter(ImageFilter.GaussianBlur(5)))
        d = ImageDraw.Draw(im)
    d.multiline_text((px - l, py - t), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill,
                     spacing=spacing, align=align)
    return im


def emoji_sprite(ch, height):
    f = ImageFont.truetype(FONT_EMOJI, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((round(im.width * height / im.height), height), Image.LANCZOS)


def hstack(*sprites, gap=16):
    w = sum(s.width for s in sprites) + gap * (len(sprites) - 1)
    h = max(s.height for s in sprites)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for s in sprites:
        out.alpha_composite(s, (x, (h - s.height) // 2))
        x += s.width + gap
    return out


def x_stamp(size=520, width=58):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    m = width
    for a, b in (((m, m), (size - m, size - m)), ((size - m, m), (m, size - m))):
        d.line([a, b], fill=(0, 0, 0, 160), width=width + 18)
    for a, b in (((m, m), (size - m, size - m)), ((size - m, m), (m, size - m))):
        d.line([a, b], fill=RED + (255,), width=width)
    return im


def tick_icon(size, color=GREEN, width=16):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).line([(size * .18, size * .55), (size * .42, size * .78), (size * .84, size * .26)],
                            fill=color + (255,), width=width, joint="curve")
    return im


def cross_icon(size, color=RED, width=16):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    a, b = size * .22, size * .78
    d.line([(a, a), (b, b)], fill=color + (255,), width=width)
    d.line([(b, a), (a, b)], fill=color + (255,), width=width)
    return im


# ----------------------------------------------------------------------------- animation

def ease_out_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def ease_out(x):
    return 1 - (1 - x) ** 3


def clamp01(x):
    return max(0.0, min(1.0, x))


def paste_scaled(frame, spr, cx, cy, s=1.0, alpha=1.0, anchor="center"):
    if s <= 0.01 or alpha <= 0.01:
        return
    if abs(s - 1) > 0.005:
        spr = spr.resize((max(1, round(spr.width * s)), max(1, round(spr.height * s))), Image.BILINEAR)
    if alpha < 0.999:
        a = spr.getchannel("A").point(lambda v: int(v * alpha))
        spr = spr.copy()
        spr.putalpha(a)
    x = round(cx - spr.width / 2) if anchor == "center" else round(cx)
    y = round(cy - spr.height / 2)
    frame.paste(spr, (x, y), spr)


class Pop:
    """A sprite that pops in at t_on (and optionally out at t_off)."""

    def __init__(self, t_on, spr, cx, cy, t_off=None, anim="pop", anchor="center", pulses=()):
        self.t_on, self.spr, self.cx, self.cy, self.t_off = t_on, spr, cx, cy, t_off
        self.anim, self.anchor, self.pulses = anim, anchor, pulses

    def draw(self, frame, t):
        if t < self.t_on or (self.t_off is not None and t >= self.t_off):
            return
        dt = t - self.t_on
        s, a = 1.0, 1.0
        if self.anim == "pop":
            x = clamp01(dt / 0.24)
            s = 0.35 + 0.65 * ease_out_back(x)
            a = clamp01(dt / 0.08)
        elif self.anim == "stamp":
            x = clamp01(dt / 0.16)
            s = 1.9 - 0.9 * ease_out(x)
            a = clamp01(dt / 0.06)
        elif self.anim == "fade":
            a = clamp01(dt / 0.2)
        for tp in self.pulses:
            if 0 <= t - tp < 0.3:
                s *= 1 + 0.12 * math.sin(math.pi * (t - tp) / 0.3)
        paste_scaled(frame, self.spr, self.cx, self.cy, s, a, self.anchor)


class Wipe:
    """Highlighter: a coloured box that wipes in left-to-right behind text."""

    def __init__(self, t_on, text, size, cx, cy, box=YELLOW, fill=BLACK, dur=0.35):
        self.t_on, self.cx, self.cy, self.box, self.dur = t_on, cx, cy, box, dur
        self.txt = text_sprite(text, size, fill=fill, box=None, shadow=False)
        self.txt_white = text_sprite(text, size, fill=WHITE, box=None, shadow=True)
        self.pad = 26

    def draw(self, frame, t):
        if t < self.t_on - 0.4:
            return
        tw, th = self.txt.size
        x0, y0 = round(self.cx - tw / 2), round(self.cy - th / 2)
        if t < self.t_on:  # text visible in white before the highlight sweeps
            paste_scaled(frame, self.txt_white, self.cx, self.cy, 1, clamp01((t - self.t_on + 0.4) / 0.15))
            return
        p = ease_out(clamp01((t - self.t_on) / self.dur))
        bw = round((tw + 2 * self.pad) * p)
        if bw > 2:
            ImageDraw.Draw(frame).rounded_rectangle([x0 - self.pad, y0 - 8, x0 - self.pad + bw, y0 + th + 8], 14, fill=self.box)
        # text: black where highlighted, white elsewhere
        cut = x0 - self.pad + bw
        full_white = self.txt_white
        frame.paste(full_white, (x0 + (self.txt.width - full_white.width) // 2, y0 + (self.txt.height - full_white.height) // 2), full_white)
        if bw > 2:
            seg_w = max(0, min(tw, cut - x0))
            if seg_w > 0:
                seg = self.txt.crop((0, 0, seg_w, th))
                # erase the white text under the highlight, then draw black text
                ImageDraw.Draw(frame).rounded_rectangle([x0 - self.pad, y0 - 8, x0 - self.pad + bw, y0 + th + 8], 14, fill=self.box)
                frame.paste(seg, (x0, y0), seg)


# ----------------------------------------------------------------------------- graphics

def bar_row(frame, y, label, pct, frac, color, label_color=WHITE, glow=0.0, pct_text=None, alpha=1.0):
    """Horizontal percentage bar. frac = fill progress 0..1 of pct."""
    if alpha <= 0:
        return
    d = ImageDraw.Draw(frame)
    x0, x1, h = 90, 990, 104
    lab = text_sprite(label, 46, fill=label_color, path=FONT_XB, shadow=False)
    paste_scaled(frame, lab, x0 - 6, y - 92, 1, alpha, anchor="left")
    if glow > 0:
        g = round(10 * glow)
        d.rounded_rectangle([x0 - g, y - h // 2 - g, x1 + g, y + h // 2 + g], h // 2 + g, outline=color, width=max(2, g))
    d.rounded_rectangle([x0, y - h // 2, x1, y + h // 2], h // 2, fill=(40, 40, 40))
    fw = round((x1 - x0) * pct / 100 * frac)
    if fw > h:
        d.rounded_rectangle([x0, y - h // 2, x0 + fw, y + h // 2], h // 2, fill=color)
    if frac > 0.6:
        t = text_sprite(pct_text or f"{round(pct * frac)}%", 58, fill=WHITE if color != YELLOW else BLACK, shadow=False)
        paste_scaled(frame, t, x0 + max(fw, h) - t.width - 22, y, 1, clamp01((frac - 0.6) / 0.3), anchor="left")


def graphic_bars_60(frame, t):
    """6.30-9.38: Snipes 90% vs most people 60% (flickers)."""
    sn = ease_out(clamp01((t - 6.30) / 0.55))
    mp = ease_out(clamp01((t - 6.75) / 0.85))
    flick = 1.0
    if t > 8.1:
        flick = 1 - 0.10 * abs(math.sin((t - 8.1) * 9)) - (0.08 if int(t * 12) % 5 == 0 else 0)
    bar_row(frame, 600, "WESLEY SNIPES", 90, sn, YELLOW)
    bar_row(frame, 840, "MOST PEOPLE", 60, mp * flick, RED, pct_text="60%")


DAYS = ["M", "T", "W", "T", "F", "S", "S"]


def week_row(frame, t, y, marks, times):
    tile, gap = 124, 12
    x0 = (W - (7 * tile + 6 * gap)) // 2
    d = ImageDraw.Draw(frame)
    for i in range(7):
        x = x0 + i * (tile + gap)
        d.rounded_rectangle([x, y, x + tile, y + 160], 18, fill=(36, 36, 36))
        lab = text_sprite(DAYS[i], 38, fill=(170, 170, 170), path=FONT_XB, shadow=False)
        paste_scaled(frame, lab, x + tile / 2, y + 34)
        if t >= times[i]:
            icon = TICK if marks[i] else CROSS
            x_ = clamp01((t - times[i]) / 0.2)
            paste_scaled(frame, icon, x + tile / 2, y + 104, 0.4 + 0.6 * ease_out_back(x_), clamp01(x_ * 3))


def graphic_week(frame, t):
    """9.38-14.84: consistent 3-4 days, the rest goes out the window, not even every week."""
    week_row(frame, t, 470, [1, 1, 1, 1, 0, 0, 0], [9.38, 9.50, 9.62, 10.44, 11.64, 11.84, 12.02])
    if t >= 13.28:
        a = clamp01((t - 13.28) / 0.2)
        week_row(frame, t, 680, [1, 0, 1, 0, 0, 0, 0], [13.40, 13.52, 13.64, 13.76, 13.88, 14.00, 14.12]) if a > 0 else None


def graphic_bars_80(frame, t):
    """52.22-57.22: 60 red / 80 green (sweet spot) / 90-95 grey."""
    g = ease_out(clamp01((t - 53.38) / 0.6))
    r = ease_out(clamp01((t - 53.80) / 0.6))
    gr = ease_out(clamp01((t - 55.42) / 0.5))
    glow = clamp01((t - 55.6) / 0.3)
    bar_row(frame, 560, "SNIPES", 95, g, GREY, label_color=(180, 180, 180), pct_text="90-95%")
    bar_row(frame, 770, "YOUR TARGET", 80, gr, GREEN, glow=glow, label_color=GREEN if gr > 0 else WHITE, pct_text="80%")
    bar_row(frame, 980, "MOST PEOPLE", 60, r, RED, pct_text="60%")


def graphic_dark(frame, t):
    pass


# ----------------------------------------------------------------------------- timeline

class Item:
    def __init__(self, t0, t1, kind, plate=None, overlays=(), graphic=None, kb=(1.0, 1.07)):
        self.t0, self.t1, self.kind, self.plate, self.graphic = t0, t1, kind, plate, graphic
        self.overlays, self.kb = list(overlays), kb


def build():
    global TICK, CROSS
    TICK, CROSS = tick_icon(96), cross_icon(96)
    u = {k: load(f"user/{k}.png") for k in ["u1", "u2", "u3", "u4", "u5", "u6"]}
    c = {k: load(f"commons/full/{k}.jpg") for k in ["oats", "eggs", "chicken", "salmon", "bread", "fried", "deadlift", "mealprep"]}

    topless = u["u2"].crop((0, 0, 196, 350))           # 1998 half of the then/now image, labels cropped off
    martial = u["u6"].crop((0, 0, u["u6"].width, 800))  # drop the "publicity picture" watermark strip
    breakfast = plate_split(c["oats"], c["eggs"]) if c["eggs"] else plate_cover(c["oats"])
    lift = plate_cover(c["deadlift"], dim=0.42) if c["deadlift"] else plate_cover(u["u5"], dim=0.32)

    HEAD_Y = 290       # headline position (below the Reels header)
    TAG_Y = 1085       # label just above Joey's head

    def head(t, text, size=84, box=None, fill=WHITE, **kw):
        return Pop(t, text_sprite(text, size, fill=fill, box=box, stroke=0 if box else 7), W / 2, HEAD_Y, **kw)

    def tag(t, text, box=YELLOW, fill=BLACK, size=64, y=TAG_Y, **kw):
        return Pop(t, text_sprite(text, size, fill=fill, box=box), W / 2, y, **kw)

    def tag2(t, title, sub, box=YELLOW, fill=BLACK, y=TAG_Y, **kw):
        a = text_sprite(title, 64, fill=fill, box=box)
        b = text_sprite(sub, 40, fill=WHITE, path=FONT_XB, stroke=5)
        im = Image.new("RGBA", (max(a.width, b.width), a.height + b.height - 4), (0, 0, 0, 0))
        im.alpha_composite(a, ((im.width - a.width) // 2, 0))
        im.alpha_composite(b, ((im.width - b.width) // 2, a.height - 4))
        return Pop(t, im, W / 2, y - 30, **kw)

    def chip(t, text, y):
        return Pop(t, text_sprite(text, 50, fill=WHITE, box=(10, 10, 10, 215), pad=(28, 14), radius=16), 60, y, anchor="left")

    def meal_row(img, label):
        return img, text_sprite(label, 46, fill=BLACK, box=YELLOW, pad=(22, 10), radius=12)

    rows = [meal_row(cover(c["oats"], 960, 270), "BREAKFAST"),
            meal_row(cover(c["chicken"], 960, 270, cy=0.55), "LUNCH"),
            meal_row(cover(c["salmon"], 960, 270, cy=0.52), "DINNER")]

    def graphic_meals(frame, t):
        ys = [360, 646, 932]
        for i, ((img, lab), y, tin, tlab) in enumerate(zip(rows, ys, [77.48, 78.00, 78.38], [79.72, 79.98, 80.60])):
            if t < tin:
                continue
            x = ease_out(clamp01((t - tin) / 0.25))
            im = img if x >= 1 else img.resize((round(960 * (0.85 + 0.15 * x)), round(270 * (0.85 + 0.15 * x))), Image.BILINEAR)
            m = rounded_mask(im.size, 22)
            if x < 1:
                m = m.point(lambda v: int(v * x))
            frame.paste(im, ((W - im.width) // 2, y + (270 - im.height) // 2), m)
            if t >= tlab:
                paste_scaled(frame, lab, 60 + 18, y + 270 - 44, 0.4 + 0.6 * ease_out_back(clamp01((t - tlab) / 0.22)), anchor="left")

    month = text_sprite("1 MONTH", 132, fill=BLACK, box=YELLOW, pad=(40, 16))
    not6 = text_sprite("NOT 6", 76, fill=WHITE, box=(200, 20, 30), pad=(26, 12))
    not12 = text_sprite("NOT 12", 76, fill=WHITE, box=(200, 20, 30), pad=(26, 12))

    dm = text_sprite("DM me '321'", 96, fill=WHITE, box=None, shadow=False)
    banner_inner = hstack(dm, emoji_sprite("\U0001F447", 92), gap=18)
    banner = Image.new("RGBA", (banner_inner.width + 90, banner_inner.height + 54), (0, 0, 0, 0))
    bd = ImageDraw.Draw(banner)
    bd.rounded_rectangle([0, 0, banner.width - 1, banner.height - 1], 26, fill=RED)
    banner.alpha_composite(banner_inner, (45, 27))

    items = [
        # 1. hook - topless 1998
        Item(0.00, 3.82, "photo", plate_card(topless, 1060, cy=0.42, sharpen=True)),
        # 2. "to look like Wesley Snipes in Blade, you'd have to live like him"
        Item(3.82, 6.30, "photo", plate_cover(u["u3"], cx=0.52, cy=0.42),
             [tag(5.56, "LIVE LIKE HIM", box=WHITE)]),
        # 3. "most people live in the 60% zone"
        Item(6.30, 9.38, "graphic", graphic=graphic_bars_60,
             overlays=[head(7.10, "THE 60% ZONE", box=RED)]),
        # 4. "consistent 3, maybe 4 days a week... rest goes out the window... not even every week"
        Item(9.38, 14.84, "graphic", graphic=graphic_week,
             overlays=[head(9.38, "3-4 DAYS A WEEK", box=GREEN),
                       Pop(11.64, text_sprite("THE REST? OUT THE WINDOW", 56, fill=WHITE, stroke=6), W / 2, 1000, t_off=13.28),
                       Pop(13.28, text_sprite("...AND NOT EVEN EVERY WEEK", 56, fill=WHITE, stroke=6), W / 2, 1000)]),
        # 5. "at 61, Snipes got called to play Blade again"
        Item(14.84, 18.44, "photo", plate_cover(u["u4"], cx=0.41, cy=0.5),
             [tag(15.52, "AGE 61", size=96), head(16.86, "CALLED BACK AS BLADE", size=62, box=(15, 15, 15))]),
        # 6. "he admitted he was not Blade ready"
        Item(18.44, 20.50, "photo", plate_cover(u["u1"], cx=0.37, cy=0.5),
             [Wipe(19.34, "“I WASN'T BLADE READY”", 70, W / 2, HEAD_Y)]),
        # 7. "but it took him one month, not six, not twelve, one month"
        Item(20.50, 24.90, "photo", plate_cover(u["u5"], cx=0.5, cy=0.5),
             [Pop(21.78, month, W / 2, HEAD_Y, pulses=(23.70,)),
              Pop(22.32, not6, W / 2 - 150, HEAD_Y + 165), Pop(22.96, not12, W / 2 + 150, HEAD_Y + 165)]),
        # 8. "and that's because of his baseline"
        Item(24.90, 26.66, "photo", plate_card(u["u2"], 820, cy=0.47), [tag(25.94, "HIS BASELINE")], kb=(1.0, 1.03)),
        # 9. training every day
        Item(26.66, 32.26, "photo", plate_cover(martial, cx=0.5, cy=0.4),
             [head(26.94, "TRAINS ~EVERY DAY", size=72, box=WHITE, fill=BLACK),
              chip(28.12, "MARTIAL ARTS", 470), chip(28.76, "STRENGTH", 575), chip(29.10, "YOGA", 680),
              chip(29.50, "MOBILITY", 785), chip(30.66, "10K STEPS / DAY", 890)]),
        # 10. same meals on repeat - breakfast
        Item(32.26, 35.56, "photo", breakfast,
             [head(32.72, "SAME MEALS ON REPEAT", size=66, box=WHITE, fill=BLACK),
              tag2(33.86, "BREAKFAST", "oats + scrambled eggs")]),
        # 11. lunch
        Item(35.56, 37.02, "photo", plate_cover(c["chicken"], cy=0.55),
             [tag2(35.56, "LUNCH", "chicken, brown rice + broccoli")]),
        # 12. dinner -> omega 3
        Item(37.02, 40.58, "photo", plate_cover(c["salmon"], cy=0.5),
             [tag2(37.02, "DINNER", "salmon + sweet potato", t_off=38.72),
              tag(38.72, "HIGH OMEGA-3", box=GREEN, fill=WHITE, size=70)]),
        # 13. very low refined carbs
        Item(40.58, 43.98, "photo", plate_cover(c["bread"]),
             [tag(41.28, "REFINED CARBS", box=WHITE, fill=BLACK, size=70),
              Pop(42.80, x_stamp(), W / 2, 600, anim="stamp")]),
        # 14. nothing fried
        Item(43.98, 46.30, "photo", plate_cover(c["fried"]),
             [tag(44.66, "ANYTHING FRIED", box=WHITE, fill=BLACK, size=70),
              Pop(45.58, x_stamp(), W / 2, 600, anim="stamp")]),
        # 15. trust moment - Joey full screen
        Item(46.30, 52.22, "joey", overlays=[
            head(48.04, "TRAINING SINCE AGE 7", size=70, box=None),
            Pop(49.64, text_sprite("IT'S HIS JOB", 84, fill=BLACK, box=YELLOW), W / 2, HEAD_Y + 150)]),
        # 16. you don't need 90%, 80% is the sweet spot
        Item(52.22, 57.22, "graphic", graphic=graphic_bars_80,
             overlays=[head(52.50, "YOU DON'T NEED 90%", size=76, box=None),
                       Pop(56.56, text_sprite("80% = THE SWEET SPOT", 64, fill=WHITE, box=GREEN), W / 2, 1130)]),
        # 17. the 3-2-1 method
        Item(57.22, 69.30, "photo", lift,
             [head(58.86, "THE 3-2-1 METHOD", size=88, box=YELLOW, fill=BLACK),
              Pop(61.46, num_row("3", "STRENGTH SESSIONS\nPER WEEK"), 70, 520, anchor="left"),
              Pop(63.26, num_row("2", "SETS PER\nCOMPOUND LIFT"), 70, 720, anchor="left"),
              Pop(66.80, num_row("1", "CARDIO + MOBILITY\nSESSION PER WEEK"), 70, 920, anchor="left")],
             kb=(1.0, 1.10)),
        # 18. 1-3 hours a week
        Item(69.30, 72.42, "graphic", graphic=graphic_dark,
             overlays=[Pop(69.58, text_sprite("ALL YOUR TRAINING IN", 62, fill=WHITE, path=FONT_XB, shadow=False), W / 2, 430),
                       Pop(70.98, text_sprite("1-3 HOURS", 168, fill=YELLOW, shadow=False), W / 2, 620),
                       Pop(71.84, text_sprite("A WEEK", 96, fill=WHITE, shadow=False), W / 2, 800)]),
        # 19. protein every meal
        Item(72.42, 73.80, "photo", plate_cover(c["chicken"], cy=0.45),
             [head(72.58, "PROTEIN EVERY MEAL", size=74, box=YELLOW, fill=BLACK)]),
        # 20. same meals on repeat
        Item(73.80, 76.84, "photo", plate_cover(c["mealprep"]),
             [head(73.80, "SAME MEALS ON REPEAT", size=66, box=WHITE, fill=BLACK)]),
        # 21. go-to breakfasts, lunches, dinners
        Item(76.84, 82.32, "graphic", graphic=graphic_meals,
             overlays=[Pop(76.84, text_sprite("2-3 GO-TO MEALS", 76, fill=WHITE, stroke=7), W / 2, 250)]),
        # 22. CTA
        Item(82.32, 999, "joey", overlays=[
            Pop(82.32, banner, W / 2, 400, pulses=(89.34, 90.14, 90.52, 90.82)),
            Pop(82.86, text_sprite("FREE fat loss + training system", 50, fill=WHITE, path=FONT_XB, stroke=6), W / 2, 560),
            Pop(88.30, text_sprite("1-3 hours a week", 50, fill=YELLOW, path=FONT_XB, stroke=6), W / 2, 640)]),
    ]
    return items


def num_row(n, text):
    circ = Image.new("RGBA", (124, 124), (0, 0, 0, 0))
    ImageDraw.Draw(circ).ellipse([0, 0, 123, 123], fill=YELLOW)
    ns = text_sprite(n, 84, fill=BLACK, shadow=False)
    circ.alpha_composite(ns, ((124 - ns.width) // 2, (124 - ns.height) // 2))
    tx = text_sprite(text, 52, fill=WHITE, path=FONT_BLACK, stroke=5, align="left", spacing=4)
    return hstack(circ, tx, gap=26)


# ----------------------------------------------------------------------------- per-frame

JOEY_ZOOM = 1.45
JOEY_CX = 480  # source x of Joey's head centre (from the matte)


def joey_layers(fg, al, zoom):
    """Return Joey RGB + alpha at output size, optionally zoomed in (bottom-anchored on his head)."""
    if zoom <= 1.001:
        return fg.resize((W, H), Image.BICUBIC), al.resize((W, H), Image.BICUBIC)
    cw, ch = SW / zoom, SH / zoom
    x0 = min(max(0, JOEY_CX - cw / 2), SW - cw)
    y0 = SH - ch
    box = (x0, y0, x0 + cw, y0 + ch)
    return fg.resize((W, H), Image.BICUBIC, box=box), al.resize((W, H), Image.BICUBIC, box=box)


def render_frame(items, n, fg, al):
    t = n / FPS
    item = next(i for i in items if i.t0 <= t < i.t1)
    p = (t - item.t0) / max(0.01, min(item.t1, 95.6) - item.t0)

    if item.kind == "photo":
        frame = Image.new("RGB", (W, H), BLACK)
        ph = ken_burns(item.plate, p, *item.kb)
        frame.paste(Image.composite(ph, BLACK_REGION, FADE_MASK), (0, 0))
    elif item.kind == "graphic":
        frame = DARK_BG.copy()
        item.graphic(frame, t)
    else:
        frame = DARK_BG.copy()

    zoom = 1.0
    if item.kind == "joey":
        zoom = 1 + (JOEY_ZOOM - 1) * ease_out(clamp01((t - item.t0) / 0.22))
    jf, ja = joey_layers(fg, al, zoom)
    frame = Image.composite(jf, frame, ja)

    for o in item.overlays:
        o.draw(frame, t)
    return frame


def frames(ss=None, count=None):
    args = ["-ss", str(ss)] if ss is not None else []
    lim = ["-frames:v", str(count)] if count else []
    pv = subprocess.Popen(["ffmpeg", "-v", "error", *args, "-i", SRC, *lim, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    pa = subprocess.Popen(["ffmpeg", "-v", "error", *args, "-i", ALPHA, *lim, "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        bv = pv.stdout.read(SW * SH * 3)
        ba = pa.stdout.read(SW * SH)
        if len(bv) < SW * SH * 3 or len(ba) < SW * SH:
            break
        yield Image.frombuffer("RGB", (SW, SH), bv), Image.frombuffer("L", (SW, SH), ba)


def main():
    mode = sys.argv[1]
    items = build()
    if mode == "stills":
        outdir = sys.argv[2]
        os.makedirs(outdir, exist_ok=True)
        for ts in sys.argv[3:]:
            t = float(ts)
            n = round(t * FPS)
            fg, al = next(frames(ss=n / FPS, count=1))
            render_frame(items, n, fg, al).save(os.path.join(outdir, f"t{t:06.2f}.jpg"), quality=88)
        return
    out = sys.argv[2]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                            "-i", "-", "-i", SRC, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                            "-shortest", out], stdin=subprocess.PIPE)
    for n, (fg, al) in enumerate(frames()):
        enc.stdin.write(render_frame(items, n, fg, al).tobytes())
        if n % 150 == 0:
            print("frame", n, flush=True)
    enc.stdin.close()
    enc.wait()
    print("done", out)


if __name__ == "__main__":
    main()
