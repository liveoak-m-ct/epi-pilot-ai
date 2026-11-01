# -*- coding: utf-8 -*-
"""Epi-Pilot AI v2 の描画（白背景の動画・図）。epi_pilot.py --render から呼ばれる。

解剖図はこのプログラム内で描いた模式図（縮尺は層ごとに調整、実寸ではない）。
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from epi_pilot import (CLASS_JA, LAYER_JA, LAYER_KEYS, V_BASELINE, V_CAREFUL, V_FAST)

W, H = 1280, 720          # 論理座標（描画はこの座標系で書く）
S = 1.5                   # 出力倍率：1280×720 × 1.5 = 1920×1080（フルHD）
OW, OH = int(W * S), int(H * S)
FPS = 15

# 色（白背景）
BG = (255, 255, 255)
INK = (28, 36, 48)
MUTED = (104, 114, 128)
LINE = (214, 220, 228)
PANEL = (248, 250, 252)
TEAL = (0, 122, 140)
GREEN = (22, 148, 88)
ORANGE = (222, 124, 24)
RED = (204, 52, 58)
BLUE = (52, 110, 200)

LAYER_COLOR = {
    'epidermis': (226, 178, 150), 'dermis': (242, 208, 188), 'subcut': (250, 228, 150),
    'supra': (222, 210, 186), 'inter': (233, 222, 243), 'flavum': (240, 194, 68),
    'epidural': (255, 245, 212), 'dura': (64, 104, 196), 'arach': (150, 128, 208),
    'subarach': (212, 234, 250), 'cord': (250, 240, 196),
}
BONE = (241, 233, 210)
BONE_EDGE = (166, 146, 102)
BONE_DOT = (214, 198, 156)

# 解剖パネルの配置
AX0, AX1 = 64, 1244
TY0, TY1 = 142, 382
NY = 265
GAP = 44


# ---------------------------------------------------------------------------
# フォント
# ---------------------------------------------------------------------------
_FONT_REG = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
    '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
    '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc',
    'C:/Windows/Fonts/YuGothM.ttc', 'C:/Windows/Fonts/meiryo.ttc',
]
_FONT_BOLD = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc',
    '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
    '/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc',
    'C:/Windows/Fonts/YuGothB.ttc', 'C:/Windows/Fonts/meiryob.ttc',
]
_cache: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _cache:
        paths = (_FONT_BOLD + _FONT_REG) if bold else _FONT_REG
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            raise RuntimeError('日本語フォントが見つかりません。render.py の _FONT_REG にパスを追加してください。')
        _cache[key] = ImageFont.truetype(path, size)
    return _cache[key]


def text(dr, xy, s, size=16, color=INK, bold=False, anchor=None):
    dr.text(xy, str(s), size=size, bold=bold, fill=color, anchor=anchor)


def tw(s, size=16, bold=False) -> int:
    return int(font(size, bold).getlength(str(s)))


def mix(c, w=0.5):
    return tuple(int(255 - (255 - v) * w) for v in c)


def dashed(dr, x0, y0, x1, y1, color, width=2, dash=7):
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(L // (dash * 2)))
    for i in range(n + 1):
        a = i * 2 * dash / L
        b = min(1.0, (i * 2 + 1) * dash / L)
        if a >= 1:
            break
        dr.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                fill=color, width=width)


# ---------------------------------------------------------------------------
# 高解像度出力のための座標変換（論理座標 × S で実際に描く）
# ---------------------------------------------------------------------------
def _sc(v):
    return v * S


def _sc_xy(xy):
    if isinstance(xy, (list, tuple)) and xy and isinstance(xy[0], (list, tuple)):
        return [(_sc(a), _sc(b)) for a, b in xy]
    return [_sc(v) for v in xy]


def _sw(w):
    return max(1, int(round(w * S)))


class ScaledDraw:
    """ImageDraw を包み、論理座標・線幅・文字サイズを S 倍して描く。"""

    def __init__(self, img):
        self.d = ImageDraw.Draw(img)

    def text(self, xy, s, size=16, bold=False, fill=None, anchor=None):
        size = max(size, 13)   # 小さすぎる文字は13相当まで引き上げる
        self.d.text(_sc_xy(xy), s, font=font(int(round(size * S)), bold), fill=fill, anchor=anchor)

    def line(self, xy, fill=None, width=1, joint=None):
        self.d.line(_sc_xy(xy), fill=fill, width=_sw(width), joint=joint)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.d.rectangle(_sc_xy(xy), fill=fill, outline=outline, width=_sw(width))

    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(_sc_xy(xy), radius=_sc(radius), fill=fill, outline=outline, width=_sw(width))

    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.d.ellipse(_sc_xy(xy), fill=fill, outline=outline, width=_sw(width))

    def polygon(self, xy, fill=None, outline=None):
        self.d.polygon(_sc_xy(xy), fill=fill, outline=outline)


def new_img(w, h):
    return Image.new('RGB', (int(round(w * S)), int(round(h * S))), BG)


def paste(img, sub, xy):
    img.paste(sub, (int(round(xy[0] * S)), int(round(xy[1] * S))))


# ---------------------------------------------------------------------------
# 深さ → 画面x座標（層ごとに表示幅を調整）
# ---------------------------------------------------------------------------
class AnatomyMap:
    MIN_PX = {'epidermis': 6, 'dermis': 12, 'dura': 8, 'arach': 5}
    # 表示上の縮尺（シミュレーションの数値には影響しない）。皮下組織は幅を抑えて描く
    DISPLAY_SCALE = {'subcut': 0.4}

    def __init__(self, p, x0=AX0, x1=AX1):
        self.p = p
        total = x1 - x0
        dt = {k: p.thick[k] * self.DISPLAY_SCALE.get(k, 1.0) for k in LAYER_KEYS}
        fixed: dict = {}
        for _ in range(6):
            free = [k for k in LAYER_KEYS if k not in fixed]
            ppm = (total - sum(fixed.values())) / sum(dt[k] for k in free)
            changed = False
            for k in free:
                if k in self.MIN_PX and dt[k] * ppm < self.MIN_PX[k]:
                    fixed[k] = self.MIN_PX[k]
                    changed = True
            if not changed:
                break
        self.w = {k: fixed.get(k, dt[k] * ppm) for k in LAYER_KEYS}
        self.xs, x = {}, float(x0)
        for k in LAYER_KEYS:
            self.xs[k] = x
            x += self.w[k]
        self.xe = {k: self.xs[k] + self.w[k] for k in LAYER_KEYS}

    def x(self, depth: float) -> float:
        p = self.p
        for k in LAYER_KEYS:
            s, e = p.bounds[k], p.bounds[k] + p.thick[k]
            if depth < e or k == LAYER_KEYS[-1]:
                frac = min(max((depth - s) / p.thick[k], 0.0), 1.0)
                return self.xs[k] + frac * self.w[k]
        return self.xe[LAYER_KEYS[-1]]


# ---------------------------------------------------------------------------
# 解剖模式図（静的部分）
# ---------------------------------------------------------------------------
def draw_anatomy(img: Image.Image, p, am: AnatomyMap, labels: bool = True):
    dr = ScaledDraw(img)
    rng = np.random.default_rng(p.pid)
    for k in LAYER_KEYS:
        x0, x1 = am.xs[k], am.xe[k]
        dr.rectangle((x0, TY0, x1, TY1), fill=LAYER_COLOR[k])
    # 質感
    sx0, sx1 = am.xs['subcut'], am.xe['subcut']
    for yy in range(TY0 + 8, TY1, 22):
        for xx in np.arange(sx0 + 9, sx1 - 6, 20):
            jx, jy = rng.uniform(-4, 4), rng.uniform(-4, 4)
            r = rng.uniform(7, 10)
            dr.ellipse((xx + jx - r, yy + jy - r, xx + jx + r, yy + jy + r), outline=(232, 200, 104), width=2)
    ix0, ix1 = int(am.xs['inter']), int(am.xe['inter'])
    sub = new_img(ix1 - ix0, TY1 - TY0)
    ScaledDraw(sub).rectangle((0, 0, ix1 - ix0, TY1 - TY0), fill=LAYER_COLOR['inter'])
    sd = ScaledDraw(sub)
    for off in range(-260, ix1 - ix0 + 10, 13):
        sd.line([(off, TY1 - TY0), (off + 250, 0)], fill=(206, 188, 228), width=2)
    paste(img, sub, (ix0, TY0))
    sup0, sup1 = am.xs['supra'], am.xe['supra']
    for yy in range(TY0 + 4, TY1, 9):
        dr.line([(sup0 + 2, yy), (sup1 - 2, yy + 3)], fill=(200, 186, 156), width=1)
    fx0, fx1 = am.xs['flavum'], am.xe['flavum']
    for xx in np.arange(fx0 + 4, fx1 - 2, 6):
        dr.line([(xx, TY0), (xx, TY1)], fill=(222, 168, 40), width=1)
    ex0, ex1 = am.xs['epidural'], am.xe['epidural']
    for yy in range(TY0 + 18, TY1 - 10, 46):
        r = min(10, (ex1 - ex0) / 2 - 3)
        cx = (ex0 + ex1) / 2 + rng.uniform(-2, 2)
        dr.ellipse((cx - r, yy - r * 1.4, cx + r, yy + r * 1.4), outline=(238, 220, 160), width=2)
    cx0, cx1 = am.xs['cord'], am.xe['cord']
    for xx in np.arange(cx0 + 10, cx1 - 4, 11):
        dr.line([(xx, TY0), (xx + rng.uniform(-3, 3), TY1)], fill=(232, 214, 150), width=2)
    dr.line([(am.xs['dermis'], TY0), (am.xs['dermis'], TY1)], fill=(206, 160, 134), width=1)

    # 棘突起（上位・下位）。針は2つの棘突起の間を通る
    bx0, bx1 = am.xs['inter'] - 3, am.xe['flavum']
    for (y0, y1, name) in ((TY0 + 4, NY - GAP, '棘突起（上位椎）'), (NY + GAP, TY1 - 4, '棘突起（下位椎）')):
        dr.rounded_rectangle((bx0, y0, bx1, y1), radius=22, fill=BONE, outline=BONE_EDGE, width=3)
        for _ in range(int((bx1 - bx0) * (y1 - y0) / 260)):
            xx, yy = rng.uniform(bx0 + 10, bx1 - 10), rng.uniform(y0 + 8, y1 - 8)
            r = rng.uniform(2, 4.5)
            dr.ellipse((xx - r, yy - r, xx + r, yy + r), outline=BONE_DOT, width=1)
        cx = (bx0 + bx1) / 2
        w = tw(name, 15, True) + 16
        dr.rounded_rectangle((cx - w / 2, (y0 + y1) / 2 - 13, cx + w / 2, (y0 + y1) / 2 + 13), radius=8,
                             fill=(255, 255, 255), outline=BONE_EDGE)
        text(dr, (cx, (y0 + y1) / 2), name, 15, INK, True, anchor='mm')

    dr.rectangle((am.xs['epidermis'], TY0, am.xe['cord'], TY1), outline=(180, 188, 198), width=1)
    text(dr, (8, TY0 - 2), '↑ 頭側', 13, MUTED)
    text(dr, (am.xs['epidermis'], TY1 + 2), '← 皮膚側（背側）', 12, MUTED)
    text(dr, (am.xe['cord'], TY1 + 2), '腹側 →', 12, MUTED, anchor='ra')

    if labels:
        draw_layer_labels(dr, am)


def draw_layer_labels(dr, am: AnatomyMap):
    rows = [76, 104]
    last_right = {0: -1e9, 1: -1e9}
    for k in LAYER_KEYS:
        cx = (am.xs[k] + am.xe[k]) / 2
        label = LAYER_JA[k]
        w = tw(label, 15, True) + 10
        placed = None
        for r in (0, 1):
            x0 = max(cx - w / 2, last_right[r] + 6)
            if x0 - (cx - w / 2) < 40:
                placed = (r, x0)
                break
        if placed is None:
            r = 0 if last_right[0] < last_right[1] else 1
            placed = (r, max(cx - w / 2, last_right[r] + 6))
        r, x0 = placed
        x0 = min(x0, AX1 - w)
        last_right[r] = x0 + w
        y = rows[r]
        col = LAYER_COLOR[k]
        dr.rounded_rectangle((x0, y, x0 + w, y + 22), radius=6, fill=mix(col, 0.45),
                             outline=tuple(max(0, v - 60) for v in col))
        text(dr, (x0 + w / 2, y + 11), label, 15, INK, True, anchor='mm')
        ty = TY0 + 6
        if k in ('inter', 'flavum'):
            ty = NY - GAP + 6
        dr.line([(x0 + w / 2, y + 22), (cx, ty)], fill=(90, 98, 110), width=1)
        dr.ellipse((cx - 2.5, ty - 2.5, cx + 2.5, ty + 2.5), fill=(90, 98, 110))


# ---------------------------------------------------------------------------
# 針・シリンジなど（動的部分）
# ---------------------------------------------------------------------------
def draw_needle(dr, tip_x: float, color):
    x_tip = tip_x
    dr.rectangle((0, NY - 3, x_tip - 7, NY + 3), fill=(168, 176, 186), outline=(110, 118, 130))
    # Tuohy針：先端がわずかに頭側（上）へ曲がっている
    dr.polygon([(x_tip - 8, NY - 3), (x_tip - 1, NY - 7), (x_tip + 1, NY - 4), (x_tip - 6, NY + 3)],
               fill=(150, 158, 170), outline=(100, 108, 120))
    dr.ellipse((x_tip - 12, NY - 13, x_tip + 10, NY + 9), outline=color, width=3)


def draw_bleb(dr, am, tip_x, dp, region):
    if dp <= 0.002:
        return
    if region == 'epidural':
        half_w = max(4.0, (am.xe['epidural'] - am.xs['epidural']) / 2 - 2)
        cx = (am.xs['epidural'] + am.xe['epidural']) / 2
        ry = min(TY1 - NY - 6, 14 + dp * 260)
        rx = min(half_w, 4 + dp * 80)
    else:
        cx, rx, ry = tip_x + 2, min(10, 3 + dp * 60), min(GAP - 6, 6 + dp * 120)
    dr.ellipse((cx - rx, NY - ry, cx + rx, NY + ry), fill=(178, 216, 246), outline=(110, 170, 225), width=2)


def draw_hands(dr, x0, y0, x1, y1, rec, i, attached):
    dr.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=PANEL, outline=LINE, width=2)
    text(dr, (x0 + 14, y0 + 10), '術者の両手 → ロボットの2つの入力', 16, INK, True)
    # 左手の役割
    by = y0 + 46
    dr.rounded_rectangle((x0 + 14, by, x0 + 96, by + 40), radius=8, fill=(222, 240, 244), outline=TEAL, width=2)
    text(dr, (x0 + 55, by + 20), '駆動', 15, TEAL, True, anchor='mm')
    dr.line([(x0 + 96, by + 20), (x0 + 128, by + 20)], fill=TEAL, width=3)
    dr.polygon([(x0 + 128, by + 13), (x0 + 140, by + 20), (x0 + 128, by + 27)], fill=TEAL)
    text(dr, (x0 + 148, by + 2), '左手の役割：針を前進', 14, INK, True)
    text(dr, (x0 + 148, by + 21), '→ 抵抗を計測（信号①）', 14, MUTED)
    # 右手の役割：シリンジ
    sy = y0 + 128
    text(dr, (x0 + 14, sy - 26), '右手親指の役割：ガラスシリンジ（生理食塩水）', 14, INK, True)
    bx0, bx1 = x0 + 92, x0 + 300
    stroke = bx1 - bx0 - 26
    disp = min(rec['plunger'][i] / 0.9, 1.0) if attached else 0.0
    head = bx0 + 10 + disp * stroke * 0.85
    dr.rounded_rectangle((bx0, sy, bx1, sy + 40), radius=8, fill=(250, 252, 255), outline=(120, 140, 160), width=2)
    if attached:
        dr.rectangle((head + 8, sy + 4, bx1 - 4, sy + 36), fill=(196, 226, 248))
    dr.rectangle((bx1, sy + 14, bx1 + 22, sy + 26), fill=(170, 180, 190))
    dr.rectangle((head, sy + 3, head + 8, sy + 37), fill=(90, 100, 112))
    dr.line([(x0 + 40, sy + 20), (head, sy + 20)], fill=(90, 100, 112), width=6)
    col = ORANGE if attached else (170, 176, 184)
    dr.polygon([(x0 + 14, sy + 8), (x0 + 36, sy + 20), (x0 + 14, sy + 32)], fill=col)
    text(dr, (x0 + 14, sy + 48), '一定の圧で押し続ける → 内圧を計測（信号②）', 14, MUTED)
    if not attached:
        text(dr, (x0 + 14, sy + 70), '未接続：棘間靱帯に入ってから接続', 14, ORANGE, True)
    else:
        moved = rec['plunger'][i]
        msg = 'プランジャーが進む＝液が入る（抵抗消失）' if rec['open'][i] > 0.3 else '抵抗あり：プランジャーは止まったまま'
        text(dr, (x0 + 14, sy + 70), msg, 14, BLUE if rec['open'][i] > 0.3 else INK, True)


class PlotBG:
    """層ごとに色分けした時系列グラフの背景（症例ごとに1回だけ作る）。"""

    def __init__(self, rec, rect, t_max, attach_t=None):
        self.rect = rect
        x0, y0, x1, y1 = rect
        self.t_max = t_max
        self.img = new_img(x1 - x0, y1 - y0)
        d = ScaledDraw(self.img)
        t, lay = rec['t'], rec['layer']
        start = 0
        for j in range(1, len(t) + 1):
            if j == len(t) or lay[j] != lay[start]:
                a = int(t[start] / t_max * (x1 - x0))
                b = int(t[j - 1] / t_max * (x1 - x0))
                d.rectangle((a, 0, b + 1, y1 - y0), fill=mix(LAYER_COLOR[lay[start]], 0.32))
                if b - a > 46:
                    text(d, ((a + b) / 2, 4), LAYER_JA[lay[start]], 12, MUTED, anchor='ma')
                start = j
        if attach_t is not None:
            a = int(attach_t / t_max * (x1 - x0))
            d.rectangle((0, 18, a, y1 - y0), fill=(238, 240, 243))
            if a > 70:
                text(d, (a / 2, (y1 - y0) / 2 + 6), 'シリンジ未接続', 13, MUTED, anchor='mm')

    def paste(self, img, t_now):
        x0, y0, x1, y1 = self.rect
        w = max(1, min(x1 - x0, int(t_now / self.t_max * (x1 - x0))))
        paste(img, self.img.crop((0, 0, int(round(w * S)), int(round((y1 - y0) * S)))), (x0, y0))


def draw_series(dr, rect, t, v, i, t_max, vmax, color, vmin=0.0):
    x0, y0, x1, y1 = rect
    xs = x0 + t[:i + 1] / t_max * (x1 - x0)
    vv = v[:i + 1]
    ok = ~np.isnan(vv)
    pts = [(float(a), float(y1 - 3 - (min(max(b, vmin), vmax) - vmin) / (vmax - vmin) * (y1 - y0 - 22)))
           for a, b in zip(xs[ok], vv[ok])]
    if len(pts) > 1:
        dr.line(pts, fill=color, width=2)
    dr.rectangle(rect, outline=(190, 198, 208), width=1)


# ---------------------------------------------------------------------------
# 1フレームの描画
# ---------------------------------------------------------------------------
class CaseView:
    def __init__(self, case):
        self.case = case
        ep = case['ai']
        self.rec = ep.rec
        self.p = case['patient']
        self.am = AnatomyMap(self.p)
        self.base = new_img(W, H)
        draw_anatomy(self.base, self.p, self.am)
        self.t_max = float(self.rec['t'][-1]) + 0.4
        att = np.where(~np.isnan(self.rec['pressure']))[0]
        self.attach_t = float(self.rec['t'][att[0]]) if len(att) else None
        self.rect_f = (408, 444, 892, 552)
        self.rect_p = (408, 590, 892, 700)
        self.bg_f = PlotBG(self.rec, self.rect_f, self.t_max)
        self.bg_p = PlotBG(self.rec, self.rect_p, self.t_max, self.attach_t)
        self.fmax = float(np.nanmax(self.rec['force'])) * 1.12
        pr = self.rec['pressure']
        self.p_plateau = float(np.nanmedian(pr[att[0] + 30: att[0] + 130])) if len(att) else 1.0
        self.th = case['th']
        self.theta = case.get('theta', 0.3)

    def index_at(self, t):
        return int(min(len(self.rec['t']) - 1, max(0, np.searchsorted(self.rec['t'], t))))

    def frame(self, t, header_tag, show_banner=False, show_result=False):
        rec, p, am = self.rec, self.p, self.am
        i = self.index_at(t)
        img = self.base.copy()
        dr = ScaledDraw(img)
        draw_header(dr, header_tag)
        depth = float(rec['depth'][i])
        tip_x = am.x(depth)
        stopped = rec['state'][i] == 'stop' and abs(rec['vel'][i]) < 0.02
        # 生理食塩水の広がり
        op = rec['open'][:i + 1]
        if op[-1] > 0:
            j = i
            while j > 0 and op[j - 1] > 0:
                j -= 1
            region = 'epidural' if depth >= p.epi_start else 'false'
            draw_bleb(dr, am, tip_x, float(rec['plunger'][i] - rec['plunger'][j]), region)
        # 比較法（圧閾値法）の停止位置
        if self.th.outcome == 'early' and depth >= self.th.stop_depth:
            xg = am.x(self.th.stop_depth)
            dashed(dr, xg, NY - GAP + 2, xg, NY + GAP - 2, RED, 3)
            lab = '従来の圧閾値法はここで停止（偽LOR）'
            text(dr, (xg + 6, NY + GAP - 22), lab, 13, RED, True)
        if show_result and self.th.outcome == 'dura':
            xd = am.xs['dura']
            dashed(dr, xd, TY0 + 2, xd, TY1 - 2, RED, 3)
            lab = '従来の圧閾値法：止まらず硬膜に到達'
            text(dr, (xd + 10, NY + GAP + 6), lab, 13, RED, True)
        draw_needle(dr, tip_x, GREEN if stopped else ORANGE)
        if stopped:
            text(dr, (tip_x + 4, NY - GAP + 4), 'STOP', 18, GREEN, True)

        attached = not np.isnan(rec['pressure'][i])
        draw_hands(dr, 16, 404, 392, 710, rec, i, attached)

        # グラフ
        dr.rounded_rectangle((400, 404, 900, 710), radius=14, fill=BG, outline=LINE, width=2)
        text(dr, (412, 412), '信号① 針を進める抵抗（左手の感覚・任意単位）', 14, INK, True)
        text(dr, (412, 558), '信号② シリンジ内圧（右手親指の感覚・任意単位）', 14, INK, True)
        self.bg_f.paste(img, rec['t'][i])
        self.bg_p.paste(img, rec['t'][i])
        draw_series(dr, self.rect_f, rec['t'], rec['force'], i, self.t_max, self.fmax, ORANGE)
        draw_series(dr, self.rect_p, rec['t'], rec['pressure'], i, self.t_max, 1.35, BLUE)
        # 圧閾値法の閾値線
        x0, y0, x1, y1 = self.rect_p
        yth = y1 - 3 - (self.theta * self.p_plateau) / 1.35 * (y1 - y0 - 22)
        dashed(dr, x0, yth, x1, yth, (200, 120, 120), 1, 5)
        text(dr, (x1 - 4, yth - 15), f'従来法の閾値（基準圧×{self.theta:.2f}）', 11, (190, 100, 100), anchor='ra')
        text(dr, (x1, y1 + 2), '時間 →', 11, MUTED, anchor='ra')

        draw_ai_panel(dr, 908, 404, 1264, 710, rec, i, depth, attached)

        if show_banner:
            banner(dr, self.case['title'], '実時間（1倍速）で表示')
        if show_result:
            result_card(dr, self.case, self.th)
        return img


def draw_header(dr, tag):
    text(dr, (22, 8), 'Epi-Pilot AI', 30, TEAL, True)
    text(dr, (232, 18), '硬膜外穿刺の抵抗消失（LOR）をAIが判定し、1軸ロボットが自動停止する', 16, INK)
    if tag:
        w = tw(tag, 14, True) + 20
        dr.rounded_rectangle((W - 22 - w, 8, W - 22, 36), radius=10, fill=(232, 244, 246), outline=TEAL)
        text(dr, (W - 22 - w / 2, 22), tag, 14, TEAL, True, anchor='mm')
    text(dr, (W - 22, 41), '合成データによるシミュレーション｜医療用ではありません', 11, RED, anchor='ra')
    dr.line([(0, 60), (W, 60)], fill=TEAL, width=3)


def draw_ai_panel(dr, x0, y0, x1, y1, rec, i, depth, attached):
    dr.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=PANEL, outline=LINE, width=2)
    text(dr, (x0 + 14, y0 + 10), 'AI判定（10 Hz・RandomForest）', 16, INK, True)
    probs = [rec['p0'][i], rec['p1'][i], rec['p2'][i]] if attached else [None, None, None]
    cols = [MUTED, ORANGE, GREEN]
    for k in range(3):
        yy = y0 + 42 + k * 44
        text(dr, (x0 + 14, yy), CLASS_JA[k], 13, INK)
        bx0, bx1 = x0 + 14, x1 - 62
        dr.rounded_rectangle((bx0, yy + 19, bx1, yy + 33), radius=6, fill=(230, 234, 238))
        if probs[k] is not None:
            w = (bx1 - bx0) * float(probs[k])
            if w > 4:
                dr.rounded_rectangle((bx0, yy + 19, bx0 + w, yy + 33), radius=6, fill=cols[k])
            text(dr, (x1 - 14, yy + 26), f'{float(probs[k]) * 100:3.0f}%', 14, INK, True, anchor='rm')
        else:
            text(dr, (x1 - 14, yy + 26), '—', 14, MUTED, anchor='rm')
    st = rec['state'][i]
    v = rec['vcmd'][i]
    if st == 'stop':
        msg, col, sub = '停止：硬膜外腔を判定', GREEN, '速度指令 0 mm/s'
    elif st == 'careful':
        msg, col, sub = '慎重前進：黄色靱帯を検知', ORANGE, f'速度指令 {v:.1f} mm/s（減速）'
    else:
        msg, col, sub = '前進中', TEAL, f'速度指令 {v:.1f} mm/s'
    by = y0 + 178
    dr.rounded_rectangle((x0 + 14, by, x1 - 14, by + 56), radius=10, fill=mix(col, 0.14), outline=col, width=2)
    text(dr, ((x0 + x1) / 2, by + 18), msg, 17, col, True, anchor='mm')
    text(dr, ((x0 + x1) / 2, by + 41), sub, 13, INK, anchor='mm')
    text(dr, (x0 + 14, by + 66), f'針先の深さ {depth:5.1f} mm（表示用の模擬値）', 13, MUTED)
    text(dr, (x0 + 14, by + 86), '※AIは深さ・位置の情報を使わない', 12, MUTED)


def banner(dr, title, sub):
    w = max(tw(title, 24, True), tw(sub, 15)) + 60
    x0 = (W - w) / 2
    dr.rounded_rectangle((x0, 300, x0 + w, 372), radius=14, fill=(255, 255, 255), outline=TEAL, width=3)
    text(dr, (W / 2, 324), title, 24, TEAL, True, anchor='mm')
    text(dr, (W / 2, 354), sub, 15, MUTED, anchor='mm')


def result_card(dr, case, th):
    ep = case['ai']
    x0, y0, x1, y1 = 120, 448, 1160, 676
    dr.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(255, 255, 255), outline=GREEN, width=3)
    text(dr, (x0 + 26, y0 + 18), f'{case["title"]} ─ 結果', 20, INK, True)
    ok = ep.outcome == 'success'
    text(dr, (x0 + 26, y0 + 62), 'Epi-Pilot AI', 18, TEAL, True)
    if ok:
        msg = f'硬膜外腔で停止（黄色靱帯を抜けてから {ep.overshoot:.1f} mm、硬膜まで {ep.margin:.1f} mm の余裕）'
    else:
        msg = {'early': '早期停止', 'dura': '硬膜に到達'}[ep.outcome]
    text(dr, (x0 + 200, y0 + 62), msg, 18, GREEN if ok else RED, True)
    text(dr, (x0 + 26, y0 + 104), '従来の圧閾値法', 18, MUTED, True)
    if th.outcome == 'success':
        m2 = f'硬膜外腔で停止（進入 {th.overshoot:.1f} mm、硬膜まで {th.margin:.1f} mm）'
        c2 = GREEN
    elif th.outcome == 'early':
        m2 = '棘間靱帯の偽LORで早期停止（硬膜外腔に届かない）'
        c2 = RED
    else:
        m2 = '抵抗消失が弱く閾値を下回らず、止まらないまま硬膜に到達'
        c2 = RED
    text(dr, (x0 + 200, y0 + 104), m2, 18, c2, True)
    notes = {
        'A': '抵抗消失がはっきりしている例では、どちらの方法でも止まる。',
        'B': 'AIは「直前にプツプツ（黄色靱帯）があったか」と「抵抗の急低下」を合わせて見るため、偽LORでは止まらない。',
        'C': '圧の下がり方が小さくても、黄色靱帯のプツプツ→抵抗の急低下→プランジャーの前進という流れから判定できる。',
    }
    text(dr, (x0 + 26, y0 + 152), notes[case['key']], 15, INK)
    text(dr, (x0 + 26, y0 + 186), '数値はすべて合成データ上のもので、臨床性能を示すものではありません。', 13, MUTED)


# ---------------------------------------------------------------------------
# 静止画シーン
# ---------------------------------------------------------------------------
def title_scene(cv: CaseView) -> Image.Image:
    img = new_img(W, H)
    dr = ScaledDraw(img)
    dr.rectangle((0, 0, W, 8), fill=TEAL)
    text(dr, (80, 70), 'Epi-Pilot AI', 64, TEAL, True)
    text(dr, (80, 160), '硬膜外穿刺の「抵抗消失（LOR）」をAIが判定し、', 28, INK, True)
    text(dr, (80, 202), '1軸ロボットが針を自動で止めるシミュレーション', 28, INK, True)
    text(dr, (80, 262), 'Physical AI 応用1 最終課題 ｜ 仮想患者100人での比較評価', 18, MUTED)
    strip = cv.base.crop((0, int(66 * S), OW, int(392 * S))).resize((int(960 * S), int(245 * S)), Image.LANCZOS)
    paste(img, strip, (160, 320))
    ScaledDraw(img).rectangle((160, 320, 1120, 565), outline=LINE, width=2)
    text(dr, (W / 2, 600), '実在の患者・実測データ・医療機器は使用していません。教育用の合成データによるシミュレーションです。',
         16, RED, True, anchor='mm')
    return img


def background_scene(cv: CaseView) -> Image.Image:
    img = cv.base.copy()
    dr = ScaledDraw(img)
    draw_header(dr, '背景：硬膜外麻酔と抵抗消失法')
    draw_needle(dr, cv.am.x(0.5), ORANGE)
    x0, y0, x1, y1 = 16, 404, 1264, 710
    dr.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=PANEL, outline=LINE, width=2)
    lines = [
        ('硬膜外麻酔は、無痛分娩・手術麻酔・術後鎮痛に欠かせない手技。', INK, True),
        ('針は 皮膚 → 皮下組織 → 棘上靱帯 → 棘間靱帯（上下の棘突起の間）→ 黄色靱帯 → 硬膜外腔 の順に進む。', INK, False),
        ('左手：Tuohy針を進める。黄色靱帯では線維を切るような「プツプツ」という手応えがある。', ORANGE, True),
        ('右手親指：生理食塩水を入れたガラスシリンジを一定の圧で押し続ける。針先が硬膜外腔に入ると', BLUE, True),
        ('　　　　　　抵抗がふっと消えて液が入る ＝ 抵抗消失（Loss of Resistance, LOR）。', BLUE, True),
        ('熟練には数年かかる。本作品は、この2つの感覚を2つのセンサー信号に置き換え、AIが判定してロボットを止める。', TEAL, True),
    ]
    for k, (s, c, b) in enumerate(lines):
        text(dr, (40, y0 + 22 + k * 46), s, 18, c, b)
    return img


def results_scene(summary: dict) -> Image.Image:
    img = new_img(W, H)
    dr = ScaledDraw(img)
    draw_header(dr, '結果：仮想患者100人')
    draw_results(dr, summary, 40, 80)
    return img


def draw_results(dr, summary, ox, oy):
    s = summary['closed_loop_test']
    pop = summary['test_population']
    text(dr, (ox, oy), '学習に使っていない仮想患者100人で、同じ条件の穿刺を比較', 22, INK, True)
    text(dr, (ox, oy + 36),
         f'皮膚〜硬膜外腔 {pop["skin_to_epidural_mm_range"][0]}〜{pop["skin_to_epidural_mm_range"][1]} mm、'
         f'偽LORあり {pop["with_false_lor"]}人、抵抗消失が弱い {pop["weak_lor"]}人（評価のため臨床より意図的に多く設定した模擬値）', 15, MUTED)
    rows = [('Epi-Pilot AI', s['ai']), (f'従来の圧閾値法（閾値 基準圧×{summary["threshold_method"]["theta"]:.2f}）', s['threshold'])]
    bx0, bx1 = ox, ox + 760
    for k, (name, r) in enumerate(rows):
        y = oy + 90 + k * 96
        text(dr, (bx0, y), name, 18, TEAL if k == 0 else MUTED, True)
        x = bx0
        for key, col, lab in (('success', GREEN, '成功'), ('early_stop', ORANGE, '早期停止'), ('dura_reached', RED, '硬膜到達')):
            n = r[key]
            w = (bx1 - bx0) * n / r['n']
            if n:
                dr.rectangle((x, y + 30, x + w, y + 70), fill=col)
                if w > 70:
                    text(dr, (x + w / 2, y + 50), f'{lab} {n}人', 16, (255, 255, 255), True, anchor='mm')
                else:
                    text(dr, (x + w + 4, y + 50), f'{lab} {n}', 13, col, True, anchor='lm')
            x += w
    # 凡例
    ly = oy + 290
    for k, (col, lab) in enumerate([(GREEN, '成功：硬膜外腔で停止'), (ORANGE, '早期停止：硬膜外腔に届かず'),
                                     (RED, '硬膜到達：硬膜穿刺に相当')]):
        dr.rectangle((ox + k * 260, ly, ox + k * 260 + 16, ly + 16), fill=col)
        text(dr, (ox + k * 260 + 24, ly - 2), lab, 14, INK)
    # 表
    tx, ty = ox + 820, oy + 90
    text(dr, (tx, ty), '成功例の停止位置など', 17, INK, True)
    items = [
        ('成功', f'{s["ai"]["success"]} 人', f'{s["threshold"]["success"]} 人'),
        ('硬膜外腔への進入量', f'{s["ai"]["overshoot_mm_mean"]} mm', f'{s["threshold"]["overshoot_mm_mean"]} mm'),
        ('硬膜までの余裕（最小）', f'{s["ai"]["margin_to_dura_mm_min"]} mm', f'{s["threshold"]["margin_to_dura_mm_min"]} mm'),
        ('所要時間（平均）', f'{s["ai"]["time_s_mean"]} 秒', f'{s["threshold"]["time_s_mean"]} 秒'),
    ]
    text(dr, (tx + 190, ty + 34), 'AI', 14, TEAL, True)
    text(dr, (tx + 290, ty + 34), '圧閾値法', 14, MUTED, True)
    for k, (a, b, c) in enumerate(items):
        y = ty + 62 + k * 34
        dr.line([(tx, y - 6), (tx + 380, y - 6)], fill=LINE)
        text(dr, (tx, y), a, 14, INK)
        text(dr, (tx + 190, y), b, 14, INK, True)
        text(dr, (tx + 290, y), c, 14, INK)
    fc = summary['frame_classification_test']
    text(dr, (ox, oy + 340), '読み取り方', 18, INK, True)
    notes = [
        '・圧だけを見る方法は、閾値を下げると「弱い抵抗消失」を見逃して硬膜に達し、上げると棘間靱帯の「偽LOR」で早く止まる。',
        '・AIは「黄色靱帯のプツプツ（予兆）→ 抵抗の急低下 → プランジャーの前進」という時間的な流れを特徴量として使う。',
        f'・10 Hzごとの層の判定精度（検証用100人）：{fc["accuracy"] * 100:.1f}%。AIは深さ・位置の情報を一切使っていない。',
        '・偽LOR・弱いLORの割合はストレステストとして臨床より意図的に高く設定しており、差は大きめに表れている可能性がある。',
    ]
    for k, s_ in enumerate(notes):
        text(dr, (ox, oy + 372 + k * 32), s_, 15, INK)


def closing_scene() -> Image.Image:
    img = new_img(W, H)
    dr = ScaledDraw(img)
    draw_header(dr, 'まとめと限界')
    text(dr, (60, 90), 'まとめ', 24, TEAL, True)
    for k, s in enumerate([
        '熟練した麻酔科医の両手の感覚（左手のプツプツ・右手親指の抵抗消失）を2つのセンサー信号に置き換えた。',
        'AI（10 Hz）が針先の状態を3つに判定し、ロボットは「前進 → 減速 → 停止」と行動を変える（感知 → 推論 → 行動）。',
        '仮想患者100人で、圧の閾値だけで止める方法より安全側に止まることを確認した（合成データ上）。',
    ]):
        text(dr, (60, 132 + k * 36), '・' + s, 17, INK)
    text(dr, (60, 260), '今回は省略した臨床上の要素（実際の手技では必須）', 24, RED, True)
    for k, s in enumerate([
        '針の刺入角度と3次元的な方向修正（実際は棘突起・椎弓に当たれば刺し直す）',
        '骨への接触、血管内への迷入：逆血の確認、くも膜下腔への誤穿刺：髄液の逆流の確認',
        '患者の体動・呼吸、組織の個人差の実測、テストドーズなどの安全確認',
        '実測データでの検証（本作品の数値はすべて合成データ。医療機器ではない）',
    ]):
        text(dr, (60, 302 + k * 36), '・' + s, 17, INK)
    text(dr, (60, 470), '今後の発展', 24, TEAL, True)
    for k, s in enumerate([
        '練習用ファントムに力・圧センサーを付け、熟練医の操作データで学習・検証する',
        '術者の片耳にだけ短い音声で状態を伝えるナビゲーション（患者に警告音を聞かせない）',
    ]):
        text(dr, (60, 512 + k * 36), '・' + s, 17, INK)
    return img


# ---------------------------------------------------------------------------
# 動画の書き出し
# ---------------------------------------------------------------------------
def _ffmpeg() -> str:
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('ffmpeg が見つかりません。README の手順でインストールしてください。') from exc


def render_all(cases, res, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = res['summary']
    for c in cases:
        c['theta'] = res['theta']
    views = [CaseView(c) for c in cases]

    # 静止画（説明資料用）
    title_scene(views[0]).save(outdir / 'fig_title.png')
    background_scene(views[0]).save(outdir / 'fig_background.png')
    results_scene(summary).save(outdir / 'fig_results.png')
    closing_scene().save(outdir / 'fig_closing.png')
    for v in views:
        t_end = v.t_max
        v.frame(t_end, f'症例{v.case["key"]}', show_result=False).save(outdir / f'fig_case{v.case["key"]}_stop.png')
    mid = views[1]
    st = mid.rec['state']
    k = next((j for j, s_ in enumerate(st) if s_ == 'careful'), len(st) // 2)
    t_mid = min(float(mid.rec['t'][k]) + 2.0, mid.t_max - 1.0)
    mid.frame(t_mid, '症例B').save(outdir / 'fig_caseB_mid.png')

    # 動画の構成
    segments = [('static', title_scene(views[0]), 6.0),
                ('static', background_scene(views[0]), 15.0)]
    for v in views:
        segments.append(('case', v, v.t_max))
        segments.append(('result', v, 5.0))
    segments.append(('static', results_scene(summary), 15.0))
    segments.append(('static', closing_scene(), 11.0))
    total = sum(s[2] for s in segments)
    print(f'動画の長さ: {total:.1f} 秒')
    if not (60 <= total <= 180):
        print('注意：動画の長さが1〜3分の範囲外です。')

    out = outdir / 'EpiPilotAI_PAI_Final.mp4'
    args = [_ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
            '-an', '-vcodec', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    try:
        for kind, obj, dur in segments:
            n = int(round(dur * FPS))
            if kind == 'static':
                buf = obj.tobytes()
                for _ in range(n):
                    proc.stdin.write(buf)
            elif kind == 'case':
                for f in range(n):
                    t = f / FPS
                    img = obj.frame(t, f'症例{obj.case["key"]}', show_banner=(t < 2.5))
                    proc.stdin.write(img.tobytes())
                print(f'  症例{obj.case["key"]} 描画完了')
            else:
                buf = obj.frame(obj.t_max, f'症例{obj.case["key"]}', show_result=True).tobytes()
                for _ in range(n):
                    proc.stdin.write(buf)
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError('ffmpeg が失敗しました。')
    except Exception:
        proc.kill()
        raise
    print(f'保存しました: {out}')
