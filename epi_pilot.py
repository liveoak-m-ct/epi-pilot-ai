#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Epi-Pilot AI v2
硬膜外穿刺の抵抗消失（LOR）をAIで判定し、1軸ロボットを自動停止させるシミュレーション。

教育用・合成データのみ。実在の患者、実測データ、医療機器は一切使用していない。
実機のアクチュエータに接続してはならない。

使い方:
  python epi_pilot.py --test      # 学習・100人評価・ログ出力（約1〜3分）
  python epi_pilot.py --render    # 上記＋動画・図の生成（数分）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# シミュレーション設定
# ---------------------------------------------------------------------------
DT = 0.02             # 制御・センサー周期 50 Hz
DECIDE_EVERY = 5      # AI推論は5周期ごと = 10 Hz
TAU_MOTOR = 0.10      # アクチュエータ速度応答の時定数 [s]
TAU_FORCE = 0.03      # 力センサーの応答 [s]
TAU_PRESS = 0.08      # シリンジ内圧の応答 [s]
TAU_ATTACH = 0.15     # シリンジ接続直後の昇圧 [s]

V_FAST = 2.0          # 通常前進 [mm/s]
V_CAREFUL = 0.6       # 慎重前進（黄色靱帯を検知後）[mm/s]
V_BASELINE = 1.0      # 比較法（圧閾値法）の一定速度 [mm/s]
P_CAREFUL_FLAVUM = 0.5
P_CAREFUL_LOR = 0.3
P_STOP = 0.8
CONSEC = 2            # 停止には2回連続（0.2秒）の判定が必要
WARMUP_S = 1.0        # シリンジ接続後、判定を始めるまでの時間

N_TRAIN = 300         # AI学習用の仮想患者
N_TUNE = 100          # 圧閾値の調整に使う仮想患者（学習用から）
N_TEST = 100          # 評価用の仮想患者（学習に一切使わない）
SEED_TRAIN = 11
SEED_TEST = 20261028

# 皮膚から脊髄までの層（背側→腹側の順）
LAYER_KEYS = ['epidermis', 'dermis', 'subcut', 'supra', 'inter', 'flavum',
              'epidural', 'dura', 'arach', 'subarach', 'cord']
LAYER_JA = {
    'epidermis': '表皮', 'dermis': '真皮', 'subcut': '皮下組織',
    'supra': '棘上靱帯', 'inter': '棘間靱帯', 'flavum': '黄色靱帯',
    'epidural': '硬膜外腔', 'dura': '硬膜', 'arach': 'くも膜',
    'subarach': 'くも膜下腔', 'cord': '脊髄（馬尾）',
}
# 各層で針が受ける抵抗（任意単位）。臨床の手応えの順序を模したもので、実測値ではない。
BASE_FORCE = {
    'epidermis': 1.3, 'dermis': 1.5, 'subcut': 0.35, 'supra': 1.1,
    'inter': 0.9, 'flavum': 1.5, 'epidural': 0.3, 'dura': 0.9,
    'arach': 0.5, 'subarach': 0.2, 'cord': 0.3,
}
CLASS_JA = ['組織内（棘間靱帯など）', '黄色靱帯（プツプツ）', '硬膜外腔（LOR）']
FEATURE_NAMES = ['圧比', '圧低下量', 'プランジャー速度', '抵抗比', '抵抗低下量',
                 'プツプツ強度（現在）', 'プツプツ強度（直近4秒の最大）', '抵抗レベル', '針の速度']


# ---------------------------------------------------------------------------
# 仮想患者
# ---------------------------------------------------------------------------
@dataclass
class Patient:
    pid: int
    seed: int
    thick: dict
    bounds: dict            # 各層の開始深さ [mm]
    k_f: float              # 抵抗の個人差
    k_p: float              # シリンジ圧の個人差
    crackle_rate: float     # 黄色靱帯での「プツプツ」の頻度 [回/mm]
    crackle_amp: float      # 「プツプツ」の強さ
    lor_open: float         # 硬膜外腔での抵抗消失の強さ（0〜1）
    weak_lor: bool
    false_lors: list        # 棘間靱帯内の偽LOR [(開始mm, 終了mm, 強さ)]
    attach_depth: float     # シリンジを接続する深さ（棘間靱帯内）
    inter_period: float
    inter_phase: float

    def layer_at(self, d: float) -> str:
        for k in reversed(LAYER_KEYS):
            if d >= self.bounds[k]:
                return k
        return LAYER_KEYS[0]

    @property
    def epi_start(self) -> float:
        return self.bounds['epidural']

    @property
    def dura_start(self) -> float:
        return self.bounds['dura']


def make_patient(rng: np.random.Generator, pid: int) -> Patient:
    th = {
        'epidermis': 0.1, 'dermis': rng.uniform(1.5, 2.5),
        'subcut': rng.uniform(8, 32), 'supra': rng.uniform(3, 6),
        'inter': rng.uniform(12, 24), 'flavum': rng.uniform(3, 6),
        'epidural': rng.uniform(3, 7), 'dura': rng.uniform(0.4, 0.6),
        'arach': 0.1, 'subarach': rng.uniform(3, 6), 'cord': 12.0,
    }
    bounds, d = {}, 0.0
    for k in LAYER_KEYS:
        bounds[k] = d
        d += th[k]
    bounds['end'] = d
    weak = bool(rng.random() < 0.25)
    lor_open = rng.uniform(0.45, 0.65) if weak else rng.uniform(0.75, 0.95)
    attach = bounds['inter'] + rng.uniform(2.0, 3.5)
    false_lors = []
    for _ in range(int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))):
        lo, hi = attach + 1.5, bounds['flavum'] - 4.0
        if hi <= lo:
            break
        s = rng.uniform(lo, hi)
        e = s + rng.uniform(1.5, 3.0)
        if all(e < a or s > b for a, b, _ in false_lors):
            false_lors.append((s, min(e, bounds['flavum'] - 1.0), rng.uniform(0.35, 0.70)))
    false_lors.sort()
    return Patient(pid=pid, seed=int(rng.integers(1, 2**31)), thick=th, bounds=bounds,
                   k_f=rng.uniform(0.8, 1.25), k_p=rng.uniform(0.85, 1.15),
                   crackle_rate=rng.uniform(2.0, 5.0), crackle_amp=rng.uniform(0.06, 0.45),
                   lor_open=lor_open, weak_lor=weak, false_lors=false_lors,
                   attach_depth=attach, inter_period=rng.uniform(2, 5),
                   inter_phase=rng.uniform(0, 2 * math.pi))


def make_patients(n: int, seed: int, start_id: int = 1) -> list[Patient]:
    rng = np.random.default_rng(seed)
    return [make_patient(rng, start_id + i) for i in range(n)]


def true_label(p: Patient, d: float) -> int:
    if d < p.bounds['flavum']:
        return 0
    if d < p.bounds['epidural']:
        return 1
    return 2


# ---------------------------------------------------------------------------
# 1軸ロボット（針）と組織・シリンジのモデル
# ---------------------------------------------------------------------------
class Plant:
    """針を進める直動アクチュエータ＋2つのセンサー（針の抵抗・シリンジ内圧）。"""

    def __init__(self, p: Patient, rng: np.random.Generator):
        self.p, self.rng = p, rng
        self.t = 0.0
        self.depth = 0.0
        self.vel = 0.0
        self.f_state = 0.3
        self.pop = 0.0
        self.attached = False
        self.t_attach = None
        self.P = 0.0
        self.art_t = 0.0
        self.art_mag = 0.0
        self.plunger = 0.0
        self.openness = 0.0

    def step(self, v_cmd: float):
        p, rng = self.p, self.rng
        self.vel += (v_cmd - self.vel) * (1 - math.exp(-DT / TAU_MOTOR))
        self.depth += self.vel * DT
        self.t += DT
        d = self.depth
        L = p.layer_at(d)

        # --- 信号①：針を進める抵抗（左手の感覚に相当）---
        base = BASE_FORCE[L]
        if L in ('epidermis', 'dermis'):
            base += 0.5 * d / max(p.bounds['subcut'], 1e-6)
        elif L == 'inter':
            base += 0.12 * math.sin(2 * math.pi * d / p.inter_period + p.inter_phase)
        elif L == 'flavum':
            frac = (d - p.bounds['flavum']) / p.thick['flavum']
            if frac > 0.7:  # 貫通直前に靱帯が押されて抵抗が高まる（テンティング）
                base += 0.55 * (frac - 0.7) / 0.3
        elif L == 'dura':
            base += 0.3
        base *= p.k_f * (0.7 + 0.3 * min(max(self.vel, 0.0), 2.5))
        dd = max(self.vel, 0.0) * DT
        if L == 'flavum':      # 黄色靱帯の線維を切る「プツプツ」
            rate, amp = p.crackle_rate, p.crackle_amp
        elif L == 'inter':
            rate, amp = 0.6, 0.10
        else:
            rate, amp = 0.0, 0.0
        if rate and rng.random() < rate * dd:
            self.pop += amp * p.k_f * rng.uniform(0.6, 1.3)
        self.pop *= math.exp(-DT / 0.035)
        f_true = base + 0.006 * d + self.pop
        self.f_state += (f_true - self.f_state) * (1 - math.exp(-DT / TAU_FORCE))
        force = self.f_state + rng.normal(0, 0.025)

        # --- 信号②：シリンジ内圧（右手親指の感覚に相当）---
        if not self.attached and d >= p.attach_depth:
            self.attached = True
            self.t_attach = self.t
        openness = 0.0
        if self.attached:
            if p.bounds['epidural'] <= d < p.bounds['dura']:
                openness = p.lor_open
            elif d >= p.bounds['subarach']:
                openness = 0.9
            else:
                for s, e, o in p.false_lors:
                    if s <= d < e:
                        openness = o
            target = p.k_p * (1 - openness)
            tau = TAU_ATTACH if self.t - self.t_attach < 0.6 else TAU_PRESS
            self.P += (target - self.P) * (1 - math.exp(-DT / tau))
            if self.art_t > 0:
                self.art_t -= DT
            elif rng.random() < 0.08 * DT:   # 圧だけの一過性の乱れ（センサー外乱）
                self.art_t = rng.uniform(0.06, 0.12)
                self.art_mag = rng.uniform(0.25, 0.45) * p.k_p
            pressure = self.P - (self.art_mag if self.art_t > 0 else 0.0) + rng.normal(0, 0.015)
            self.plunger += 0.25 * openness * DT
        else:
            pressure = float('nan')
        self.openness = openness
        return force, pressure, openness, L


# ---------------------------------------------------------------------------
# 特徴量（位置・深さは使わない）
# ---------------------------------------------------------------------------
def _movmean(a: np.ndarray, n: int = 5) -> np.ndarray:
    if len(a) < n:
        return a.copy()
    return np.convolve(a, np.ones(n) / n, mode='valid')


class FeatureExtractor:
    def __init__(self):
        self.f = deque(maxlen=250)
        self.p = deque(maxlen=250)
        self.pl = deque(maxlen=250)
        self.f_all: list[float] = []
        self.crk_hist = deque(maxlen=40)

    def push(self, f: float, p: float, pl: float):
        self.f.append(f)
        self.p.append(p)
        self.pl.append(pl)
        self.f_all.append(f)

    def ready(self) -> bool:
        return len(self.p) >= int(WARMUP_S / DT)

    def features(self, speed: float) -> list[float]:
        f = np.fromiter(self.f, float)
        p = np.fromiter(self.p, float)
        pl = np.fromiter(self.pl, float)
        p_now = p[-5:].mean()
        p_ref = max(float(np.percentile(p, 90)), 0.2)
        ps = _movmean(p[-34:])
        p_drop = (ps.max() - p_now) / p_ref
        plv = (pl[-1] - pl[-16]) / 0.3
        fs = _movmean(f[-104:])
        f_now = f[-5:].mean()
        f_ref = max(fs.max(), 1e-3)
        f_ratio = f_now / f_ref
        f_drop = (fs[-30:].max() - f_now) / f_ref
        crk = float(np.std(np.diff(f[-26:])))
        self.crk_hist.append(crk)
        crk_mem = max(self.crk_hist)
        f_level = f_now / max(float(np.median(self.f_all)), 1e-3)
        return [p_now / p_ref, p_drop, plv, f_ratio, f_drop, crk, crk_mem, f_level, speed]


# ---------------------------------------------------------------------------
# AIの学習（合成データ）
# ---------------------------------------------------------------------------
def collect_dataset(patients: list[Patient], seed: int):
    """速度をランダムに変えながら針を進め、10 Hzで特徴量と正解（針先の層）を記録する。"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for p in patients:
        plant = Plant(p, np.random.default_rng(p.seed))
        fx = FeatureExtractor()
        v, next_switch, step = V_FAST, None, 0
        while plant.depth < p.dura_start and plant.t < 200:
            if plant.attached and (next_switch is None or plant.t >= next_switch):
                v = rng.uniform(0.4, 2.2)
                next_switch = plant.t + rng.uniform(1.5, 5.0)
            f, pr, _, _ = plant.step(v)
            step += 1
            if plant.attached:
                fx.push(f, pr, plant.plunger)
                if step % DECIDE_EVERY == 0 and fx.ready():
                    X.append(fx.features(plant.vel))
                    y.append(true_label(p, plant.depth))
    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=3,
                                 class_weight='balanced', random_state=SEED_TRAIN, n_jobs=-1)
    clf.fit(X, y)
    clf.set_params(n_jobs=1)
    return clf


# ---------------------------------------------------------------------------
# 閉ループ制御（1人分の穿刺）
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    patient: Patient
    method: str                 # 'ai' または 'threshold'
    stop_depth: float = 0.0
    outcome: str = ''           # 'success' / 'early' / 'dura'
    t_total: float = 0.0
    t_stop_cmd: float | None = None
    rec: dict = field(default_factory=dict)

    @property
    def overshoot(self) -> float:   # 硬膜外腔に入ってから停止までの距離
        return self.stop_depth - self.patient.epi_start

    @property
    def margin(self) -> float:      # 停止位置から硬膜までの余裕
        return self.patient.dura_start - self.stop_depth


REC_KEYS = ['t', 'depth', 'vel', 'vcmd', 'force', 'pressure', 'plunger', 'open',
            'layer', 'p0', 'p1', 'p2', 'state', 'pratio']


def run_episode(p: Patient, method: str, model=None, theta: float = 0.5,
                record: bool = False) -> Episode:
    plant = Plant(p, np.random.default_rng(p.seed + 7))
    fx = FeatureExtractor()
    state, consec, step = 'advance', 0, 0
    probs = np.array([1.0, 0.0, 0.0])
    pratio = float('nan')
    ep = Episode(patient=p, method=method)
    rec = {k: [] for k in REC_KEYS} if record else None
    while True:
        if state == 'stop':
            v = 0.0
        elif method == 'ai':
            v = V_CAREFUL if state == 'careful' else V_FAST
        else:
            v = V_BASELINE if plant.attached else V_FAST
        f, pr, op, L = plant.step(v)
        step += 1
        if plant.attached:
            fx.push(f, pr, plant.plunger)
            if state != 'stop' and step % DECIDE_EVERY == 0 and fx.ready():
                x = fx.features(plant.vel)
                pratio = x[0]
                if method == 'ai':
                    probs = model.predict_proba(np.array([x]))[0]
                    if state == 'advance' and (probs[1] >= P_CAREFUL_FLAVUM or probs[2] >= P_CAREFUL_LOR):
                        state = 'careful'
                    consec = consec + 1 if probs[2] >= P_STOP else 0
                else:
                    consec = consec + 1 if x[0] < theta else 0
                if consec >= CONSEC:
                    state = 'stop'
                    ep.t_stop_cmd = plant.t
        if record:
            for k, val in zip(REC_KEYS, [plant.t, plant.depth, plant.vel, v, f, pr, plant.plunger,
                                         op, L, probs[0], probs[1], probs[2], state, pratio]):
                rec[k].append(val)
        if plant.depth >= p.dura_start + 0.3:
            break
        if state == 'stop' and abs(plant.vel) < 0.005:
            break
        if plant.t > 150:
            break
    d = plant.depth
    ep.stop_depth = d
    ep.t_total = plant.t
    ep.outcome = 'early' if d < p.epi_start else ('success' if d < p.dura_start else 'dura')
    if record:
        ep.rec = {k: (np.array(v) if k not in ('layer', 'state') else v) for k, v in rec.items()}
    return ep


def tune_threshold(patients: list[Patient]) -> tuple[float, list]:
    table = []
    for th in np.round(np.arange(0.30, 0.81, 0.05), 2):
        eps = [run_episode(p, 'threshold', theta=float(th)) for p in patients]
        s = sum(e.outcome == 'success' for e in eps)
        du = sum(e.outcome == 'dura' for e in eps)
        table.append((float(th), s, du))
    best = max(table, key=lambda r: (r[1], -r[2]))
    return best[0], table


def summarize(eps: list[Episode]) -> dict:
    n = len(eps)
    succ = [e for e in eps if e.outcome == 'success']
    return {
        'n': n,
        'success': len(succ),
        'early_stop': sum(e.outcome == 'early' for e in eps),
        'dura_reached': sum(e.outcome == 'dura' for e in eps),
        'overshoot_mm_mean': round(float(np.mean([e.overshoot for e in succ])), 2) if succ else None,
        'overshoot_mm_max': round(float(np.max([e.overshoot for e in succ])), 2) if succ else None,
        'margin_to_dura_mm_mean': round(float(np.mean([e.margin for e in succ])), 2) if succ else None,
        'margin_to_dura_mm_min': round(float(np.min([e.margin for e in succ])), 2) if succ else None,
        'time_s_mean': round(float(np.mean([e.t_total for e in eps])), 1),
    }


# ---------------------------------------------------------------------------
# 実験全体
# ---------------------------------------------------------------------------
def run_experiment(outdir: Path, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    train_pts = make_patients(N_TRAIN, SEED_TRAIN, start_id=1001)
    test_pts = make_patients(N_TEST, SEED_TEST, start_id=1)

    log(f'[1/5] 学習用データを生成中（仮想患者 {N_TRAIN}人）...')
    X, y = collect_dataset(train_pts, seed=SEED_TRAIN + 1)
    log(f'      サンプル {len(y)}件（組織内 {int((y==0).sum())} / 黄色靱帯 {int((y==1).sum())} / 硬膜外腔 {int((y==2).sum())}）')
    log('[2/5] AI（RandomForest）を学習中...')
    model = train_model(X, y)

    log(f'[3/5] 比較法（圧閾値法）の閾値を学習用患者 {N_TUNE}人で調整中...')
    theta, tune_table = tune_threshold(train_pts[:N_TUNE])
    log(f'      最良の閾値 = 基準圧の {theta:.2f} 倍未満で停止')

    log(f'[4/5] 評価用の仮想患者 {N_TEST}人で閉ループ試験中（AIと圧閾値法）...')
    ai_eps = [run_episode(p, 'ai', model=model) for p in test_pts]
    th_eps = [run_episode(p, 'threshold', theta=theta) for p in test_pts]

    log('[5/5] フレーム単位の分類精度を評価中...')
    Xt, yt = collect_dataset(test_pts, seed=SEED_TEST + 1)
    yp = model.predict(Xt)
    cm = np.zeros((3, 3), int)
    for a, b in zip(yt, yp):
        cm[a, b] += 1
    recall = [round(float(cm[i, i] / max(cm[i].sum(), 1)), 3) for i in range(3)]
    precision = [round(float(cm[i, i] / max(cm[:, i].sum(), 1)), 3) for i in range(3)]

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'results_100_patients.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['患者ID', '皮膚〜硬膜外腔の深さmm', '黄色靱帯の厚さmm', '硬膜外腔の幅mm',
                    '偽LORの数', 'LORが弱い', 'プツプツ頻度/mm', 'プツプツ強さ',
                    'AI_結果', 'AI_停止深さmm', 'AI_硬膜外腔への進入量mm', 'AI_硬膜までの余裕mm', 'AI_所要時間s',
                    '閾値法_結果', '閾値法_停止深さmm', '閾値法_硬膜外腔への進入量mm', '閾値法_硬膜までの余裕mm', '閾値法_所要時間s'])
        ja = {'success': '成功（硬膜外腔で停止）', 'early': '早期停止', 'dura': '硬膜に到達'}
        for p, a, b in zip(test_pts, ai_eps, th_eps):
            w.writerow([p.pid, round(p.epi_start, 1), round(p.thick['flavum'], 1), round(p.thick['epidural'], 1),
                        len(p.false_lors), 'はい' if p.weak_lor else 'いいえ',
                        round(p.crackle_rate, 2), round(p.crackle_amp, 2),
                        ja[a.outcome], round(a.stop_depth, 2), round(a.overshoot, 2), round(a.margin, 2), round(a.t_total, 1),
                        ja[b.outcome], round(b.stop_depth, 2), round(b.overshoot, 2), round(b.margin, 2), round(b.t_total, 1)])

    summary = {
        'note': '合成データによる教育用シミュレーション。臨床性能を示すものではない。',
        'train_patients': N_TRAIN, 'test_patients': N_TEST,
        'train_samples': int(len(y)),
        'model': 'RandomForestClassifier(n_estimators=80, max_depth=12)',
        'features': FEATURE_NAMES,
        'feature_importance': {n: round(float(v), 3) for n, v in zip(FEATURE_NAMES, model.feature_importances_)},
        'threshold_method': {'theta': theta, 'tuning_table(theta, success, dura)': tune_table},
        'closed_loop_test': {'ai': summarize(ai_eps), 'threshold': summarize(th_eps)},
        'frame_classification_test': {
            'classes': CLASS_JA, 'confusion_matrix(rows=true, cols=pred)': cm.tolist(),
            'recall': recall, 'precision': precision,
            'accuracy': round(float(np.trace(cm) / cm.sum()), 3)},
        'test_population': {
            'with_false_lor': sum(1 for p in test_pts if p.false_lors),
            'weak_lor': sum(1 for p in test_pts if p.weak_lor),
            'skin_to_epidural_mm_mean': round(float(np.mean([p.epi_start for p in test_pts])), 1),
            'skin_to_epidural_mm_range': [round(min(p.epi_start for p in test_pts), 1),
                                          round(max(p.epi_start for p in test_pts), 1)]},
        'runtime_s': round(time.time() - t0, 1),
    }
    (outdir / 'result_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(json.dumps(summary['closed_loop_test'], ensure_ascii=False, indent=2))
    return {'model': model, 'theta': theta, 'test_pts': test_pts, 'ai_eps': ai_eps,
            'th_eps': th_eps, 'summary': summary}


def pick_cases(res: dict) -> list[dict]:
    """動画用に3症例を選ぶ（典型例・偽LOR・弱いLOR）。"""
    pts, ai, th = res['test_pts'], res['ai_eps'], res['th_eps']
    idx = range(len(pts))

    def first(cond, fallback):
        for i in idx:
            if cond(i):
                return i
        for i in idx:
            if fallback(i):
                return i
        return 0

    a = first(lambda i: not pts[i].false_lors and not pts[i].weak_lor and ai[i].outcome == 'success'
              and th[i].outcome == 'success' and pts[i].epi_start < 50,
              lambda i: ai[i].outcome == 'success')
    b = first(lambda i: th[i].outcome == 'early' and ai[i].outcome == 'success' and pts[i].false_lors
              and pts[i].epi_start < 55, lambda i: bool(pts[i].false_lors))
    c = first(lambda i: th[i].outcome == 'dura' and ai[i].outcome == 'success' and pts[i].weak_lor
              and pts[i].epi_start < 55, lambda i: pts[i].weak_lor)
    cases = []
    for key, i, title in [('A', a, '症例A：典型例（はっきりした抵抗消失）'),
                          ('B', b, '症例B：棘間靱帯内の「偽の抵抗消失」'),
                          ('C', c, '症例C：抵抗消失が弱く、わかりにくい例')]:
        p = pts[i]
        cases.append({'key': key, 'title': title, 'patient': p,
                      'ai': run_episode(p, 'ai', model=res['model'], record=True),
                      'th': th[i]})
    return cases


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', action='store_true', help='学習・100人評価・ログ出力')
    ap.add_argument('--render', action='store_true', help='評価に加えて動画と図を生成')
    ap.add_argument('--out', type=Path, default=HERE / 'output', help='出力先フォルダ')
    a = ap.parse_args()
    res = run_experiment(a.out)
    s = res['summary']['closed_loop_test']
    assert s['ai']['success'] >= s['threshold']['success'], 'AIが比較法を下回りました。'
    assert s['ai']['dura_reached'] == 0, 'AIで硬膜到達が発生しました。'
    if a.render:
        from render import render_all   # 描画は別ファイル
        cases = pick_cases(res)
        render_all(cases, res, a.out)
    print(f'完了。出力先: {a.out}')


if __name__ == '__main__':
    main()
