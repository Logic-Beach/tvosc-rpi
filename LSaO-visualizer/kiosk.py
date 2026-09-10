#!/usr/bin/env python3
"""Fullscreen CRT kiosk for LSaO live visualizers.

Renders each frame at 300x300, then nearest-neighbor stretches to fill 720x480.
pygame SCALED letterboxed the square buffer; we blit onto a native 720x480
window instead. SIGUSR2 cycles visualizer types.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "0")

import numpy as np
import pygame
from scipy.signal import lfilter

import main as lsao

DISPLAY_W = 720
DISPLAY_H = 480
# Match NTSC composite field rate. The UHF modulator just rebroadcasts analog
# video; it does not lock the vis to 24 fps. Override with OSC_FPS if needed.
FPS = int(os.environ.get("OSC_FPS", "60"))
CHANNEL = "Both (Merge to mono)"
STEREO = "Both (Stereo)"
# Linear spectrum currently maps 1–xhigh Hz across the full width.
# 2000 Hz puts bass/mids on screen instead of spreading out to 13 kHz.
SPECTRUM_XLOW = 1
SPECTRUM_XHIGH = 2000
SCOPE_DOT_THICKNESS = 2

# Every vis renders at 300x300, then nearest-neighbor stretched to fill the CRT.
# Order matches the upstream LSaO menu, with a triggered bass waveform after
# Short Waveform. Chladni is one plate (Cosine).
INTERNAL = 300
CHLADNI_PLATE = "Cosine"
MODES = [
    #"Spectrum",
    #"SpectrumdB",
   # "SpecBalance",
   # "Histogram",
   # "Waveform",
    "Bass Harmonic Anchor",
    "Spectral Seismograph",
   # "Recurrence",
    "Bass-anchored recurrence plot",
    "Oscilloscope",
    #"Polar",
    "PolarStereo",
    #"Poincare",
    "DelayEmbed",
    "Strange Attractor",
    "Starfield Zoom",
    "Hall of Mirrors",
   #"Chladni",
   # "Envelope",
]
RENDER_SIZE = {name: (INTERNAL, INTERNAL) for name in MODES}
MIN_DIM = INTERNAL
DEFAULT_MODE = "Bass Harmonic Anchor"

pending_next = False
pending_prev = False
latest_block: np.ndarray | None = None
_audio_stream = None
_audio_proc: subprocess.Popen | None = None
NEXT_LOCKOUT_S = 0.08
AFFECT_FIFO = Path.home() / ".local/state" / "lsao-next"
BACKEND_FILE = Path.home() / ".config" / "lsao" / "backend"
_fifo_fd: int | None = None

# Bass Harmonic Anchor: a real scope trigger over 30–150 Hz. The filtered signal
# only decides where a sweep starts; the displayed signal is always fresh,
# unfiltered audio. A Schmitt-armed upward zero crossing is stable even when
# the raw synth has large, moving harmonics.
RING_SAMPLES = 16384
# Free-running short trace: ~30 ms of audio (was one ~5 ms capture chunk).
SHORT_WINDOW = 1440
TRIG_WINDOW = 2400
TRIG_WINDOW_MIN = 640
TRIG_WINDOW_MAX = 3600
TRIG_CYCLES = 2.0
TRIG_SEARCH = 2400
TRIG_NEED = TRIG_WINDOW_MAX + TRIG_SEARCH
TRIG_GATE_MIN = 0.0015
TRIG_GATE_FRAC = 0.12
TRIG_LP_HZ = 180.0
TRIG_LP_POLES = 4
TRIG_FMIN = 30.0
TRIG_FMAX = 150.0
TRIG_PERIOD_TOL = 0.14
_sticky_win = 0
_ring = np.zeros((RING_SAMPLES, 2), dtype=np.float32)
_lp = np.zeros(RING_SAMPLES, dtype=np.float32)
_ring_i = 0
_ring_filled = 0
_ring_lock = threading.Lock()
_lp_alpha = float(1.0 - np.exp(-2.0 * np.pi * TRIG_LP_HZ / 48000.0))
_lp_b = np.array([_lp_alpha], dtype=np.float64)
_lp_a = np.array([1.0, _lp_alpha - 1.0], dtype=np.float64)
_lp_zi = [np.zeros(1, dtype=np.float64) for _ in range(TRIG_LP_POLES)]
_total = 0
_period = 0.0

# Bass-anchored recurrence: compare local three-sample states over the same
# fresh two-cycle sweep used by Bass Harmonic Anchor.
BASS_REC_EMBED_DIM = 3
BASS_REC_DELAY_CYCLE = 0.125
BASS_REC_SIGMA = 0.30
BASS_REC_SILENCE = 0.003
BASS_REC_FULL_LEVEL = 0.040
BASS_REC_SIZE = 220
BASS_REC_LUT_SIZE = 2048
_bass_rec_spread2 = BASS_REC_EMBED_DIM * BASS_REC_SIGMA**2
_bass_rec_max_distance2 = 16.0 * _bass_rec_spread2
_bass_rec_lut_scale = (BASS_REC_LUT_SIZE - 1) / _bass_rec_max_distance2
_bass_rec_lut = np.rint(
    255.0
    * np.exp(
        -np.linspace(0.0, _bass_rec_max_distance2, BASS_REC_LUT_SIZE)
        / (2.0 * _bass_rec_spread2)
    )
).astype(np.uint8)

# Spectral Seismograph: three live band-filtered waveform lanes with guard gaps
# and no AGC. Every lane uses the same bass-derived trigger, keeping harmonics
# aligned while fresh audio continues to replace every frame.
LONG_BANDS = ((30.0, 180.0), (180.0, 1800.0), (2500.0, 20000.0))
LONG_TRACE_GAIN = 1.75
LONG_FILTER_POLES = (3, 2, 1)
LONG_FILTER_PAD = 768
_long_filter_coefficients = []
for (_low_hz, _high_hz), _poles in zip(LONG_BANDS, LONG_FILTER_POLES):
    _long_lp_alpha = float(1.0 - np.exp(-2.0 * np.pi * _high_hz / 48000.0))
    _long_hp_alpha = float(np.exp(-2.0 * np.pi * _low_hz / 48000.0))
    _long_filter_coefficients.append(
        (
            np.array([_long_hp_alpha, -_long_hp_alpha], dtype=np.float64),
            np.array([1.0, -_long_hp_alpha], dtype=np.float64),
            np.array([_long_lp_alpha], dtype=np.float64),
            np.array([1.0, _long_lp_alpha - 1.0], dtype=np.float64),
            _poles,
        )
    )

# Stereo fractals use fixed input calibration, not AGC. Left and right RMS
# envelopes independently alter equation coefficients.
FRACTAL_GATE = 0.0015
FRACTAL_ATTACK_S = 0.010
FRACTAL_RELEASE_S = 0.045
FRACTAL_UPDATE_S = 1.0 / 24.0
FRACTAL_BASS_SAMPLES = 2048
FRACTAL_BASS_LOW_HZ = 30.0
FRACTAL_BASS_HIGH_HZ = 180.0
FRACTAL_BASS_LP_POLES = 2
_fractal_bass_lp_alpha = float(
    1.0 - np.exp(-2.0 * np.pi * FRACTAL_BASS_HIGH_HZ / 48000.0)
)
_fractal_bass_lp_b = np.array([_fractal_bass_lp_alpha], dtype=np.float64)
_fractal_bass_lp_a = np.array(
    [1.0, _fractal_bass_lp_alpha - 1.0],
    dtype=np.float64,
)
_fractal_bass_hp_alpha = float(
    np.exp(-2.0 * np.pi * FRACTAL_BASS_LOW_HZ / 48000.0)
)
_fractal_bass_hp_b = np.array(
    [_fractal_bass_hp_alpha, -_fractal_bass_hp_alpha],
    dtype=np.float64,
)
_fractal_bass_hp_a = np.array(
    [1.0, -_fractal_bass_hp_alpha],
    dtype=np.float64,
)
_fractal_bass_lp_zi = [
    np.zeros((1, 2), dtype=np.float64) for _ in range(FRACTAL_BASS_LP_POLES)
]
_fractal_bass_hp_zi = np.zeros((1, 2), dtype=np.float64)
_fractal_bass_ring = np.zeros((RING_SAMPLES, 2), dtype=np.float32)
_fractal_input_rms = np.zeros(2, dtype=np.float32)
_fractal_levels = np.zeros(2, dtype=np.float32)
_fractal_bass_input = np.zeros(2, dtype=np.float32)
_fractal_bass_levels = np.zeros(2, dtype=np.float32)
_fractal_level_time = 0.0

ATTRACTOR_SIZE = 200
ATTRACTOR_POINTS = 1400
ATTRACTOR_BURN_IN = 80
_attractor_frame = np.zeros((ATTRACTOR_SIZE, ATTRACTOR_SIZE), dtype=np.uint8)
_attractor_frame_time = 0.0
_attractor_phase = np.zeros(2, dtype=np.float64)
_attractor_coefficients = np.array([1.40, -2.30, 2.40, -2.10])
_attractor_trail = np.zeros((ATTRACTOR_SIZE, ATTRACTOR_SIZE), dtype=np.float32)

STARFIELD_MAX_STARS = 800
STARFIELD_MIN_STARS = 0
STARFIELD_NEAR_Z = 0.08
STARFIELD_FAR_Z = 4#4.8
STARFIELD_SPEED = 0.72
STARFIELD_HIGH_FULL = 0.05
_starfield_rng = np.random.default_rng(1983)
_starfield_x = _starfield_rng.uniform(-1.25, 1.25, STARFIELD_MAX_STARS)
_starfield_y = _starfield_rng.uniform(-1.25, 1.25, STARFIELD_MAX_STARS)
_starfield_z = _starfield_rng.uniform(
    STARFIELD_NEAR_Z,
    STARFIELD_FAR_Z,
    STARFIELD_MAX_STARS,
)
_starfield_active = np.zeros(STARFIELD_MAX_STARS, dtype=bool)
_starfield_active[:STARFIELD_MIN_STARS] = True
_starfield_volume = 0.0
_starfield_kick = 0.0
_starfield_high = 0.0
_starfield_pan = 0.0
_starfield_warp = 0.0
_starfield_spawn_credit = 0.0
_starfield_time = 0.0

MIRROR_SHRINK = 0.97
MIRROR_DECAY = 0.975
MIRROR_CORNER_RADIUS = 0.10
MIRROR_VOLUME_GATE = 0.005
MIRROR_VOLUME_FULL = 0.45
MIRROR_BASS_FULL = 0.12
MIRROR_HIGH_FULL = 0.04
_mirror_frame = np.zeros((INTERNAL, INTERNAL), dtype=np.float32)
_mirror_levels = np.zeros(2, dtype=np.float32)
_mirror_tilt = 0.0
_mirror_bass = 0.0
_mirror_high = 0.0
_mirror_border_width = 50
_mirror_time = 0.0
_mirror_scale_phase = False
_mirror_edge_shape = (0, 0)
_mirror_edge_layers: tuple[np.ndarray, ...] = ()
_mirror_screen_mask = np.zeros((INTERNAL, INTERNAL), dtype=bool)


def request_next(*_args) -> None:
    global pending_next
    pending_next = True


def request_prev(*_args) -> None:
    global pending_prev
    pending_prev = True


def open_affect_fifo() -> None:
    global _fifo_fd
    AFFECT_FIFO.parent.mkdir(parents=True, exist_ok=True)
    if AFFECT_FIFO.exists() and not AFFECT_FIFO.is_fifo():
        AFFECT_FIFO.unlink()
    if not AFFECT_FIFO.exists():
        os.mkfifo(AFFECT_FIFO, 0o600)
    _fifo_fd = os.open(AFFECT_FIFO, os.O_RDWR | os.O_NONBLOCK)


def fifo_step() -> int:
    """+1 next, -1 previous, 0 none."""
    if _fifo_fd is None:
        return 0
    try:
        data = os.read(_fifo_fd, 256)
    except BlockingIOError:
        return 0
    except OSError:
        return 0
    if not data:
        return 0
    for c in data:
        if c in (ord("p"), ord("P")):
            return -1
        if c in (ord("n"), ord("N"), ord("x"), ord("X")):
            return 1
    return 1


def to_u8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.bool_ or arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.5:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def as_image_array(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(raw), nan=0.0, posinf=0.0, neginf=0.0)
    # Some renderers store (width, height) instead of image (height, width).
    # On a square canvas those shapes are identical, so never transpose squares
    # or a horizontal Y-T waveform becomes a vertical trace.
    if (
        arr.ndim == 2
        and width != height
        and arr.shape[0] == width
        and arr.shape[1] == height
    ):
        arr = arr.T
    return np.ascontiguousarray(to_u8(arr))


def to_internal_array(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """Force a 240x240 raster without PIL."""
    w = max(int(width), 1)
    h = max(int(height), 1)
    arr = as_image_array(raw, w, h)
    if arr.ndim != 2:
        arr = np.zeros((h, w), dtype=np.uint8)
    if arr.shape == (h, w):
        return arr
    y = np.linspace(0, arr.shape[0] - 1, h).astype(np.int32)
    x = np.linspace(0, arr.shape[1] - 1, w).astype(np.int32)
    return np.ascontiguousarray(arr[y[:, None], x])


def stereo_block(block: np.ndarray) -> np.ndarray:
    if block.ndim == 1:
        return np.stack([block, block], axis=1)
    if block.shape[-1] == 1:
        return np.repeat(block, 2, axis=1)
    return block


def prep_block(block: np.ndarray) -> np.ndarray:
    b = np.nan_to_num(np.asarray(block, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    b = stereo_block(b)
    peak = float(np.max(np.abs(b))) if b.size else 0.0
    if peak > 1.0:
        b = b / peak
    return b


def push_audio(block: np.ndarray) -> None:
    """Keep a rolling stereo ring for Bass Harmonic Anchor (and latest_block)."""
    global latest_block, _ring_i, _ring_filled, _lp_zi, _total
    global _fractal_bass_lp_zi, _fractal_bass_hp_zi
    b = stereo_block(np.asarray(block, dtype=np.float32))
    if b.size == 0:
        return
    latest_block = b
    n = min(int(b.shape[0]), RING_SAMPLES)
    b = b[-n:]
    mono = np.mean(b, axis=1)
    y = mono
    for k in range(TRIG_LP_POLES):
        y, _lp_zi[k] = lfilter(_lp_b, _lp_a, y, zi=_lp_zi[k])
    filt = np.asarray(y, dtype=np.float32)
    bass = b
    for k in range(FRACTAL_BASS_LP_POLES):
        bass, _fractal_bass_lp_zi[k] = lfilter(
            _fractal_bass_lp_b,
            _fractal_bass_lp_a,
            bass,
            axis=0,
            zi=_fractal_bass_lp_zi[k],
        )
    bass, _fractal_bass_hp_zi = lfilter(
        _fractal_bass_hp_b,
        _fractal_bass_hp_a,
        bass,
        axis=0,
        zi=_fractal_bass_hp_zi,
    )
    bass = np.asarray(bass, dtype=np.float32)
    with _ring_lock:
        i = _ring_i
        first = min(n, RING_SAMPLES - i)
        _ring[i : i + first] = b[:first]
        _lp[i : i + first] = filt[:first]
        _fractal_bass_ring[i : i + first] = bass[:first]
        rest = n - first
        if rest:
            _ring[0:rest] = b[first:]
            _lp[0:rest] = filt[first:]
            _fractal_bass_ring[0:rest] = bass[first:]
        _ring_i = (i + n) % RING_SAMPLES
        _ring_filled = min(RING_SAMPLES, _ring_filled + n)
        _total += n


def fractal_bass_tail(count: int) -> np.ndarray:
    """Copy the newest stereo 30–150 Hz samples from the filter ring."""
    with _ring_lock:
        n = _ring_filled
        take = min(int(n), int(count))
        if take <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        i = _ring_i
        if n < RING_SAMPLES:
            return _fractal_bass_ring[n - take : n].copy()
        start = (i - take) % RING_SAMPLES
        if start < i:
            return _fractal_bass_ring[start:i].copy()
        return np.concatenate(
            [_fractal_bass_ring[start:], _fractal_bass_ring[:i]]
        )


def ring_tail(count: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Copy only the newest `count` stereo+LP samples (avoids a 16k unwrap)."""
    with _ring_lock:
        n = _ring_filled
        total = _total
        take = min(int(n), int(count))
        if take <= 0:
            z = np.zeros((0, 2), dtype=np.float32)
            return z, np.zeros(0, dtype=np.float32), total
        i = _ring_i
        if n < RING_SAMPLES:
            sl = slice(n - take, n)
            return _ring[sl].copy(), _lp[sl].copy(), total
        start = (i - take) % RING_SAMPLES
        if start < i:
            sl = slice(start, i)
            return _ring[sl].copy(), _lp[sl].copy(), total
        return (
            np.concatenate([_ring[start:], _ring[:i]]),
            np.concatenate([_lp[start:], _lp[:i]]),
            total,
        )


def smooth_scope(stereo: np.ndarray, width: int, height: int) -> np.ndarray:
    """Piecewise-linear curve through every sample, one column per pixel."""
    w = max(int(width), 2)
    h = max(int(height), 2)
    b = prep_block(stereo)
    if b.shape[0] < 2:
        return np.zeros((h, w), dtype=np.uint8)
    mono = np.mean(b, axis=1)
    n = int(mono.shape[0])
    y = np.interp(np.linspace(0.0, n - 1.0, w), np.arange(n, dtype=np.float32), mono)
    ys = np.clip(h * 0.5 - y * (0.95 * h * 0.5), 0, h - 1).astype(np.int32)
    frame = np.zeros((h, w), dtype=np.uint8)
    for i in range(w - 1):
        a = int(ys[i])
        b2 = int(ys[i + 1])
        lo, hi = (b2, a) if a > b2 else (a, b2)
        frame[lo : hi + 1, i] = 255
    frame[int(ys[-1]), w - 1] = 255
    return frame


def _polyline_frame(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize an ordered series of points as gap-free DDA segments."""
    w = max(2, int(width))
    h = max(2, int(height))
    xs = np.clip(np.asarray(xs, dtype=np.int32), 0, w - 1)
    ys = np.clip(np.asarray(ys, dtype=np.int32), 0, h - 1)
    if xs.size < 2 or ys.size < 2:
        return np.zeros((h, w), dtype=np.uint8)
    dx = np.diff(xs)
    dy = np.diff(ys)
    steps = np.maximum(np.abs(dx), np.abs(dy))
    max_steps = int(np.max(steps))
    frame = np.zeros((h, w), dtype=np.uint8)
    if max_steps <= 0:
        frame[int(ys[0]), int(xs[0])] = 255
        return frame
    k = np.arange(max_steps + 1, dtype=np.float32)[None, :]
    count = np.maximum(steps, 1)[:, None]
    valid = k <= steps[:, None]
    line_x = np.rint(xs[:-1, None] + dx[:, None] * k / count).astype(np.int32)
    line_y = np.rint(ys[:-1, None] + dy[:, None] * k / count).astype(np.int32)
    frame[line_y[valid], line_x[valid]] = 255
    return frame


def xy_oscilloscope_line(block: np.ndarray, width: int, height: int) -> np.ndarray:
    """Connect successive stereo XY samples with continuous segments."""
    w = max(2, int(width))
    h = max(2, int(height))
    stereo = prep_block(block)
    if stereo.shape[0] < 2:
        return np.zeros((h, w), dtype=np.uint8)
    ys = np.rint((stereo[:, 0] + 1.0) * (h - 1) * 0.5)
    xs = np.rint((-stereo[:, 1] + 1.0) * (w - 1) * 0.5)
    return _polyline_frame(xs, ys, w, h)


def _long_bandpass(wave: np.ndarray, band_i: int) -> np.ndarray:
    """Apply a low-cost causal bandpass with reflected warm-up samples."""
    if wave.size < 2:
        return np.asarray(wave, dtype=np.float32)
    pad = min(LONG_FILTER_PAD, int(wave.size) - 1)
    filtered = np.pad(
        np.asarray(wave, dtype=np.float64),
        (pad, 0),
        mode="reflect",
    )
    hp_b, hp_a, lp_b, lp_a, poles = _long_filter_coefficients[band_i]
    filtered = lfilter(hp_b, hp_a, filtered)
    for _ in range(poles):
        filtered = lfilter(lp_b, lp_a, filtered)
    return np.asarray(filtered[pad:], dtype=np.float32)


def spectral_seismograph(width: int, height: int) -> np.ndarray:
    """Three vertical waveforms sharing Bass Harmonic Anchor's trigger."""
    w = max(6, int(width))
    h = max(2, int(height))
    frame = np.zeros((h, w), dtype=np.uint8)
    sweep = prep_block(locked_window())
    if sweep.shape[0] < 2:
        return frame
    mono = np.mean(sweep, axis=1)
    source_y = np.arange(mono.size, dtype=np.float32)
    target_y = np.linspace(0.0, float(mono.size - 1), h)
    lane_w = w / 3.0
    gutter = max(4, int(round(w * 0.015)))
    for band_i in range(len(LONG_BANDS)):
        col0 = int(round(band_i * lane_w))
        col1 = int(round((band_i + 1) * lane_w)) - 1
        if band_i > 0:
            col0 += gutter // 2
        if band_i < len(LONG_BANDS) - 1:
            col1 -= gutter - gutter // 2
        center = 0.5 * (col0 + col1)
        half = max(2.0, 0.5 * (col1 - col0) - 3.0)
        band = _long_bandpass(mono, band_i)
        trace = np.interp(target_y, source_y, band) * LONG_TRACE_GAIN
        x = np.clip(
            np.rint(center + trace * half),
            col0 + 2,
            col1 - 2,
        ).astype(np.int32)
        lo = np.minimum(x[:-1], x[1:])
        hi = np.maximum(x[:-1], x[1:])
        columns = np.arange(col0, col1 + 1, dtype=np.int32)[None, :]
        connected = (columns >= lo[:, None]) & (columns <= hi[:, None])
        lane = frame[: h - 1, col0 : col1 + 1]
        lane[connected] = 255
        frame[h - 1, int(x[-1])] = 255
    return frame


def _window_len(period: float, available: int) -> int:
    global _sticky_win
    if period >= 16:
        win = int(round(TRIG_CYCLES * period))
    else:
        win = TRIG_WINDOW
    win = int(np.clip(win, TRIG_WINDOW_MIN, TRIG_WINDOW_MAX))
    win = max(64, min(win, available - 16))
    if _sticky_win >= 64 and abs(win - _sticky_win) < 0.18 * _sticky_win:
        win = min(_sticky_win, available - 16)
    else:
        _sticky_win = win
    return win


def _smooth_period(measured: float) -> float:
    global _period
    if measured <= 0:
        return _period
    if _period <= 0:
        _period = measured
        return _period
    rel = abs(measured - _period) / _period
    if rel < 0.08:
        _period = 0.85 * _period + 0.15 * measured
    else:
        # Two matching trigger intervals are enough evidence of a new note.
        # Do not glue the tracker to the previous pitch.
        _period = measured
    return _period


def _scope_crossings(x: np.ndarray, gate: float) -> np.ndarray:
    """Return upward zero crossings armed by a real negative excursion.

    Unlike peak selection, the trigger point does not move when the relative
    strength of a synth's fundamental and crunchy harmonics changes.
    """
    if x.size < 3:
        return np.empty(0, dtype=np.int64)
    zeros = np.flatnonzero((x[:-1] <= 0.0) & (x[1:] > 0.0)).astype(np.int64) + 1
    if zeros.size == 0:
        return zeros
    hits: list[int] = []
    start = 0
    for crossing in zeros:
        crossing = int(crossing)
        if crossing > start and float(np.min(x[start:crossing])) <= -gate:
            hits.append(crossing)
        start = crossing
    return np.asarray(hits, dtype=np.int64)


def _period_from_crossings(hits: np.ndarray, fs: float) -> float:
    """Use the newest pair of mutually consistent trigger intervals."""
    if hits.size < 3:
        return 0.0
    gaps = np.diff(hits.astype(np.float64))
    min_period = fs / TRIG_FMAX
    max_period = fs / TRIG_FMIN
    for end in range(int(gaps.size) - 1, 0, -1):
        a = float(gaps[end - 1])
        b = float(gaps[end])
        if not (min_period <= a <= max_period and min_period <= b <= max_period):
            continue
        center = 0.5 * (a + b)
        if abs(a - b) > TRIG_PERIOD_TOL * center:
            continue
        run = [a, b]
        for j in range(end - 2, max(-1, end - 5), -1):
            g = float(gaps[j])
            med = float(np.median(run))
            if min_period <= g <= max_period and abs(g - med) <= TRIG_PERIOD_TOL * med:
                run.append(g)
            else:
                break
        return float(np.median(run))
    return 0.0


def locked_window() -> np.ndarray:
    """Draw fresh raw audio starting at a stable filtered zero crossing."""
    stereo, filt, _total = ring_tail(TRIG_NEED)
    n = int(stereo.shape[0])
    if n < TRIG_WINDOW_MIN + 16:
        return stereo
    need = TRIG_WINDOW_MAX + TRIG_SEARCH
    if n > need:
        stereo = stereo[-need:]
        filt = filt[-need:]
        n = need
    centered = np.asarray(filt, dtype=np.float32)
    level = float(np.percentile(np.abs(centered[-4096:]), 90))
    if level < TRIG_GATE_MIN:
        return stereo[-_window_len(_period, n) :]
    gate = max(TRIG_GATE_MIN, TRIG_GATE_FRAC * level)
    hits = _scope_crossings(centered, gate)
    fs = float(lsao.SAMPLERATE)
    measured = _period_from_crossings(hits, fs)
    T = _smooth_period(measured)
    window = _window_len(T, n)
    tail = stereo[-window:]
    search_end = n - window
    if search_end <= 16 or T < 16 or hits.size == 0:
        return tail
    eligible = hits[hits <= search_end]
    if eligible.size == 0:
        return tail
    # The newest complete sweep is new raw input; no held/copied frame exists.
    i = int(eligible[-1])
    return stereo[i : i + window]


def bass_anchored_recurrence(width: int, height: int) -> np.ndarray:
    """Weighted recurrence of local shape over a phase-locked bass sweep."""
    w = max(2, min(int(width), BASS_REC_SIZE))
    h = max(2, min(int(height), BASS_REC_SIZE))
    stereo = prep_block(locked_window())
    if stereo.shape[0] < 8:
        return np.zeros((h, w), dtype=np.uint8)
    mono = np.mean(stereo, axis=1)
    base = max(w, h)
    delay = max(
        1,
        int(round(base * BASS_REC_DELAY_CYCLE / max(TRIG_CYCLES, 1.0))),
    )
    state_points = base + (BASS_REC_EMBED_DIM - 1) * delay
    source = np.arange(mono.size, dtype=np.float32)
    target = np.linspace(0.0, float(mono.size - 1), state_points)
    wave = np.interp(target, source, mono).astype(np.float32)
    wave -= float(np.mean(wave))
    level = float(np.percentile(np.abs(wave), 90))
    if level < BASS_REC_SILENCE:
        return np.zeros((h, w), dtype=np.uint8)
    activity = float(
        np.clip(
            (level - BASS_REC_SILENCE) / (BASS_REC_FULL_LEVEL - BASS_REC_SILENCE),
            0.0,
            1.0,
        )
    )
    wave = np.clip(wave / level, -3.0, 3.0)
    states = np.column_stack(
        [wave[k * delay : k * delay + base] for k in range(BASS_REC_EMBED_DIM)]
    )
    row_i = np.rint(np.linspace(0, base - 1, h)).astype(np.int32)
    col_i = np.rint(np.linspace(0, base - 1, w)).astype(np.int32)
    rows = states[row_i]
    cols = states[col_i]
    row_norm2 = np.einsum("ij,ij->i", rows, rows)
    col_norm2 = np.einsum("ij,ij->i", cols, cols)
    distance2 = row_norm2[:, None] + col_norm2[None, :] - 2.0 * (rows @ cols.T)
    distance2 = np.maximum(distance2, 0.0)
    lut_i = np.clip(
        np.rint(distance2 * _bass_rec_lut_scale),
        0,
        BASS_REC_LUT_SIZE - 1,
    ).astype(np.int32)
    brightness = _bass_rec_lut[lut_i].astype(np.float32) * activity
    return np.rint(brightness).astype(np.uint8)


def _stereo_fractal_levels(
    block: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return smoothed broadband and 30–150 Hz levels for both channels."""
    global _fractal_level_time
    stereo = prep_block(block)
    if stereo.size:
        rms = np.sqrt(np.mean(np.square(stereo[:, :2]), axis=0))
        _fractal_input_rms[:] = rms
        activity = np.maximum(rms - FRACTAL_GATE, 0.0)
        target = np.clip(activity, 0.0, 1.0)
    else:
        _fractal_input_rms[:] = 0.0
        target = np.zeros(2, dtype=np.float32)

    bass_stereo = fractal_bass_tail(FRACTAL_BASS_SAMPLES)
    if bass_stereo.shape[0] < 64:
        bass_stereo = stereo
        for _ in range(FRACTAL_BASS_LP_POLES):
            bass_stereo = lfilter(
                _fractal_bass_lp_b,
                _fractal_bass_lp_a,
                bass_stereo,
                axis=0,
            )
        bass_stereo = lfilter(
            _fractal_bass_hp_b,
            _fractal_bass_hp_a,
            bass_stereo,
            axis=0,
        )
    n = int(bass_stereo.shape[0])
    if n >= 64:
        centered = bass_stereo - np.mean(bass_stereo, axis=0, keepdims=True)
        bass_rms = np.sqrt(np.mean(np.square(centered), axis=0))
        _fractal_bass_input[:] = bass_rms
        bass_target = np.clip(
            np.maximum(bass_rms - FRACTAL_GATE, 0.0),
            0.0,
            1.0,
        )
    else:
        _fractal_bass_input[:] = 0.0
        bass_target = np.zeros(2, dtype=np.float32)

    now = time.monotonic()
    if _fractal_level_time <= 0.0:
        dt = 1.0 / max(FPS, 1)
    else:
        dt = float(np.clip(now - _fractal_level_time, 0.001, 0.1))
    _fractal_level_time = now
    tau = np.where(
        target > _fractal_levels,
        FRACTAL_ATTACK_S,
        FRACTAL_RELEASE_S,
    )
    blend = 1.0 - np.exp(-dt / tau)
    _fractal_levels[:] += (target - _fractal_levels) * blend
    bass_tau = np.where(
        bass_target > _fractal_bass_levels,
        FRACTAL_ATTACK_S,
        0.080,
    )
    bass_blend = 1.0 - np.exp(-dt / bass_tau)
    _fractal_bass_levels[:] += (
        bass_target - _fractal_bass_levels
    ) * bass_blend
    return (
        float(_fractal_levels[0]),
        float(_fractal_levels[1]),
        float(_fractal_bass_levels[0]),
        float(_fractal_bass_levels[1]),
    )


def strange_attractor(block: np.ndarray) -> np.ndarray:
    """De Jong orbit whose parameter phases advance from stereo levels."""
    global _attractor_frame, _attractor_frame_time
    now = time.monotonic()
    if (
        _attractor_frame_time > 0.0
        and now - _attractor_frame_time < FRACTAL_UPDATE_S
    ):
        return _attractor_frame
    left, right, bass_left, bass_right = _stereo_fractal_levels(block)
    dt = (
        FRACTAL_UPDATE_S
        if _attractor_frame_time <= 0.0
        else float(np.clip(now - _attractor_frame_time, 0.001, 0.1))
    )
    _attractor_frame_time = now
    phase_velocity = (
        0.08
        + 2.20 * np.array((left, right))
        + 5.50 * np.array((bass_left, bass_right))
    )
    _attractor_phase[:] = np.mod(
        _attractor_phase + dt * phase_velocity,
        2.0 * np.pi,
    )
    phase_l, phase_r = _attractor_phase
    coefficient_target = np.array(
        [
            1.40 + 0.24 * math.sin(phase_l),
            -2.30 + 0.24 * math.sin(phase_r),
            2.40,
            -2.10,
        ],
        dtype=np.float64,
    )
    coefficient_blend = 1.0 - math.exp(-dt / 0.18)
    _attractor_coefficients[:] += (
        coefficient_target - _attractor_coefficients
    ) * coefficient_blend
    a, b, c, d = _attractor_coefficients

    xs = np.empty(ATTRACTOR_POINTS, dtype=np.float32)
    ys = np.empty(ATTRACTOR_POINTS, dtype=np.float32)
    x = 0.1
    y = 0.1
    out_i = 0
    for iteration in range(ATTRACTOR_BURN_IN + ATTRACTOR_POINTS):
        next_x = math.sin(a * y) - math.cos(b * x)
        next_y = math.sin(c * x) - math.cos(d * y)
        x, y = next_x, next_y
        if iteration >= ATTRACTOR_BURN_IN:
            xs[out_i] = x
            ys[out_i] = y
            out_i += 1

    view_angle = (
        0.12 * math.sin(0.37 * phase_l + 0.29 * phase_r)
        + 0.18 * (right - left)
        + 0.24 * (bass_right - bass_left)
    )
    cos_view = math.cos(view_angle)
    sin_view = math.sin(view_angle)
    scale_x = 0.72 + 0.18 * math.sqrt(left) + 0.24 * math.sqrt(bass_left)
    scale_y = 0.72 + 0.18 * math.sqrt(right) + 0.24 * math.sqrt(bass_right)
    view_x = scale_x * (cos_view * xs - sin_view * ys)
    view_y = scale_y * (sin_view * xs + cos_view * ys)
    scale = (ATTRACTOR_SIZE - 1) / 4.0
    px = np.clip(np.rint((view_x + 2.0) * scale), 0, ATTRACTOR_SIZE - 1).astype(
        np.int32
    )
    py = np.clip(np.rint((2.0 - view_y) * scale), 0, ATTRACTOR_SIZE - 1).astype(
        np.int32
    )
    flat = py * ATTRACTOR_SIZE + px
    density = np.bincount(
        flat,
        minlength=ATTRACTOR_SIZE * ATTRACTOR_SIZE,
    ).reshape(ATTRACTOR_SIZE, ATTRACTOR_SIZE)
    current = np.sqrt(density.astype(np.float32)) * (
        50.0
        + 65.0 * np.sqrt(max(left, right))
        + 55.0 * np.sqrt(max(bass_left, bass_right))
    )
    _attractor_trail[:] *= 0.87
    np.maximum(_attractor_trail, current, out=_attractor_trail)
    _attractor_frame = np.clip(np.rint(_attractor_trail), 0, 255).astype(np.uint8)
    return _attractor_frame


def _starfield_levels(
    block: np.ndarray,
    dt: float,
) -> tuple[float, float, float]:
    """Return volume onset, high energy, and stereo balance."""
    global _starfield_volume, _starfield_kick, _starfield_high
    global _starfield_pan
    stereo = prep_block(block)
    mono = np.mean(stereo, axis=1) if stereo.size else np.zeros(0, dtype=np.float32)
    high = _long_bandpass(mono, 2)
    high_level = float(np.sqrt(np.mean(np.square(high)))) if high.size else 0.0
    volume_level = (
        float(np.sqrt(np.mean(np.square(stereo[:, :2])))) if stereo.size else 0.0
    )
    if stereo.size:
        channel_rms = np.sqrt(np.mean(np.square(stereo[:, :2]), axis=0))
        pan_target = float(
            np.clip(
                (channel_rms[1] - channel_rms[0])
                / max(float(channel_rms[0] + channel_rms[1]), 1e-5),
                -1.0,
                1.0,
            )
        )
    else:
        pan_target = 0.0
    volume_target = float(np.clip(volume_level, 0.0, 1.0))
    high_target = float(np.clip(high_level / STARFIELD_HIGH_FULL, 0.0, 1.0))
    volume_tau = 0.015 if volume_target > _starfield_volume else 0.12
    high_tau = 0.012 if high_target > _starfield_high else 0.08
    previous_volume = _starfield_volume
    _starfield_volume += (volume_target - _starfield_volume) * (
        1.0 - math.exp(-dt / volume_tau)
    )
    _starfield_kick = max(0.0, _starfield_volume - previous_volume)
    _starfield_high += (high_target - _starfield_high) * (
        1.0 - math.exp(-dt / high_tau)
    )
    _starfield_pan += (pan_target - _starfield_pan) * (
        1.0 - math.exp(-dt / 0.10)
    )
    return _starfield_kick, _starfield_high, _starfield_pan


def starfield_zoom(block: np.ndarray, width: int, height: int) -> np.ndarray:
    """Deep perspective field: volume onsets spawn, highs brighten and enlarge."""
    global _starfield_time, _starfield_warp
    global _starfield_spawn_credit
    w = max(2, int(width))
    h = max(2, int(height))
    now = time.monotonic()
    dt = (
        1.0 / max(FPS, 1)
        if _starfield_time <= 0.0
        else float(np.clip(now - _starfield_time, 0.001, 0.05))
    )
    _starfield_time = now
    kick, high, pan = _starfield_levels(block, dt)
    _starfield_warp *= math.exp(-dt / 0.13)
    _starfield_warp = max(_starfield_warp, min(0.65, 4.0 * kick))

    old_z = _starfield_z.copy()
    active_i = np.flatnonzero(_starfield_active)
    _starfield_z[active_i] -= (
        STARFIELD_SPEED
        * (1.0 + 1.20 * _starfield_volume + 0.75 * _starfield_warp)
        * dt
    )
    projection = 0.52 * min(w, h) * (1.0 + 0.25 * _starfield_warp)
    center_x = 0.5 * (w - 1) + 0.18 * w * pan
    center_y = 0.5 * (h - 1)
    world_x = _starfield_x
    world_y = _starfield_y
    if active_i.size:
        active_px = center_x + projection * world_x[active_i] / _starfield_z[
            active_i
        ]
        active_py = center_y + projection * world_y[active_i] / _starfield_z[
            active_i
        ]
        expired = (
            (_starfield_z[active_i] <= STARFIELD_NEAR_Z)
            | (active_px < 0.0)
            | (active_px >= w)
            | (active_py < 0.0)
            | (active_py >= h)
        )
        _starfield_active[active_i[expired]] = False

    active_count = int(np.count_nonzero(_starfield_active))
    _starfield_spawn_credit += kick * 90.0
    burst = int(_starfield_spawn_credit)
    _starfield_spawn_credit -= burst
    spawn_count = burst + max(0, STARFIELD_MIN_STARS - active_count)
    available = np.flatnonzero(~_starfield_active)
    spawn_i = available[:spawn_count]
    if spawn_i.size:
        _starfield_x[spawn_i] = _starfield_rng.uniform(-1.25, 1.25, spawn_i.size)
        _starfield_y[spawn_i] = _starfield_rng.uniform(-1.25, 1.25, spawn_i.size)
        _starfield_z[spawn_i] = _starfield_rng.uniform(
            3.2,
            STARFIELD_FAR_Z,
            spawn_i.size,
        )
        old_z[spawn_i] = _starfield_z[spawn_i]
        _starfield_active[spawn_i] = True

    active_i = np.flatnonzero(_starfield_active)
    current_z = _starfield_z[active_i]
    px = center_x + projection * world_x[active_i] / current_z
    py = center_y + projection * world_y[active_i] / current_z
    x1 = np.rint(px).astype(np.int32)
    y1 = np.rint(py).astype(np.int32)
    x0 = np.rint(
        center_x + projection * world_x[active_i] / old_z[active_i]
    ).astype(np.int32)
    y0 = np.rint(
        center_y + projection * world_y[active_i] / old_z[active_i]
    ).astype(np.int32)
    depth = np.clip(
        (STARFIELD_FAR_Z - current_z) / (STARFIELD_FAR_Z - STARFIELD_NEAR_Z),
        0.0,
        1.0,
    )
    proximity = np.power(depth, 0.70)
    brightness = np.clip(
        np.rint(4.0 + proximity * (20.0 + 320.0 * high)),
        0.0,
        255.0,
    ).astype(np.uint8)

    dx = x1 - x0
    dy = y1 - y0
    steps = np.maximum(np.maximum(np.abs(dx), np.abs(dy)), 1)
    max_steps = int(np.max(steps))
    k = np.arange(max_steps + 1, dtype=np.float32)[None, :]
    valid = k <= steps[:, None]
    line_x = np.rint(x0[:, None] + dx[:, None] * k / steps[:, None]).astype(
        np.int32
    )
    line_y = np.rint(y0[:, None] + dy[:, None] * k / steps[:, None]).astype(
        np.int32
    )
    line_x = np.clip(line_x, 0, w - 1)
    line_y = np.clip(line_y, 0, h - 1)
    values = np.broadcast_to(brightness[:, None], line_x.shape)
    frame = np.zeros((h, w), dtype=np.uint8)
    np.maximum.at(frame, (line_y[valid], line_x[valid]), values[valid])

    size_score = depth + high
    medium = size_score >= 0.58
    for offset_y, offset_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx = np.clip(x1[medium] + offset_x, 0, w - 1)
        ny = np.clip(y1[medium] + offset_y, 0, h - 1)
        np.maximum.at(
            frame,
            (ny, nx),
            (brightness[medium] * 0.72).astype(np.uint8),
        )
    large = size_score >= 1.12
    for offset_y, offset_x in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nx = np.clip(x1[large] + offset_x, 0, w - 1)
        ny = np.clip(y1[large] + offset_y, 0, h - 1)
        np.maximum.at(
            frame,
            (ny, nx),
            (brightness[large] * 0.52).astype(np.uint8),
        )
    return frame


def _mirror_edges(width: int, height: int) -> tuple[np.ndarray, ...]:
    """Return cached inside edge layers for a rounded CRT silhouette."""
    global _mirror_edge_shape, _mirror_edge_layers, _mirror_screen_mask
    shape = (height, width)
    if _mirror_edge_shape == shape:
        return _mirror_edge_layers

    yy, xx = np.ogrid[:height, :width]
    base_radius = max(5, int(round(min(width, height) * MIRROR_CORNER_RADIUS)))
    fills = []
    for inset in range(6):
        radius = min(
            max(1, base_radius - inset),
            max(1, (width - 2 * inset - 1) // 2),
            max(1, (height - 2 * inset - 1) // 2),
        )
        nearest_x = np.clip(xx, inset + radius, width - 1 - inset - radius)
        nearest_y = np.clip(yy, inset + radius, height - 1 - inset - radius)
        fills.append(
            (xx - nearest_x) ** 2 + (yy - nearest_y) ** 2 <= radius * radius
        )
    _mirror_edge_layers = tuple(
        fills[inset] & ~fills[inset + 1] for inset in range(5)
    )
    _mirror_screen_mask = fills[0]
    _mirror_edge_shape = shape
    return _mirror_edge_layers


def hall_of_mirrors(block: np.ndarray, width: int, height: int) -> np.ndarray:
    """RMS-lit edges recede toward a stereo- and spectrum-driven vanishing point."""
    global _mirror_frame, _mirror_time, _mirror_tilt
    global _mirror_bass, _mirror_high, _mirror_border_width
    global _mirror_scale_phase
    w = max(2, int(width))
    h = max(2, int(height))
    now = time.monotonic()
    dt = (
        1.0 / max(FPS, 1)
        if _mirror_time <= 0.0
        else float(np.clip(now - _mirror_time, 0.001, 0.05))
    )
    _mirror_time = now

    stereo = prep_block(block)
    if stereo.size:
        channel_rms = np.sqrt(np.mean(np.square(stereo[:, :2]), axis=0))
        mono = np.mean(stereo[:, :2], axis=1)
    else:
        channel_rms = np.zeros(2, dtype=np.float32)
        mono = np.zeros(0, dtype=np.float32)

    bass = fractal_bass_tail(FRACTAL_BASS_SAMPLES)
    bass_level = float(np.sqrt(np.mean(np.square(bass)))) if bass.size else 0.0
    high = _long_bandpass(mono, 2)
    high_level = float(np.sqrt(np.mean(np.square(high)))) if high.size else 0.0
    bass_target = float(np.clip(bass_level / MIRROR_BASS_FULL, 0.0, 1.0))
    high_target = float(np.clip(high_level / MIRROR_HIGH_FULL, 0.0, 1.0))
    tilt_target = float(np.clip(bass_target - high_target, -1.0, 1.0))
    level_targets = np.clip(
        (channel_rms - MIRROR_VOLUME_GATE)
        / (MIRROR_VOLUME_FULL - MIRROR_VOLUME_GATE),
        0.0,
        1.0,
    )

    level_tau = np.where(level_targets > _mirror_levels, 0.012, 0.075)
    _mirror_levels[:] += (level_targets - _mirror_levels) * (
        1.0 - np.exp(-dt / level_tau)
    )
    _mirror_tilt += (tilt_target - _mirror_tilt) * (
        1.0 - math.exp(-dt / 0.12)
    )
    _mirror_bass += (bass_target - _mirror_bass) * (
        1.0 - math.exp(-dt / 0.08)
    )
    _mirror_high += (high_target - _mirror_high) * (
        1.0 - math.exp(-dt / 0.08)
    )

    if _mirror_frame.shape != (h, w):
        _mirror_frame = np.zeros((h, w), dtype=np.float32)
    scaled_w = max(2, int(round(w * MIRROR_SHRINK)))
    scaled_h = max(2, int(round(h * MIRROR_SHRINK)))
    if (w - scaled_w) % 2:
        scaled_w += 1 if _mirror_scale_phase else -1
    if (h - scaled_h) % 2:
        scaled_h += 1 if _mirror_scale_phase else -1
    _mirror_scale_phase = not _mirror_scale_phase
    center_x = 0.5 * (w - 1)
    center_y = 0.5 * (h - 1) + 0.025 * h * _mirror_tilt
    x0 = int(round(center_x - 0.5 * (scaled_w - 1)))
    y0 = int(round(center_y - 0.5 * (scaled_h - 1)))
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(w, x0 + scaled_w)
    dy1 = min(h, y0 + scaled_h)
    next_frame = np.zeros((h, w), dtype=np.float32)
    if dx1 > dx0 and dy1 > dy0:
        sx = np.rint(
            (np.arange(dx0, dx1) - x0) * (w - 1) / (scaled_w - 1)
        ).astype(np.int32)
        sy = np.rint(
            (np.arange(dy0, dy1) - y0) * (h - 1) / (scaled_h - 1)
        ).astype(np.int32)
        next_frame[dy0:dy1, dx0:dx1] = (
            _mirror_frame[np.ix_(sy, sx)] * MIRROR_DECAY
        )

    edge_layers = _mirror_edges(w, h)
    edge_levels = _mirror_levels * _mirror_levels * (3.0 - 2.0 * _mirror_levels)
    _mirror_border_width = 2.5 + 1.5 * float(np.mean(edge_levels))
    horizontal_edge = np.linspace(
        edge_levels[0],
        edge_levels[1],
        w,
        dtype=np.float32,
    ) * 255.0
    edge_brightness = np.broadcast_to(horizontal_edge[None, :], (h, w))
    for layer_i, (layer, gain) in enumerate(
        zip(
            edge_layers,
            (1.0, 0.72, 0.50, 0.34, 0.22),
            strict=True,
        )
    ):
        coverage = float(np.clip(_mirror_border_width - layer_i, 0.0, 1.0))
        if coverage <= 0.0:
            break
        next_frame[layer] = np.maximum(
            next_frame[layer],
            edge_brightness[layer] * gain * coverage,
        )
    next_frame[~_mirror_screen_mask] = 0.0

    _mirror_frame = next_frame
    return np.clip(np.rint(_mirror_frame), 0.0, 255.0).astype(np.uint8)


def render_frame(mode: str, block: np.ndarray, width: int, height: int) -> np.ndarray:
    w, h = int(width), int(height)
    match mode:
        case "Spectrum":
            return lsao.live_spectrum(
                block, CHANNEL, w, h, SPECTRUM_XLOW, SPECTRUM_XHIGH, False, 0, 0, 0, "Filled Spectrum", 1
            )
        case "SpectrumdB":
            return lsao.live_spectrum_dB(
                block, CHANNEL, w, h, 1, 13000, -80, "Filled Spectrum", 1
            )
        case "SpecBalance":
            return lsao.live_spec_balance(block, w, h, 1, 13000, "Curve", 1)
        case "Histogram":
            return lsao.live_histogram(
                block, CHANNEL, w, h, 64, 0.1, "Flat", "Filled Histogram", 1
            )
        case "Waveform":
            tail, _, _ = ring_tail(SHORT_WINDOW)
            if tail.shape[0] < 8:
                tail = block
            return smooth_scope(tail, w, h)
        case "Bass Harmonic Anchor":
            return smooth_scope(locked_window(), w, h)
        case "Spectral Seismograph":
            return spectral_seismograph(w, h)
        case "Recurrence":
            return lsao.live_recurrence(block, CHANNEL, w, h, 0.15, 1)
        case "Bass-anchored recurrence plot":
            return bass_anchored_recurrence(w, h)
        case "Oscilloscope":
            return xy_oscilloscope_line(block, w, h)
        case "Polar":
            return lsao.live_polar(
                block, CHANNEL, w, h, 0, "C4", 1, SCOPE_DOT_THICKNESS
            )
        case "PolarStereo":
            return lsao.live_polar_stereo(
                block, w, h, 0, "C4", 1, SCOPE_DOT_THICKNESS
            )
        case "Poincare":
            return lsao.live_poincare(
                block, CHANNEL, w, h, 10, 1, SCOPE_DOT_THICKNESS
            )
        case "DelayEmbed":
            return lsao.live_delay_embed(
                block,
                STEREO,
                w,
                h,
                10,
                20,
                0,
                0.25,
                1,
                SCOPE_DOT_THICKNESS,
            )
        case "Strange Attractor":
            return strange_attractor(block)
        case "Starfield Zoom":
            return starfield_zoom(block, w, h)
        case "Hall of Mirrors":
            return hall_of_mirrors(block, w, h)
        case "Envelope":
            return lsao.live_envelope(block, CHANNEL, w, h, 1, "Filled Envelope", 1)
        case "Chladni":
            return lsao.live_chladni(block, STEREO, w, h, CHLADNI_PLATE, 1000, 0.2, 0.5, 1)
        case _:
            return np.zeros((h, w), dtype=np.uint8)


def audio_backend() -> str:
    """alsa (default) or pipewire. Revert: echo pipewire > ~/.config/lsao/backend"""
    env = os.environ.get("OSC_AUDIO_BACKEND", "").strip().lower()
    if env in {"alsa", "pipewire", "pulse"}:
        return "pipewire" if env == "pulse" else env
    try:
        text = BACKEND_FILE.read_text().strip().splitlines()[0].strip().lower()
    except OSError:
        text = ""
    if text in {"alsa", "pipewire", "pulse"}:
        return "pipewire" if text == "pulse" else text
    return "alsa"


def find_alsa_device() -> str | None:
    override = os.environ.get("OSC_ALSA_DEVICE", "").strip()
    if override:
        return override
    try:
        lines = Path("/proc/asound/cards").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        low = line.lower()
        if "usb-audio" in low or "usb audio" in low or " usb" in low:
            try:
                return f"plughw:{int(line.strip().split()[0])},0"
            except (ValueError, IndexError):
                continue
    return None


def _stop_audio_proc() -> None:
    global _audio_proc
    proc = _audio_proc
    _audio_proc = None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.kill()
        proc.wait(timeout=1)
    except Exception:
        pass


def _spawn_capture(cmd: list[str]) -> subprocess.Popen | None:
    global _audio_proc
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return None
    time.sleep(0.2)
    if proc.poll() is not None:
        return None
    _audio_proc = proc
    print(f"audio capture: {' '.join(cmd)}", flush=True)
    return proc


def start_audio() -> None:
    import atexit

    atexit.register(_stop_audio_proc)
    BACKEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not BACKEND_FILE.exists():
        BACKEND_FILE.write_text("alsa\n")
    backend = audio_backend()
    print(
        f"audio backend={backend} (revert: echo pipewire > {BACKEND_FILE})",
        flush=True,
    )
    if backend == "alsa":
        threading.Thread(target=_alsa_then_pulse, daemon=True).start()
        return
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    os.environ.setdefault("PULSE_SERVER", f"unix:{runtime}/pulse/native")
    threading.Thread(target=_pulse_pump, daemon=True).start()


def _alsa_then_pulse() -> None:
    if _alsa_pump():
        return
    print("alsa capture failed; falling back to pipewire", flush=True)
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    os.environ.setdefault("PULSE_SERVER", f"unix:{runtime}/pulse/native")
    _pulse_pump()


def _alsa_pump() -> bool:
    """Direct stereo USB ALSA capture. Returns False if arecord never started."""
    global latest_block
    device = find_alsa_device()
    if not device:
        print("alsa: no USB capture card in /proc/asound/cards", flush=True)
        return False
    rate = lsao.SAMPLERATE
    channels = 2
    cmd = [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-c",
        str(channels),
        "-r",
        str(rate),
        "-t",
        "raw",
        "-F",
        "5000",
        "-B",
        "15000",
    ]
    proc = _spawn_capture(cmd)
    if proc is None or proc.stdout is None:
        return False
    frame_bytes = 4
    chunk = max(256, rate // 200) * frame_bytes
    try:
        while True:
            data = proc.stdout.read(chunk)
            if not data:
                break
            n = len(data) // 4
            if n < 8:
                continue
            samples = (
                np.frombuffer(data, dtype=np.int16, count=n * 2).astype(np.float32)
                / 32768.0
            )
            push_audio(samples.reshape(n, 2))
    finally:
        _stop_audio_proc()
    return True


def _pulse_pump() -> None:
    global latest_block
    # pw-cat defaults to 100ms node latency; 10ms is the main clap-to-picture win.
    latency = os.environ.get("OSC_PW_LATENCY", "10ms").strip() or "10ms"
    msec = latency.lower().removesuffix("ms")
    cmds = [
        ["pw-cat", "-r", f"--latency={latency}", "--format=f32", "--rate=48000", "--channels=2", "-"],
        ["pw-cat", "-r", "--format=f32", "--rate=48000", "--channels=2", "-"],
        ["pw-record", f"--latency={latency}", "--format=f32", "--rate=48000", "--channels=2", "-"],
        ["parec", f"--latency-msec={msec}", "--format=float32le", "--rate=48000", "--channels=2"],
        ["parec", "--format=float32le", "--rate=48000", "--channels=2"],
    ]
    proc = None
    for cmd in cmds:
        proc = _spawn_capture(cmd)
        if proc is not None:
            break
    if proc is None or proc.stdout is None:
        _portaudio_fallback()
        return
    chunk = (lsao.SAMPLERATE // FPS) * 2 * 4
    while True:
        data = proc.stdout.read(chunk)
        if not data:
            break
        n = len(data) // 8
        if n < 8:
            continue
        push_audio(np.frombuffer(data, dtype=np.float32, count=n * 2).reshape(n, 2))


def _portaudio_fallback() -> None:
    global _audio_stream
    import sounddevice as sd

    source = os.environ.get("OSC_AUDIO_SOURCE", "").strip()
    kwargs = {
        "samplerate": lsao.SAMPLERATE,
        "blocksize": max(256, lsao.SAMPLERATE // FPS),
        "dtype": "float32",
        "callback": _audio_cb,
    }
    if source:
        kwargs["device"] = source
    last_err = None
    try:
        _audio_stream = sd.InputStream(channels=2, **kwargs)
        _audio_stream.start()
        return
    except Exception as exc:
        last_err = exc
    print(f"audio start failed: {last_err}", file=sys.stderr)


def _audio_cb(indata, frames, time_info, status) -> None:
    push_audio(indata.copy())


class Kiosk:
    def __init__(self) -> None:
        self.mode_index = MODES.index(DEFAULT_MODE)
        self.last_advance = 0.0
        self._fps_n = 0
        self._fps_t = time.perf_counter()
        self._draw_ms = 0.0
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "0")
        pygame.init()
        pygame.mixer.quit()
        pygame.mouse.set_visible(False)
        pygame.event.set_allowed((pygame.QUIT, pygame.KEYDOWN))
        pygame.display.set_caption("lsao-kiosk")
        flags = pygame.FULLSCREEN | pygame.DOUBLEBUF
        self.screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), flags)
        self.clock = pygame.time.Clock()
        self._small = pygame.Surface((INTERNAL, INTERNAL))
        self._rgb = np.empty((INTERNAL, INTERNAL, 3), dtype=np.uint8)
        print(
            f"pygame display={self.screen.get_size()} internal={INTERNAL}x{INTERNAL}",
            file=sys.stderr,
            flush=True,
        )

    @property
    def mode(self) -> str:
        return MODES[self.mode_index]

    def advance(self, step: int = 1) -> None:
        self.mode_index = (self.mode_index + int(step)) % len(MODES)

    def _poll_input(self) -> bool:
        global pending_next, pending_prev
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_n:
                    request_next()
                elif event.key == pygame.K_p:
                    request_prev()
        step = 0
        if pending_next:
            pending_next = False
            step = 1
        elif pending_prev:
            pending_prev = False
            step = -1
        else:
            step = fifo_step()
        now = time.time()
        if step and (now - self.last_advance) >= NEXT_LOCKOUT_S:
            self.last_advance = now
            self.advance(step)
        return running

    def draw(self) -> None:
        block = latest_block
        mode = self.mode
        w = h = INTERNAL
        t0 = time.perf_counter()
        try:
            if block is None:
                gray = np.zeros((h, w), dtype=np.uint8)
            else:
                raw = render_frame(mode, prep_block(block), w, h)
                gray = to_internal_array(raw, w, h)
            g = np.ascontiguousarray(gray.T)
            self._rgb[:, :, 0] = g
            self._rgb[:, :, 1] = g
            self._rgb[:, :, 2] = g
            pygame.surfarray.blit_array(self._small, self._rgb)
            pygame.transform.scale(self._small, self.screen.get_size(), self.screen)
            pygame.display.flip()
        except Exception as exc:
            print(f"lsao render {mode}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return
        self._draw_ms += time.perf_counter() - t0
        self._fps_n += 1
        if self._fps_n >= 300:
            elapsed = time.perf_counter() - self._fps_t
            fractal_status = ""
            if mode == "Strange Attractor":
                fractal_status = (
                    f" rms={_fractal_input_rms[0]:.4f}/{_fractal_input_rms[1]:.4f}"
                    f" env={_fractal_levels[0]:.2f}/{_fractal_levels[1]:.2f}"
                    f" bass={_fractal_bass_levels[0]:.3f}/"
                    f"{_fractal_bass_levels[1]:.3f}"
                )
            elif mode == "Starfield Zoom":
                fractal_status = (
                    f" stars={np.count_nonzero(_starfield_active)}"
                    f" volume={_starfield_volume:.2f}"
                    f" kick={_starfield_kick:.3f}"
                    f" high={_starfield_high:.2f}"
                    f" pan={_starfield_pan:.2f}"
                    f" warp={_starfield_warp:.2f}"
                )
            elif mode == "Hall of Mirrors":
                fractal_status = (
                    f" rms={_mirror_levels[0]:.2f}/{_mirror_levels[1]:.2f}"
                    f" bass={_mirror_bass:.2f}"
                    f" high={_mirror_high:.2f}"
                    f" tilt={_mirror_tilt:.2f}"
                    f" border={_mirror_border_width:.1f}"
                )
            print(
                f"lsao fps={self._fps_n / elapsed:.1f} "
                f"avg_draw_ms={self._draw_ms / self._fps_n * 1000:.1f} "
                f"mode={mode}{fractal_status}",
                file=sys.stderr,
                flush=True,
            )
            self._fps_n = 0
            self._draw_ms = 0.0
            self._fps_t = time.perf_counter()

    def run(self) -> None:
        try:
            while self._poll_input():
                self.draw()
                self.clock.tick(FPS)
        finally:
            pygame.display.quit()


def main() -> int:
    signal.signal(signal.SIGUSR2, request_next)
    signal.signal(signal.SIGTERM, lambda *_: (_stop_audio_proc(), sys.exit(0)))
    open_affect_fifo()
    start_audio()
    Kiosk().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
