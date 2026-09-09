#!/usr/bin/env python3
"""Fullscreen CRT kiosk for LSaO live visualizers.

Renders each frame at 300x300, then nearest-neighbor stretches to fill 720x480.
pygame SCALED letterboxed the square buffer; we blit onto a native 720x480
window instead. SIGUSR2 cycles visualizer types.
"""

from __future__ import annotations

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

# Every vis renders at 300x300, then nearest-neighbor stretched to fill the CRT.
# Order matches the upstream LSaO menu, with a triggered bass waveform after
# Short Waveform. Chladni is one plate (Cosine).
INTERNAL = 300
CHLADNI_PLATE = "Cosine"
MODES = [
    "Spectrum",
    "SpectrumdB",
    "SpecBalance",
    "Histogram",
    "Waveform",
    "WaveformTrig",
    "LongWaveform",
    "Recurrence",
    "Oscilloscope",
    "Polar",
    "PolarStereo",
    "Poincare",
    "DelayEmbed",
    "Chladni",
    "Envelope",
]
RENDER_SIZE = {name: (INTERNAL, INTERNAL) for name in MODES}
MIN_DIM = INTERNAL
DEFAULT_MODE = "Waveform"  # Short Waveform

pending_next = False
latest_block: np.ndarray | None = None
_audio_stream = None
_audio_proc: subprocess.Popen | None = None
NEXT_LOCKOUT_S = 0.08
AFFECT_FIFO = Path.home() / ".local/state" / "lsao-next"
BACKEND_FILE = Path.home() / ".config" / "lsao" / "backend"
_fifo_fd: int | None = None

# Triggered waveform: analog-scope rising-edge lock.
# Bass LP is only for period/holdoff so harmonics don't steal the edge.
# Group-delay is backed out, then the start snaps to the raw rising zero.
RING_SAMPLES = 16384
TRIG_WINDOW = 6000
TRIG_WINDOW_MIN = 4000
TRIG_WINDOW_MAX = 8000
TRIG_CYCLES = 5.0
TRIG_SEARCH = 2400
TRIG_NEED = TRIG_WINDOW_MAX + TRIG_SEARCH
TRIG_HYST = 0.03
TRIG_HOLDOFF = 0.82
TRIG_LP_HZ = 55.0
TRIG_FMIN = 28.0
TRIG_FMAX = 110.0
TRIG_DS = 8
_period_skip = 0
_ring = np.zeros((RING_SAMPLES, 2), dtype=np.float32)
_lp = np.zeros(RING_SAMPLES, dtype=np.float32)
_ring_i = 0
_ring_filled = 0
_ring_lock = threading.Lock()
_lp_alpha = float(1.0 - np.exp(-2.0 * np.pi * TRIG_LP_HZ / 48000.0))
_lp_b = np.array([_lp_alpha], dtype=np.float64)
_lp_a = np.array([1.0, _lp_alpha - 1.0], dtype=np.float64)
_lp_zi1 = np.zeros(1, dtype=np.float64)
_lp_zi2 = np.zeros(1, dtype=np.float64)
_total = 0
_period = 0.0


def request_next(*_args) -> None:
    global pending_next
    pending_next = True


def open_affect_fifo() -> None:
    global _fifo_fd
    AFFECT_FIFO.parent.mkdir(parents=True, exist_ok=True)
    if AFFECT_FIFO.exists() and not AFFECT_FIFO.is_fifo():
        AFFECT_FIFO.unlink()
    if not AFFECT_FIFO.exists():
        os.mkfifo(AFFECT_FIFO, 0o600)
    _fifo_fd = os.open(AFFECT_FIFO, os.O_RDWR | os.O_NONBLOCK)


def fifo_has_pulse() -> bool:
    if _fifo_fd is None:
        return False
    try:
        return bool(os.read(_fifo_fd, 256))
    except BlockingIOError:
        return False
    except OSError:
        return False


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
    """Keep a rolling stereo ring for triggered waveform (and latest_block)."""
    global latest_block, _ring_i, _ring_filled, _lp_zi1, _lp_zi2, _total
    b = stereo_block(np.asarray(block, dtype=np.float32))
    if b.size == 0:
        return
    latest_block = b
    n = min(int(b.shape[0]), RING_SAMPLES)
    b = b[-n:]
    y1, _lp_zi1 = lfilter(_lp_b, _lp_a, np.mean(b, axis=1), zi=_lp_zi1)
    y2, _lp_zi2 = lfilter(_lp_b, _lp_a, y1, zi=_lp_zi2)
    filt = np.asarray(y2, dtype=np.float32)
    with _ring_lock:
        i = _ring_i
        first = min(n, RING_SAMPLES - i)
        _ring[i : i + first] = b[:first]
        _lp[i : i + first] = filt[:first]
        rest = n - first
        if rest:
            _ring[0:rest] = b[first:]
            _lp[0:rest] = filt[first:]
        _ring_i = (i + n) % RING_SAMPLES
        _ring_filled = min(RING_SAMPLES, _ring_filled + n)
        _total += n


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


def _window_len(period: float, available: int) -> int:
    if period >= 16:
        win = int(round(TRIG_CYCLES * period))
    else:
        win = TRIG_WINDOW
    win = int(np.clip(win, TRIG_WINDOW_MIN, TRIG_WINDOW_MAX))
    return max(64, min(win, available - 16))


def _first_ac_lag(region: np.ndarray, energy: float) -> int:
    """Prefer the fundamental lag, not 2×T, when both correlate."""
    peak = float(np.max(region))
    if peak < 0.28 * energy:
        return -1
    thr = 0.65 * peak
    if region.size < 3:
        return int(np.argmax(region))
    loc = (region[1:-1] >= region[:-2]) & (region[1:-1] >= region[2:]) & (region[1:-1] >= thr)
    found = np.flatnonzero(loc)
    if found.size == 0:
        return int(np.argmax(region))
    return int(found[0]) + 1


def _estimate_period(filt: np.ndarray, fs: float) -> float:
    ds = max(1, int(TRIG_DS))
    x = np.asarray(filt[-4096:], dtype=np.float32)
    if x.size < 256:
        return 0.0
    x = x[::ds]
    x = x - float(np.mean(x))
    energy = float(np.dot(x, x))
    if energy < 1e-8:
        return 0.0
    min_lag = max(2, int(round(fs / (TRIG_FMAX * ds))))
    max_lag = min(int(x.size) - 2, int(round(fs / (TRIG_FMIN * ds))))
    if max_lag <= min_lag + 2:
        return 0.0
    ac = np.correlate(x, x, mode="full")
    mid = int(x.size) - 1
    region = ac[mid + min_lag : mid + max_lag + 1]
    k = _first_ac_lag(region, energy)
    if k < 0:
        return 0.0
    lag = float(min_lag + k)
    if 0 < k < region.size - 1:
        a, b, c = float(region[k - 1]), float(region[k]), float(region[k + 1])
        den = a - 2.0 * b + c
        if abs(den) > 1e-12:
            lag += 0.5 * (a - c) / den
    return lag * ds


def _smooth_period(measured: float) -> float:
    global _period
    if measured <= 0:
        return _period
    if _period <= 0:
        _period = measured
        return _period
    rel = abs(measured - _period) / _period
    if rel < 0.18:
        _period = 0.88 * _period + 0.12 * measured
    elif abs(measured - 2.0 * _period) / _period < 0.12:
        pass
    elif abs(2.0 * measured - _period) / _period < 0.12:
        _period = 0.6 * _period + 0.4 * measured
    else:
        _period = measured
    return _period


def _lp_delay_samples() -> int:
    a = float(_lp_alpha)
    if a <= 1e-6:
        return 0
    return int(round(2.0 * (1.0 - a) / a))


def _rising_hits(x: np.ndarray, hyst: float, end: int | None = None) -> np.ndarray:
    if end is None:
        end = int(x.size) - 1
    end = min(int(end), int(x.size) - 1)
    if end < 1:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((x[:end] < -hyst) & (x[1 : end + 1] >= hyst)).astype(np.int64)


def _holdoff_hits(hits: np.ndarray, period: float) -> np.ndarray:
    """Keep one rising edge per bass period (latest in each cluster)."""
    if hits.size == 0 or period < 16:
        return hits
    gap = max(8, int(round(TRIG_HOLDOFF * period)))
    kept: list[int] = [int(hits[0])]
    for h in hits[1:]:
        h = int(h)
        if h - kept[-1] >= gap:
            kept.append(h)
        else:
            kept[-1] = h
    return np.asarray(kept, dtype=np.int64)


def locked_window() -> np.ndarray:
    """Rising-edge lock with bass-period holdoff; start on the visible rise."""
    global _period_skip
    stereo, filt, _unused = ring_tail(TRIG_NEED)
    n = int(stereo.shape[0])
    if n < TRIG_WINDOW_MIN + 16:
        return stereo
    need = TRIG_WINDOW_MAX + TRIG_SEARCH
    if n > need:
        stereo = stereo[-need:]
        filt = filt[-need:]
        n = need
    if float(np.max(np.abs(filt))) < TRIG_HYST * 2:
        return stereo[-_window_len(0.0, n) :]
    fs = float(lsao.SAMPLERATE)
    _period_skip += 1
    if _period <= 0.0 or _period_skip % 3 == 0:
        T = _smooth_period(_estimate_period(filt, fs))
    else:
        T = _period
    window = _window_len(T, n)
    delay = min(_lp_delay_samples(), max(0, window // 5))
    search_end = n - window
    if search_end <= delay + 8:
        return stereo[-window:]
    hits = _rising_hits(filt, TRIG_HYST, search_end)
    hits = hits[hits >= delay]
    if T >= 16:
        hits = _holdoff_hits(hits, T)
    if hits.size == 0:
        return stereo[-window:]
    i = int(hits[-1]) - delay
    mono = np.mean(stereo, axis=1)
    slop = max(12, int(T * 0.12)) if T >= 16 else 32
    lo = max(0, i - slop)
    hi = min(search_end, i + slop)
    peak = float(np.max(np.abs(mono[lo : hi + 2]))) if hi > lo else 0.0
    hyst_r = max(0.015, min(TRIG_HYST, 0.12 * peak if peak > 1e-6 else TRIG_HYST))
    raw_hits = _rising_hits(mono[lo : hi + 2], hyst_r)
    if raw_hits.size:
        cand = raw_hits.astype(np.int64) + lo
        cand = cand[(cand >= 0) & (cand <= search_end)]
        if cand.size:
            i = int(cand[np.argmin(np.abs(cand - i))])
    i = int(np.clip(i, 0, search_end))
    return stereo[i : i + window]


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
            return lsao.live_waveform(block, CHANNEL, w, h, "Curve", 1)
        case "WaveformTrig":
            return smooth_scope(locked_window(), w, h)
        case "LongWaveform":
            return lsao.live_waveform_long(block, CHANNEL, w, h, 1)
        case "Recurrence":
            return lsao.live_recurrence(block, CHANNEL, w, h, 0.15, 1)
        case "Oscilloscope":
            return lsao.live_oscilloscope(block, w, h, 1, 1)
        case "Polar":
            return lsao.live_polar(block, CHANNEL, w, h, 0, "C4", 1, 1)
        case "PolarStereo":
            return lsao.live_polar_stereo(block, w, h, 0, "C4", 1, 1)
        case "Poincare":
            return lsao.live_poincare(block, CHANNEL, w, h, 10, 1, 1)
        case "DelayEmbed":
            return lsao.live_delay_embed(block, STEREO, w, h, 10, 20, 0, 0.25, 1, 1)
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

    def advance(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(MODES)

    def _poll_input(self) -> bool:
        global pending_next
        running = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_n:
                    request_next()
        pulsed = fifo_has_pulse()
        if pending_next:
            pending_next = False
            pulsed = True
        now = time.time()
        if pulsed and (now - self.last_advance) >= NEXT_LOCKOUT_S:
            self.last_advance = now
            self.advance()
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
            print(
                f"lsao fps={self._fps_n / elapsed:.1f} "
                f"avg_draw_ms={self._draw_ms / self._fps_n * 1000:.1f} mode={mode}",
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
