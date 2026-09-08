#!/usr/bin/env python3
"""Fullscreen CRT kiosk for LSaO live visualizers.

Renders each frame at a reduced internal resolution, then nearest-neighbor
scales to 720x480 at 60 fps (NTSC 60i CRT). SIGUSR2 cycles visualizer types.
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

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

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

# Every vis renders at 240x240, then nearest-neighbor stretched to fill the CRT.
# Order matches the upstream LSaO menu. Chladni's six plate formulas are
# separate affect steps (Sine through Cosecant).
INTERNAL = 240
CHLADNI_MODES = ["Sine", "Cosine", "Tangent", "Cotangent", "Secant", "Cosecant"]
MODES = [
    "Spectrum",
    "SpectrumdB",
    "SpecBalance",
    "Histogram",
    "Waveform",
    "LongWaveform",
    "Recurrence",
    "Oscilloscope",
    "Polar",
    "PolarStereo",
    "Poincare",
    "DelayEmbed",
    *[f"Chladni-{name}" for name in CHLADNI_MODES],
    "Envelope",
]
RENDER_SIZE = {name: (INTERNAL, INTERNAL) for name in MODES}
MIN_DIM = INTERNAL
DEFAULT_MODE = "Waveform"  # Short Waveform

pending_next = False
latest_block: np.ndarray | None = None
_audio_stream = None
NEXT_LOCKOUT_S = 0.08
AFFECT_FIFO = Path.home() / ".local/state" / "lsao-next"
_fifo_fd: int | None = None


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


def _nearest():
    return getattr(Image, "Resampling", Image).NEAREST


def fill_screen(img: Image.Image, width: int, height: int) -> Image.Image:
    """Nearest-neighbor stretch so every vis stays 240p on the CRT."""
    w = max(int(width), 1)
    h = max(int(height), 1)
    return img.convert("L").resize((w, h), _nearest())


def to_internal_frame(raw: np.ndarray, width: int, height: int) -> Image.Image:
    """Force a 240x240 raster even if a renderer returned another shape."""
    img = Image.fromarray(as_image_array(raw, width, height), mode="L")
    return img.resize((width, height), _nearest())


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
        case _ if mode.startswith("Chladni-"):
            plate = mode.split("-", 1)[1]
            return lsao.live_chladni(block, STEREO, w, h, plate, 1000, 0.2, 0.5, 1)
        case "Chladni":
            return lsao.live_chladni(block, STEREO, w, h, "Cosine", 1000, 0.2, 0.5, 1)
        case _:
            return np.zeros((h, w), dtype=np.uint8)


def start_audio() -> None:
    """Capture the PipeWire/Pulse default source (USB mic), not ALSA playback."""
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    os.environ.setdefault("PULSE_SERVER", f"unix:{runtime}/pulse/native")
    threading.Thread(target=_pulse_pump, daemon=True).start()


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
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            continue
        time.sleep(0.15)
        if proc.poll() is not None:
            continue
        print(f"audio capture: {' '.join(cmd[:-1])} -", flush=True)
        break
    if proc is None or proc.stdout is None or proc.poll() is not None:
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
        latest_block = np.frombuffer(data, dtype=np.float32, count=n * 2).reshape(n, 2)


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
    for channels in (2, 1):
        try:
            _audio_stream = sd.InputStream(channels=channels, **kwargs)
            _audio_stream.start()
            return
        except Exception as exc:
            last_err = exc
    print(f"audio start failed: {last_err}", file=sys.stderr)


def _audio_cb(indata, frames, time_info, status) -> None:
    global latest_block
    latest_block = stereo_block(indata.copy())


class Kiosk:
    def __init__(self) -> None:
        self.mode_index = MODES.index(DEFAULT_MODE)
        self.root = tk.Tk()
        self.root.title("lsao-kiosk")
        self.root.configure(bg="black")
        self.root.geometry(f"{DISPLAY_W}x{DISPLAY_H}+0+0")
        try:
            self.root.attributes("-fullscreen", True)
        except tk.TclError:
            pass
        try:
            self.root.config(cursor="none")
        except tk.TclError:
            pass
        self.win_w = DISPLAY_W
        self.win_h = DISPLAY_H
        self.label = tk.Label(
            self.root,
            bg="black",
            borderwidth=0,
            highlightthickness=0,
            anchor="nw",
        )
        self.label.place(x=0, y=0, relwidth=1, relheight=1)
        self.photo = None
        self.last_draw = 0.0
        self.last_advance = 0.0
        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("<Key-n>", lambda e: request_next())
        self.root.bind("<Key-q>", lambda e: self.root.destroy())
        self.tick()

    def on_resize(self, event) -> None:
        if event.widget is self.root and event.width > 1 and event.height > 1:
            self.win_w = event.width
            self.win_h = event.height

    @property
    def mode(self) -> str:
        return MODES[self.mode_index]

    def advance(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(MODES)

    def tick(self) -> None:
        global pending_next
        pulsed = fifo_has_pulse()
        if pending_next:
            pending_next = False
            pulsed = True
        now = time.time()
        if pulsed and (now - self.last_advance) >= NEXT_LOCKOUT_S:
            self.last_advance = now
            self.advance()
        if now - self.last_draw >= 1.0 / FPS:
            self.draw()
            self.last_draw = now
        self.root.after(10, self.tick)

    def draw(self) -> None:
        block = latest_block
        if block is None:
            return
        mode = self.mode
        w = h = INTERNAL
        try:
            raw = render_frame(mode, prep_block(block), w, h)
            img = to_internal_frame(raw, w, h)
            shown = fill_screen(img, self.win_w, self.win_h)
            self.photo = ImageTk.PhotoImage(shown)
            self.label.config(image=self.photo)
        except Exception as exc:
            print(f"lsao render {mode}: {type(exc).__name__}: {exc}", file=sys.stderr)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    signal.signal(signal.SIGUSR2, request_next)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    open_affect_fifo()
    start_audio()
    Kiosk().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
