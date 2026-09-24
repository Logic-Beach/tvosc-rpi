#!/usr/bin/env python3
"""GPIO kiosk supervisor: resident LSaO visualizer and USB VLC.

GPIO 27 (header pin 13) to GND: toggle LSaO <-> USB video.
GPIO 22 (header pin 15) to GND: short press next, hold to go back. In LSaO,
a five-second hold toggles ten-second automatic cycling.
Boot default is LSaO.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from gpiozero import Button
except ImportError:
    print("python3-gpiozero is required", file=sys.stderr)
    raise

HOME = Path.home()
MEDIA_ROOT = Path("/media/pi")
CACHE_DIR = HOME / ".cache" / "crt-videos"
LOG_PATH = HOME / ".local/state/crt-mode.log"
PIDFILE = HOME / ".local/state/crt-mode.pid"
LSAO_FIFO = HOME / ".local/state/lsao-next"
LSAO_READY = HOME / ".local/state/lsao-ready"
PREFERRED_LABELS = ("NO_NAME", "no_name", "NONAME")
VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".mpg", ".mpeg",
    ".m4v", ".wmv", ".flv", ".ts", ".m2ts", ".vob", ".ogv",
}
GPIO_PIN = 27
AFFECT_PIN = 22
BOUNCE_S = 0.03
SWITCH_LOCKOUT_S = 1.0
AFFECT_LOCKOUT_S = 0.08
AFFECT_HOLD_S = 0.4
AFFECT_AUTO_HOLD_S = 5.0
AUTO_CYCLE_S = 10.0
LSAO_DIR = HOME / "LSaO-visualizer"


def lsao_command() -> list[str]:
    venv_py = LSAO_DIR / "venv" / "bin" / "python3"
    python = str(venv_py) if venv_py.is_file() else "python3"
    return [python, str(LSAO_DIR / "kiosk.py")]


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )


def acquire_pidfile() -> int:
    """Keep a single supervisor: a second copy would double every GPIO press."""
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(PIDFILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.error("supervisor already running; exiting")
        sys.exit(0)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def poke_lsao(step: bytes) -> None:
    """Pulse the kiosk FIFO: b'n' next, b'p' previous."""
    LSAO_FIFO.parent.mkdir(parents=True, exist_ok=True)
    if not LSAO_FIFO.exists():
        os.mkfifo(LSAO_FIFO, 0o600)
    try:
        fd = os.open(LSAO_FIFO, os.O_RDWR | os.O_NONBLOCK)
        try:
            os.write(fd, step)
        finally:
            os.close(fd)
    except OSError as exc:
        logging.warning("lsao affect fifo: %s", exc)


def wait_for_session(timeout: float = 30.0) -> None:
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))
    os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    os.environ.setdefault("PULSE_SERVER", f"unix:{runtime}/pulse/native")
    if Path("/tmp/.X11-unix/X0").exists():
        os.environ.setdefault("DISPLAY", ":0")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (runtime / "wayland-0").exists() or (runtime / "pulse/native").exists():
            return
        time.sleep(0.25)


def is_usb_mount(path: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["findmnt", "-n", "-o", "SOURCE", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    if not out:
        return False
    src = Path(out)
    name = src.name
    sysfs = Path("/sys/class/block") / name
    if not sysfs.exists():
        return False
    probe = sysfs
    for _ in range(6):
        if (probe / "device" / "subsystem").exists():
            try:
                sub = os.readlink(probe / "device" / "subsystem")
            except OSError:
                sub = ""
            if sub.endswith("/usb") or "/usb" in sub:
                return True
        if (probe / "removable").is_file():
            try:
                if probe.joinpath("removable").read_text().strip() == "1":
                    break
            except OSError:
                pass
        probe = probe.parent
    try:
        tran = subprocess.check_output(
            ["lsblk", "-no", "TRAN", out],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return any(t.strip() == "usb" for t in tran)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return path.is_relative_to(MEDIA_ROOT)


def unmounted_usb_partitions() -> list[str]:
    """USB partitions that are present but not mounted (no desktop automounter)."""
    try:
        data = json.loads(
            subprocess.check_output(
                ["lsblk", "-J", "-p", "-o", "NAME,TYPE,TRAN,MOUNTPOINT"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return []
    found: list[str] = []

    def walk(nodes: list, usb: bool = False) -> None:
        for node in nodes:
            this_usb = usb or node.get("tran") == "usb"
            if (
                this_usb
                and node.get("type") == "part"
                and not node.get("mountpoint")
                and node.get("name")
            ):
                found.append(str(node["name"]))
            walk(node.get("children") or [], this_usb)

    walk(data.get("blockdevices") or [])
    return found


def ensure_usb_mounted() -> None:
    """pcmanfm used to automount; the kiosk does it itself via udisks."""
    for dev in unmounted_usb_partitions():
        logging.info("mounting USB %s", dev)
        try:
            subprocess.run(
                ["udisksctl", "mount", "-b", dev, "--no-user-interaction"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
        except Exception as exc:
            logging.warning("usb mount %s failed: %s", dev, exc)


def iter_media_mounts() -> list[Path]:
    mounts: list[Path] = []
    if MEDIA_ROOT.is_dir():
        for child in sorted(MEDIA_ROOT.iterdir()):
            if child.is_dir() and child.is_mount():
                mounts.append(child)
    try:
        rows = subprocess.check_output(
            ["lsblk", "-nrpo", "NAME,TRAN,RM,MOUNTPOINT"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        rows = []
    disk_tran: dict[str, str] = {}
    pending: list[tuple[str, str, str, str]] = []
    for line in rows:
        parts = line.split()
        if len(parts) < 3:
            continue
        name, tran, rm = parts[0], parts[1], parts[2]
        mnt = parts[3] if len(parts) > 3 else ""
        pending.append((name, tran, rm, mnt))
        disk_tran[name] = tran
    for name, tran, rm, mnt in pending:
        if not mnt:
            continue
        inherited = tran if tran and tran != "0" else ""
        usb = inherited == "usb"
        if not usb:
            for disk, dtran in disk_tran.items():
                if name.startswith(disk) and dtran == "usb":
                    usb = True
                    break
        if usb:
            p = Path(mnt)
            if p not in mounts:
                mounts.append(p)
    return mounts


def videos_on(root: Path) -> list[Path]:
    found: list[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith(".") or name.startswith("._"):
                    continue
                if Path(name).suffix.lower() in VIDEO_EXTS:
                    found.append(Path(dirpath) / name)
    except OSError as exc:
        logging.warning("scan failed for %s: %s", root, exc)

    # The offline converter writes SMALL_name.mp4 beside name.mkv/name.webm.
    # Prefer that Pi-friendly copy without showing both versions in VLC.
    converted = {
        (path.parent, path.stem.removeprefix("SMALL_").casefold())
        for path in found
        if path.suffix.lower() == ".mp4" and path.name.startswith("SMALL_")
    }
    return sorted(
        path
        for path in found
        if path.name.startswith("SMALL_")
        or (path.parent, path.stem.casefold()) not in converted
    )


def find_usb_videos() -> tuple[Path | None, list[Path]]:
    env_dir = os.environ.get("CRT_VIDEO_DIR", "").strip()
    if env_dir:
        root = Path(env_dir)
        if root.is_dir():
            return root, videos_on(root)
        logging.warning("CRT_VIDEO_DIR is not a directory: %s", root)

    ensure_usb_mounted()
    mounts = iter_media_mounts()
    usb_mounts = [m for m in mounts if is_usb_mount(m) or m.parent == MEDIA_ROOT]
    if not usb_mounts:
        usb_mounts = mounts

    # Scan each mount once. The old ranking pass recursively scanned every
    # drive and then scanned the selected drive again.
    libraries = {mount: videos_on(mount) for mount in usb_mounts}

    def rank(path: Path) -> tuple[int, str]:
        name = path.name
        if name in PREFERRED_LABELS or name.upper() == "NO_NAME":
            return (0, name.lower())
        vids = libraries[path]
        return (1 if vids else 2, name.lower())

    usb_mounts.sort(key=rank)
    for mount in usb_mounts:
        vids = libraries[mount]
        if vids:
            return mount, vids
    if usb_mounts:
        return usb_mounts[0], []
    return None, []


def cache_path(src: Path) -> Path:
    try:
        st = src.stat()
        key = f"{src}\0{st.st_mtime_ns}\0{st.st_size}"
    except OSError:
        key = str(src)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.mp4"


def immediate_playable_path(src: Path) -> Path:
    """Choose an existing cache if available, without probing or converting."""
    if src.suffix.lower() == ".mp4" and src.name.startswith("SMALL_"):
        return src
    full = cache_path(src)
    try:
        if full.is_file() and full.stat().st_size > 2048:
            return full
    except OSError:
        pass
    proxy = full.with_suffix(".proxy.mp4")
    try:
        if proxy.is_file() and proxy.stat().st_size > 2048:
            return proxy
    except OSError:
        pass
    return src


def vlc_command(files: list[str], *, heavy: bool) -> list[str]:
    # Let VLC pick Wayland (labwc). Forcing xcb+canvas made 480p .mov
    # fail with "video output creation failed" and a blank CRT.
    del heavy
    return [
        "cvlc",
        "--intf", "dummy",
        "--fullscreen",
        "--no-video-deco",
        "--play-and-exit",
        "--no-osd",
        "--no-video-title-show",
        "--no-qt-privacy-ask",
        "--aspect-ratio", "4:3",
        "--aout", "alsa",
        "--alsa-audio-device", "plughw:0,0",
    ] + files


def kill_proc(
    proc: subprocess.Popen | None,
    name: str,
    *,
    timeout: float = 3.0,
) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.warning("force-killing %s", name)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=2)


def spawn(
    cmd: list[str],
    name: str,
    cwd: Path | None = None,
) -> subprocess.Popen:
    logging.info("starting %s: %s", name, " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    env = os.environ.copy()
    env.setdefault("GDK_BACKEND", "x11")
    log_file = open(LOG_PATH, "a")
    try:
        return subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    finally:
        log_file.close()


class ModeSupervisor:
    def __init__(self) -> None:
        self.mode = "lsao"
        self.vlc: subprocess.Popen | None = None
        self.lsao: subprocess.Popen | None = None
        self.lsao_paused = False
        self.last_switch = 0.0
        self.last_affect = 0.0
        self.busy = False
        self.pending_toggle = False
        self.pending_video_step = 0
        self.affect_held = False
        self.affect_long = False
        self.affect_pressed_at = 0.0
        self.auto_cycle = False
        self.last_auto_advance = time.monotonic()
        self.first_lsao_start = True
        self.media_root: Path | None = None
        self.current_sources: list[Path] = []
        self.video_paths: list[str] = []
        self.video_index = 0
        self.video_started_at = 0.0
        self.video_fail_streak = 0
        self.media_lock = threading.Lock()
        self.media_ready = threading.Event()
        self.waiting_for_video = False
        self.stop_media = threading.Event()
        self.media_thread = threading.Thread(
            target=self._media_loop, name="crt-media-index", daemon=True
        )
        self.media_thread.start()

    def lsao_is_ready(self) -> bool:
        proc = self.lsao
        if proc is None or proc.poll() is not None:
            return False
        try:
            return int(LSAO_READY.read_text().strip()) == proc.pid
        except (OSError, ValueError):
            return False

    def request_toggle(self) -> None:
        now = time.time()
        if self.pending_toggle or (now - self.last_switch) < SWITCH_LOCKOUT_S:
            return
        if self.waiting_for_video:
            self.waiting_for_video = False
            logging.info("pending video switch cancelled")
            return
        self.pending_toggle = True

    def request_affect(self, *_args) -> None:
        if self.affect_held:
            return
        self.affect_held = True
        self.affect_long = False
        self.affect_pressed_at = time.monotonic()

    def affect_held_cb(self, *_args) -> None:
        if not self.affect_held or self.affect_long:
            return
        # In video mode every hold means "previous", including a 5+ second
        # hold. Automatic cycling only applies to visualizers.
        if self.mode == "video":
            return
        self.affect_long = True
        self.toggle_auto_cycle()

    def release_affect(self, *_args) -> None:
        if self.affect_held and not self.affect_long:
            held_for = time.monotonic() - self.affect_pressed_at
            self.affect(-1 if held_for >= AFFECT_HOLD_S else 1)
        self.affect_held = False
        self.affect_long = False

    def toggle_auto_cycle(self) -> None:
        if self.mode != "lsao" or self.lsao is None or self.lsao.poll() is not None:
            logging.info("auto cycle toggle ignored in mode=%s", self.mode)
            return
        self.auto_cycle = not self.auto_cycle
        self.last_auto_advance = time.monotonic()
        logging.info("lsao: auto cycle %s", "on" if self.auto_cycle else "off")

    def maybe_auto_advance(self) -> None:
        if (
            not self.auto_cycle
            or self.mode != "lsao"
            or self.lsao is None
            or self.lsao.poll() is not None
        ):
            return
        if (time.monotonic() - self.last_auto_advance) >= AUTO_CYCLE_S:
            self.affect(1)

    def start_lsao(self) -> bool:
        if not (LSAO_DIR / "kiosk.py").is_file():
            logging.warning("LSaO kiosk missing at %s; staying on current mode", LSAO_DIR)
            return False
        self.waiting_for_video = False
        self.pending_video_step = 0
        if self.first_lsao_start:
            self.first_lsao_start = False
            subprocess.run(
                ["pkill", "-x", "vlc"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["pkill", "-x", "cvlc"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["pkill", "-f", "LSaO-visualizer/kiosk.py"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if self.lsao is None or self.lsao.poll() is not None:
            LSAO_READY.unlink(missing_ok=True)
            self.lsao = spawn(lsao_command(), "lsao", cwd=LSAO_DIR)
        elif self.lsao_is_ready():
            try:
                self.lsao.send_signal(signal.SIGHUP)
            except ProcessLookupError:
                LSAO_READY.unlink(missing_ok=True)
                self.lsao = spawn(lsao_command(), "lsao", cwd=LSAO_DIR)
        self.lsao_paused = False
        kill_proc(self.vlc, "vlc")
        self.vlc = None
        self.mode = "lsao"
        self.last_auto_advance = time.monotonic()
        logging.info("mode=lsao")
        return True

    def start_video(self) -> bool:
        with self.media_lock:
            root = self.media_root
            videos = list(self.current_sources)
        if not self.media_ready.is_set():
            self.waiting_for_video = True
            logging.info("USB index is still loading; video will open when ready")
            return False
        if not videos:
            logging.warning(
                "no videos on USB (mount=%s); staying on LSaO",
                root,
            )
            return False
        self.waiting_for_video = False
        paths = [str(immediate_playable_path(src)) for src in videos]
        if paths != self.video_paths:
            self.video_paths = paths
            self.video_index = 0
        else:
            self.video_index %= len(self.video_paths)
        logging.info(
            "USB library %s (%d files); opening %d: %s",
            root,
            len(paths),
            self.video_index + 1,
            Path(self.video_paths[self.video_index]).name,
        )
        cmd = vlc_command(
            [self.video_paths[self.video_index]],
            heavy=False,
        )
        try:
            self.vlc = spawn(cmd, "vlc")
            self.video_started_at = time.monotonic()
            self.video_fail_streak = 0
        except Exception:
            if self.lsao_is_ready():
                self.lsao.send_signal(signal.SIGHUP)
            raise
        if self.lsao_is_ready():
            # Keep pygame, NumPy/SciPy and the ALSA capture process warm. The
            # frozen visualizer window remains behind VLC and is ready as soon
            # as VLC exits.
            self.lsao.send_signal(signal.SIGUSR1)
            self.lsao_paused = True
        self.mode = "video"
        logging.info("mode=video")
        return True

    def toggle(self) -> None:
        now = time.time()
        if self.busy or (now - self.last_switch) < SWITCH_LOCKOUT_S:
            return
        self.busy = True
        self.last_switch = now
        try:
            if self.mode == "lsao":
                if not self.start_video():
                    logging.info("mode switch skipped; still LSaO")
            else:
                self.start_lsao()
        except Exception:
            logging.exception("mode switch failed")
        finally:
            self.busy = False

    def _media_loop(self) -> None:
        """Mount and index USB media away from the GPIO/button path."""
        while not self.stop_media.is_set():
            try:
                root, videos = find_usb_videos()
                with self.media_lock:
                    changed = (
                        root != self.media_root
                        or videos != self.current_sources
                    )
                    self.media_root = root
                    self.current_sources = videos
                self.media_ready.set()
                if changed:
                    logging.info(
                        "USB index ready: %s (%d videos)", root, len(videos)
                    )
            except Exception:
                logging.exception("USB index refresh failed")
                self.media_ready.set()
            if self.stop_media.wait(5.0):
                return

    def switch_video(self, step: int) -> bool:
        if not self.video_paths:
            return False
        next_index = (self.video_index + (-1 if step < 0 else 1)) % len(
            self.video_paths
        )
        next_path = self.video_paths[next_index]
        logging.info(
            "video: opening %d/%d: %s",
            next_index + 1,
            len(self.video_paths),
            Path(next_path).name,
        )
        kill_proc(self.vlc, "vlc", timeout=0.75)
        self.vlc = None
        try:
            self.vlc = spawn(
                vlc_command([next_path], heavy=False),
                "vlc",
            )
        except Exception:
            logging.exception("failed to start selected video")
            self.start_lsao()
            return False
        self.video_index = next_index
        self.video_started_at = time.monotonic()
        return True

    def affect(self, step: int = 1) -> None:
        now = time.time()
        if (now - self.last_affect) < AFFECT_LOCKOUT_S:
            return
        if self.mode == "lsao" and self.lsao is not None and self.lsao.poll() is None:
            self.last_affect = now
            self.last_auto_advance = time.monotonic()
            if step < 0:
                poke_lsao(b"p")
                logging.info("lsao: previous visualizer")
            else:
                poke_lsao(b"n")
                logging.info("lsao: next visualizer")
        elif self.mode == "video":
            self.last_affect = now
            self.pending_video_step = -1 if step < 0 else 1
        else:
            logging.info("affect ignored in mode=%s", self.mode)

    def reap(self) -> None:
        if self.pending_toggle:
            self.pending_toggle = False
            self.toggle()
        if (
            self.waiting_for_video
            and self.media_ready.is_set()
            and self.mode == "lsao"
            and not self.busy
            and (time.time() - self.last_switch) >= SWITCH_LOCKOUT_S
        ):
            self.waiting_for_video = False
            self.toggle()
        self.maybe_auto_advance()
        if self.busy:
            return
        if self.mode == "video" and self.pending_video_step:
            step = self.pending_video_step
            self.pending_video_step = 0
            self.switch_video(step)
        if (
            self.mode == "video"
            and not self.lsao_paused
            and self.lsao_is_ready()
        ):
            self.lsao.send_signal(signal.SIGUSR1)
            self.lsao_paused = True
        if self.mode == "lsao" and self.lsao is not None and self.lsao.poll() is not None:
            logging.warning("lsao exited (%s); restarting", self.lsao.returncode)
            self.start_lsao()
        elif self.mode == "video" and self.vlc is not None and self.vlc.poll() is not None:
            code = self.vlc.returncode
            self.vlc = None
            elapsed = time.monotonic() - self.video_started_at
            if elapsed < 2.0:
                self.video_fail_streak += 1
                logging.warning(
                    "vlc exited (%s) after %.2fs; skipping to next",
                    code,
                    elapsed,
                )
            else:
                self.video_fail_streak = 0
                logging.info("video finished; opening next")
            if (
                not self.video_paths
                or self.video_fail_streak >= len(self.video_paths)
            ):
                logging.warning("no playable videos left; returning to LSaO")
                self.start_lsao()
                return
            self.switch_video(1)


def main() -> int:
    setup_logging()
    wait_for_session()
    _pid_lock_fd = acquire_pidfile()  # flock held until this process exits
    logging.info("crt-mode supervisor starting (GPIO %s mode, GPIO %s affect)", GPIO_PIN, AFFECT_PIN)
    sup = ModeSupervisor()
    sup.start_lsao()

    button = Button(GPIO_PIN, pull_up=True, bounce_time=BOUNCE_S)
    button.when_pressed = sup.request_toggle
    affect = Button(
        AFFECT_PIN,
        pull_up=True,
        bounce_time=BOUNCE_S,
        hold_time=AFFECT_AUTO_HOLD_S,
        hold_repeat=False,
    )
    affect.when_pressed = sup.request_affect
    affect.when_held = sup.affect_held_cb
    affect.when_released = sup.release_affect
    signal.signal(signal.SIGUSR1, lambda *_: sup.request_toggle())
    signal.signal(signal.SIGUSR2, lambda *_: sup.affect(1))
    logging.info(
        "GPIO %s toggles LSaO/VLC; GPIO %s short=next, hold=previous, "
        "5s in LSaO=toggle 10s auto cycle",
        GPIO_PIN, AFFECT_PIN,
    )

    try:
        while True:
            sup.reap()
            time.sleep(0.05)
    except KeyboardInterrupt:
        logging.info("stopping")
    finally:
        sup.stop_media.set()
        kill_proc(sup.vlc, "vlc")
        kill_proc(sup.lsao, "lsao")
        button.close()
        affect.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
