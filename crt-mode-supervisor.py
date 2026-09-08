#!/usr/bin/env python3
"""GPIO kiosk supervisor: LSaO visualizer and USB VLC.

GPIO 27 (header pin 13) to GND: toggle LSaO <-> USB video.
GPIO 22 (header pin 15) to GND: next LSaO visualizer (ignored during video).
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
HEAVY_PIXELS = 1280 * 720
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


def poke_lsao_next() -> None:
    """One FIFO pulse. The kiosk drains extra bytes as a single vis change."""
    LSAO_FIFO.parent.mkdir(parents=True, exist_ok=True)
    if not LSAO_FIFO.exists():
        os.mkfifo(LSAO_FIFO, 0o600)
    try:
        fd = os.open(LSAO_FIFO, os.O_RDWR | os.O_NONBLOCK)
        try:
            os.write(fd, b"x")
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
    return sorted(found)


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

    def rank(path: Path) -> tuple[int, str]:
        name = path.name
        if name in PREFERRED_LABELS or name.upper() == "NO_NAME":
            return (0, name.lower())
        vids = videos_on(path)
        return (1 if vids else 2, name.lower())

    usb_mounts.sort(key=rank)
    for mount in usb_mounts:
        vids = videos_on(mount)
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


def video_is_heavy(src: Path) -> bool:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0", str(src),
            ],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = [p for p in out.replace("|", ",").split(",") if p]
        w, h = int(parts[0]), int(parts[1])
        return (w * h) > HEAVY_PIXELS
    except Exception:
        return True


def transcode_to_cache(src: Path) -> Path | None:
    dest = cache_path(src)
    if dest.is_file() and dest.stat().st_size > 2048:
        return dest
    if not video_is_heavy(src):
        return src
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    logging.info("transcoding %s for CRT cache", src.name)
    cmd = [
        "nice", "-n", "15",
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-i", str(src),
        "-vf", "scale=720:480:force_original_aspect_ratio=decrease,pad=720:480:(ow-iw)/2:(oh-ih)/2",
        "-r", "24",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "fastdecode",
        "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p",
        "-crf", "28",
        "-c:a", "aac", "-ac", "2", "-ar", "44100", "-b:a", "96k",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.replace(dest)
        logging.info("cache ready: %s", dest)
        return dest
    except Exception as exc:
        logging.warning("transcode failed for %s: %s", src.name, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def playable_path(src: Path) -> tuple[Path | None, str]:
    """Return (path, kind) where kind is full, proxy, or none."""
    full = cache_path(src)
    if full.is_file() and full.stat().st_size > 2048:
        return full, "full"
    proxy = full.with_suffix(".proxy.mp4")
    if proxy.is_file() and proxy.stat().st_size > 2048:
        return proxy, "proxy"
    if not video_is_heavy(src):
        return src, "original"
    return None, "none"


def transcode_proxy(src: Path) -> Path | None:
    """Fast keyframe-only 720x480 proxy so VLC can start playing."""
    dest = cache_path(src).with_suffix(".proxy.mp4")
    if dest.is_file() and dest.stat().st_size > 2048:
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    logging.info("building CRT proxy for %s", src.name)
    cmd = [
        "nice", "-n", "5",
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-skip_frame", "nokey",
        "-i", str(src),
        "-an",
        "-vf", "scale=720:480:force_original_aspect_ratio=decrease,pad=720:480:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-crf", "32",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.replace(dest)
        logging.info("proxy ready: %s", dest)
        return dest
    except Exception as exc:
        logging.warning("proxy failed for %s: %s", src.name, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def vlc_command(files: list[str], *, heavy: bool) -> list[str]:
    # Let VLC pick Wayland (labwc). Forcing xcb+canvas made 480p .mov
    # fail with "video output creation failed" and a blank CRT.
    del heavy
    return [
        "cvlc",
        "--intf", "dummy",
        "--fullscreen",
        "--no-video-deco",
        "--loop",
        "--no-osd",
        "--no-video-title-show",
        "--no-qt-privacy-ask",
        "--aspect-ratio", "4:3",
        "--aout", "pulse",
    ] + files


def kill_proc(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        logging.warning("force-killing %s", name)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=2)


def spawn(cmd: list[str], name: str, cwd: Path | None = None) -> subprocess.Popen:
    logging.info("starting %s: %s", name, " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    env = os.environ.copy()
    env.setdefault("GDK_BACKEND", "x11")
    return subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=open(LOG_PATH, "a"),
        env=env,
        cwd=str(cwd) if cwd else None,
    )


class ModeSupervisor:
    def __init__(self) -> None:
        self.mode = "lsao"
        self.vlc: subprocess.Popen | None = None
        self.lsao: subprocess.Popen | None = None
        self.last_switch = 0.0
        self.last_affect = 0.0
        self.busy = False
        self.pending_toggle = False
        self.pending_affect = False
        self.affect_held = False
        self.playing_cached = False
        self.current_sources: list[Path] = []
        self.stop_transcode = threading.Event()
        self.transcode_thread = threading.Thread(
            target=self._transcode_loop, name="crt-transcode", daemon=True
        )
        self.transcode_thread.start()

    def request_toggle(self) -> None:
        now = time.time()
        if self.pending_toggle or (now - self.last_switch) < SWITCH_LOCKOUT_S:
            return
        self.pending_toggle = True

    def request_affect(self) -> None:
        # Momentary: one vis change per hold. Next press works as soon as it is released.
        if self.affect_held:
            return
        self.affect_held = True
        self.affect()

    def release_affect(self) -> None:
        self.affect_held = False

    def kill_visuals(self) -> None:
        kill_proc(self.lsao, "lsao")
        self.lsao = None
        kill_proc(self.vlc, "vlc")
        self.vlc = None
        subprocess.run(["pkill", "-x", "vlc"], check=False)
        subprocess.run(["pkill", "-x", "cvlc"], check=False)
        subprocess.run(["pkill", "-x", "osc_braille"], check=False)
        subprocess.run(["pkill", "-f", "title=oscilloscope"], check=False)
        subprocess.run(["pkill", "-f", "LSaO-visualizer/kiosk.py"], check=False)

    def start_lsao(self) -> bool:
        if not (LSAO_DIR / "kiosk.py").is_file():
            logging.warning("LSaO kiosk missing at %s; staying on current mode", LSAO_DIR)
            return False
        self.kill_visuals()
        self.playing_cached = False
        time.sleep(0.3)
        self.lsao = spawn(lsao_command(), "lsao", cwd=LSAO_DIR)
        self.mode = "lsao"
        logging.info("mode=lsao")
        return True

    def start_video(self) -> bool:
        root, videos = find_usb_videos()
        if not videos:
            logging.warning(
                "no videos on USB (mount=%s); staying on LSaO",
                root,
            )
            return False
        kill_proc(self.lsao, "lsao")
        self.lsao = None
        subprocess.run(["pkill", "-x", "osc_braille"], check=False)
        subprocess.run(["pkill", "-f", "title=oscilloscope"], check=False)
        subprocess.run(["pkill", "-f", "LSaO-visualizer/kiosk.py"], check=False)
        time.sleep(0.3)
        subprocess.run(["pkill", "-x", "ffmpeg"], check=False)
        self.current_sources = videos
        paths: list[str] = []
        kinds: list[str] = []
        for src in videos:
            play, kind = playable_path(src)
            if play is None:
                logging.info("no CRT-sized copy yet for %s; building proxy", src.name)
                play = transcode_proxy(src)
                kind = "proxy" if play else "none"
            if play is None:
                logging.warning("cannot play %s on this Pi yet", src.name)
                continue
            paths.append(str(play))
            kinds.append(kind)
        if not paths:
            logging.warning("no playable CRT copies; staying on LSaO")
            self.start_lsao()
            return False
        self.playing_cached = all(k in {"full", "original"} for k in kinds)
        logging.info(
            "USB library %s (%d files, kinds=%s)",
            root, len(paths), ",".join(kinds),
        )
        cmd = vlc_command(paths, heavy=False)
        self.vlc = spawn(cmd, "vlc")
        self.mode = "video"
        logging.info("mode=video")
        return True

    def maybe_hotswap_cache(self) -> None:
        if self.mode != "video" or self.playing_cached or not self.current_sources:
            return
        paths: list[str] = []
        for src in self.current_sources:
            play, kind = playable_path(src)
            if kind != "full" and video_is_heavy(src):
                return
            if play is None:
                return
            paths.append(str(play))
        logging.info("full SD cache ready; restarting VLC")
        kill_proc(self.vlc, "vlc")
        subprocess.run(["pkill", "-x", "vlc"], check=False)
        subprocess.run(["pkill", "-x", "cvlc"], check=False)
        time.sleep(0.2)
        self.playing_cached = True
        self.vlc = spawn(vlc_command(paths, heavy=False), "vlc")

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

    def _transcode_loop(self) -> None:
        while not self.stop_transcode.is_set():
            if self.mode == "video" and not self.playing_cached:
                if self.stop_transcode.wait(2.0):
                    return
                continue
            try:
                _, videos = find_usb_videos()
                for src in videos:
                    if self.stop_transcode.is_set():
                        return
                    if self.mode == "video" and not self.playing_cached:
                        break
                    if cache_path(src).is_file():
                        continue
                    if not video_is_heavy(src):
                        continue
                    if not cache_path(src).with_suffix(".proxy.mp4").is_file():
                        transcode_proxy(src)
                    transcode_to_cache(src)
            except Exception:
                logging.exception("transcode loop error")
            if self.stop_transcode.wait(10.0):
                return

    def affect(self) -> None:
        if self.mode == "lsao" and self.lsao is not None and self.lsao.poll() is None:
            poke_lsao_next()
            logging.info("lsao: next visualizer")
        else:
            logging.info("affect ignored in mode=%s", self.mode)

    def reap(self) -> None:
        if self.pending_toggle:
            self.pending_toggle = False
            self.toggle()
        if self.busy:
            return
        if self.mode == "video":
            self.maybe_hotswap_cache()
        if self.mode == "lsao" and self.lsao is not None and self.lsao.poll() is not None:
            logging.warning("lsao exited (%s); restarting", self.lsao.returncode)
            self.start_lsao()
        elif self.mode == "video" and self.vlc is not None and self.vlc.poll() is not None:
            logging.warning("vlc exited (%s); returning to LSaO", self.vlc.returncode)
            self.start_lsao()


def main() -> int:
    setup_logging()
    wait_for_session()
    _pid_lock_fd = acquire_pidfile()  # flock held until this process exits
    logging.info("crt-mode supervisor starting (GPIO %s mode, GPIO %s affect)", GPIO_PIN, AFFECT_PIN)
    ensure_usb_mounted()
    sup = ModeSupervisor()
    sup.start_lsao()

    button = Button(GPIO_PIN, pull_up=True, bounce_time=BOUNCE_S)
    button.when_pressed = sup.request_toggle
    affect = Button(AFFECT_PIN, pull_up=True, bounce_time=BOUNCE_S)
    affect.when_pressed = sup.request_affect
    affect.when_released = sup.release_affect
    signal.signal(signal.SIGUSR1, lambda *_: sup.request_toggle())
    signal.signal(signal.SIGUSR2, lambda *_: sup.affect())
    logging.info(
        "GPIO %s toggles LSaO/VLC; GPIO %s cycles LSaO visualizers",
        GPIO_PIN, AFFECT_PIN,
    )

    try:
        while True:
            sup.reap()
            time.sleep(0.05)
    except KeyboardInterrupt:
        logging.info("stopping")
    finally:
        sup.stop_transcode.set()
        kill_proc(sup.vlc, "vlc")
        kill_proc(sup.lsao, "lsao")
        button.close()
        affect.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
