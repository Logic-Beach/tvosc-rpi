# TVOSC-RPI

A Raspberry Pi kiosk for an analog CRT: live audio visualizers plus looping
video from a USB drive.

The live visuals are based on
[Aaron F. Bianchi's LSaO-visualizer](https://github.com/aaronfbianchi/LSaO-visualizer)
(GPL-3). The kiosk-specific renderer uses pygame; VLC handles USB video.

## Reference installation

The working installation this repository mirrors is:

- Raspberry Pi 3B+, 64-bit
- Debian 13 (Trixie), Raspberry Pi desktop packages
- LightDM automatically logging in user `pi`
- `rpd-labwc` Wayland session
- Composite output `Composite-1`, 720×480 at 59.998 Hz
- Stereo USB audio capture at 48 kHz
- GPIO buttons using gpiozero's internal pull-ups
- No keyboard, mouse, or desktop panel required

The Pi boots into LSaO. GPIO 27 changes between LSaO and USB video. GPIO 22
changes the active LSaO visualization.

## How the pieces fit together

1. LightDM automatically opens the `pi` user's labwc session.
2. `~/.config/labwc/autostart` starts `/home/pi/bin/crt-mode-supervisor.py`.
3. The supervisor starts `/home/pi/LSaO-visualizer/kiosk.py` by default.
4. The kiosk captures the first available USB ALSA capture card and renders a
   300×300 grayscale frame.
5. pygame nearest-neighbor scales that frame directly into a fullscreen
   720×480 surface. A square internal frame is intentionally stretched to fill
   the 4:3 CRT.
6. GPIO 22 sends `n` or `p` through `~/.local/state/lsao-next`, a named FIFO
   read by the kiosk.
7. GPIO 27 stops the current process and either starts LSaO or scans a mounted
   USB drive and starts fullscreen VLC.
8. The supervisor reaps and restarts LSaO if the renderer exits.

The audio trigger and the displayed audio are separate. For example, Bass
Harmonic Anchor filters a hidden trigger signal but displays fresh, unfiltered
audio. Spectral Seismograph analyzes three bands without automatic gain:
30–120 Hz, 180–1800 Hz, and 2.5–20 kHz. Bass-anchored recurrence plot compares
three-sample delay states over a fresh, phase-locked two-cycle bass sweep.

## Repository and installed paths

The repository is the source tree. The current Pi runs installed copies at
fixed paths expected by the supervisor:

- `LSaO-visualizer/` → `/home/pi/LSaO-visualizer/`
- `crt-mode-supervisor.py` → `/home/pi/bin/crt-mode-supervisor.py`
- `packaging/10-pi-udisks-mount.rules` →
  `/etc/polkit-1/rules.d/10-pi-udisks-mount.rules`
- Python environment → `/home/pi/LSaO-visualizer/venv/`
- Runtime log → `/home/pi/.local/state/crt-mode.log`
- PID lock → `/home/pi/.local/state/crt-mode.pid`
- Video cache → `/home/pi/.cache/crt-videos/`

Keeping these paths stable matters because they are encoded in
`crt-mode-supervisor.py` and the labwc autostart file.

## Hardware wiring

Both controls are ordinary momentary switches wired from a GPIO to ground.
gpiozero enables the pull-up resistors, so external resistors are unnecessary.

- Mode: BCM GPIO 27, physical pin 13
- Affect: BCM GPIO 22, physical pin 15
- Convenient ground: physical pin 14
- Composite output: Pi composite jack to the CRT, directly or through a
  composite-to-UHF modulator
- Audio: class-compliant USB capture interface or USB microphone

Controls:

- GPIO 27 press: toggle LSaO ↔ USB video
- GPIO 22 short press: next visualization
- GPIO 22 hold for 0.4 seconds: previous visualization
- Affect presses are ignored while USB video is active

## Install the software

Install the packages used by the reference Pi:

```sh
sudo apt update
sudo apt install -y \
  git lightdm labwc \
  python3-venv python3-numpy python3-scipy python3-pygame \
  python3-pil python3-tk python3-gpiozero \
  libportaudio2 alsa-utils \
  ffmpeg vlc udisks2
```

Clone the repository and install its runtime files:

```sh
git clone https://github.com/Logic-Beach/tvosc-rpi.git /home/pi/TVOSC-RPI
cd /home/pi/TVOSC-RPI

install -d /home/pi/LSaO-visualizer /home/pi/bin
cp -a LSaO-visualizer/. /home/pi/LSaO-visualizer/
install -m 755 crt-mode-supervisor.py /home/pi/bin/crt-mode-supervisor.py

python3 -m venv --system-site-packages /home/pi/LSaO-visualizer/venv
/home/pi/LSaO-visualizer/venv/bin/pip install sounddevice
```

The kiosk normally captures through `arecord`; `sounddevice` is still needed
because the imported upstream LSaO module imports it and it provides the last
audio fallback.

Select direct ALSA capture:

```sh
install -d /home/pi/.config/lsao
printf 'alsa\n' > /home/pi/.config/lsao/backend
```

No AudioBox or other product name is hardcoded. The kiosk reads
`/proc/asound/cards`, selects a USB capture device, and opens
`plughw:<card>,0` as 48 kHz stereo. Its current ALSA buffering is a 5 ms period
and 15 ms buffer.

## Configure composite video

The reference Pi has these effective settings in
`/boot/firmware/config.txt`:

```ini
display_auto_detect=0
enable_tvout=1
dtoverlay=vc4-fkms-v3d
max_framebuffers=2
disable_fw_kms_setup=1
arm_64bit=1
disable_overscan=1
```

Reboot after changing firmware display settings:

```sh
sudo reboot
```

Once labwc is running, verify the active output:

```sh
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 wlr-randr
```

The reference output reports `Composite-1` at 720×480 and 59.998 Hz.

## Configure LightDM and labwc

LightDM is enabled with `graphical.target` and these values in
`/etc/lightdm/lightdm.conf`:

```ini
[Seat:*]
user-session=rpd-labwc
autologin-user=pi
autologin-session=rpd-labwc
```

The kiosk deliberately runs without libinput devices. Create
`/etc/systemd/system/lightdm.service.d/wlr-no-devices.conf`:

```ini
[Service]
Environment=WLR_LIBINPUT_NO_DEVICES=1
```

Then reload systemd:

```sh
sudo systemctl daemon-reload
sudo systemctl enable lightdm
sudo systemctl set-default graphical.target
```

The relevant `/home/pi/.config/labwc/environment` settings are:

```ini
XCURSOR_THEME=PiXtrix
XCURSOR_SIZE=24
XCURSOR_PATH=/home/pi/.icons:/home/pi/.local/share/icons:/usr/share/icons
WLR_LIBINPUT_NO_DEVICES=1
```

Create `/home/pi/.config/labwc/autostart`:

```sh
#!/bin/sh
/home/pi/bin/crt-mode-supervisor.py &
```

Make it executable:

```sh
chmod 755 /home/pi/.config/labwc/autostart
```

Merge these window rules into
`/home/pi/.config/labwc/rc.xml` under `<windowRules>`. They remove decorations,
force fullscreen, and keep the pointer off the CRT:

```xml
<windowRule identifier="vlc" serverDecoration="no">
  <action name="ToggleFullscreen"/>
  <action name="HideCursor"/>
  <action name="WarpCursor" to="output" x="-1" y="-1"/>
</windowRule>
<windowRule identifier="org.videolan.VLC" serverDecoration="no">
  <action name="ToggleFullscreen"/>
  <action name="HideCursor"/>
  <action name="WarpCursor" to="output" x="-1" y="-1"/>
</windowRule>
<windowRule title="lsao-kiosk" serverDecoration="no">
  <action name="ToggleFullscreen"/>
  <action name="HideCursor"/>
  <action name="WarpCursor" to="output" x="-1" y="-1"/>
</windowRule>
```

## Configure USB mounting

The desktop automounter is not required. The supervisor discovers unmounted
USB partitions and calls `udisksctl mount` itself. Install the included polkit
rule so this does not require an interactive password:

```sh
sudo install -m 644 packaging/10-pi-udisks-mount.rules \
  /etc/polkit-1/rules.d/10-pi-udisks-mount.rules
```

USB video behavior:

- Drives normally appear below `/media/pi/`
- Directories are searched recursively
- Hidden files and macOS `._*` files are ignored
- Common VLC video extensions are accepted
- VLC loops all discovered files fullscreen at 4:3
- Video larger than 1280×720 is converted to a 720×480 H.264 cache
- The first conversion can start with a quick proxy while the complete cache
  is built in the background

Set `CRT_VIDEO_DIR` to use a fixed directory instead of removable media.

## Start and verify

Reboot for the complete boot path, or start the supervisor from an existing
labwc session:

```sh
/home/pi/bin/crt-mode-supervisor.py &
```

Useful checks:

```sh
tail -f /home/pi/.local/state/crt-mode.log
pgrep -af '[c]rt-mode-supervisor.py'
pgrep -af '[L]SaO-visualizer/kiosk.py'
arecord -l
```

Manual controls without the GPIO buttons:

```sh
# Next and previous visualization
printf n > /home/pi/.local/state/lsao-next
printf p > /home/pi/.local/state/lsao-next

# Toggle LSaO / USB video through the supervisor
kill -USR1 "$(cat /home/pi/.local/state/crt-mode.pid)"
```

## Updating an installed Pi

After pulling repository changes, copy only the runtime files that changed:

```sh
cd /home/pi/TVOSC-RPI
git pull
cp LSaO-visualizer/kiosk.py LSaO-visualizer/main.py \
  /home/pi/LSaO-visualizer/
install -m 755 crt-mode-supervisor.py /home/pi/bin/crt-mode-supervisor.py
```

To restart only LSaO, kill its exact process; the supervisor will launch it
again:

```sh
pid="$(pgrep -n -f '[L]SaO-visualizer/kiosk.py')"
test -n "$pid" && kill "$pid"
```

Restart the graphical session after changing labwc, LightDM, or display
configuration:

```sh
sudo systemctl restart lightdm
```

## Runtime overrides

- `OSC_FPS`: visualizer frame cap; default `60`
- `OSC_AUDIO_BACKEND`: `alsa` or `pipewire`
- `OSC_ALSA_DEVICE`: explicit ALSA device such as `plughw:1,0`
- `OSC_AUDIO_SOURCE`: PortAudio fallback source
- `OSC_PW_LATENCY`: PipeWire capture latency; default `10ms`
- `CRT_VIDEO_DIR`: fixed video directory instead of USB discovery

The primary operational log is `/home/pi/.local/state/crt-mode.log`.
