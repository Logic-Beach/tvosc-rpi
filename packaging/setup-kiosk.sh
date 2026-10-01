#!/bin/sh
# Install the TVOSC CRT kiosk on a Raspberry Pi (3 or 4).
# Run on the Pi as user pi, from a copy of this repository:
#   chmod 755 packaging/setup-kiosk.sh
#   ./packaging/setup-kiosk.sh
set -eu

if [ "$(id -un)" != "pi" ]; then
    echo "run this as user pi" >&2
    exit 1
fi

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ ! -f "$REPO/crt-mode-supervisor.py" ] || [ ! -f "$REPO/LSaO-visualizer/kiosk.py" ]; then
    echo "cannot find repo files under $REPO" >&2
    exit 1
fi

sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt install -y \
    git lightdm labwc \
    python3-venv python3-numpy python3-scipy python3-pygame \
    python3-pil python3-tk python3-gpiozero \
    libportaudio2 alsa-utils \
    ffmpeg vlc udisks2

install -d /home/pi/LSaO-visualizer /home/pi/bin \
    /home/pi/.config/lsao \
    /home/pi/.config/labwc \
    /home/pi/.local/state \
    /home/pi/.cache/crt-videos

cp -a "$REPO/LSaO-visualizer/." /home/pi/LSaO-visualizer/
install -m 755 "$REPO/crt-mode-supervisor.py" /home/pi/bin/crt-mode-supervisor.py
sudo install -m 644 "$REPO/packaging/10-pi-udisks-mount.rules" \
    /etc/polkit-1/rules.d/10-pi-udisks-mount.rules

python3 -m venv --system-site-packages /home/pi/LSaO-visualizer/venv
/home/pi/LSaO-visualizer/venv/bin/pip install --upgrade pip
/home/pi/LSaO-visualizer/venv/bin/pip install sounddevice

printf 'alsa\n' > /home/pi/.config/lsao/backend

# --- LightDM autologin into labwc ---
sudo tee /etc/lightdm/lightdm.conf >/dev/null <<'EOF'
[Seat:*]
user-session=rpd-labwc
autologin-user=pi
autologin-session=rpd-labwc
EOF

sudo install -d /etc/systemd/system/lightdm.service.d
sudo tee /etc/systemd/system/lightdm.service.d/wlr-no-devices.conf >/dev/null <<'EOF'
[Service]
Environment=WLR_LIBINPUT_NO_DEVICES=1
EOF

sudo systemctl daemon-reload
sudo systemctl enable lightdm
sudo systemctl set-default graphical.target

# --- labwc ---
cat > /home/pi/.config/labwc/environment <<'EOF'
XCURSOR_THEME=PiXtrix
XCURSOR_SIZE=24
XCURSOR_PATH=/home/pi/.icons:/home/pi/.local/share/icons:/usr/share/icons
WLR_LIBINPUT_NO_DEVICES=1
EOF

cat > /home/pi/.config/labwc/autostart <<'EOF'
#!/bin/sh
# /etc/xdg/labwc/autostart still launches the desktop panel and wallpaper.
pkill -f 'lwrespawn /usr/bin/wf-panel-pi' || true
pkill -f 'lwrespawn /usr/bin/pcmanfm-pi' || true
pkill -x wf-panel-pi || true
pkill -x pcmanfm || true
pkill -x kanshi || true
/home/pi/bin/crt-mode-supervisor.py &
EOF
chmod 755 /home/pi/.config/labwc/autostart

if [ ! -f /home/pi/.config/labwc/rc.xml ]; then
    if [ -f /etc/xdg/labwc/rc.xml ]; then
        cp /etc/xdg/labwc/rc.xml /home/pi/.config/labwc/rc.xml
    elif [ -f /usr/share/labwc/rc.xml ]; then
        cp /usr/share/labwc/rc.xml /home/pi/.config/labwc/rc.xml
    else
        cat > /home/pi/.config/labwc/rc.xml <<'EOF'
<?xml version="1.0"?>
<labwc>
  <windowRules>
  </windowRules>
</labwc>
EOF
    fi
fi

python3 - "$HOME/.config/labwc/rc.xml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needles = (
    'identifier="vlc"',
    'identifier="org.videolan.VLC"',
    'title="lsao-kiosk"',
)
if all(n in text for n in needles):
    raise SystemExit(0)

rules = """
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
"""
if "</windowRules>" in text:
    text = text.replace("</windowRules>", rules + "  </windowRules>", 1)
elif "</labwc>" in text:
    text = text.replace(
        "</labwc>",
        "  <windowRules>" + rules + "  </windowRules>\n</labwc>",
        1,
    )
else:
    raise SystemExit("could not find </windowRules> or </labwc> in rc.xml")
path.write_text(text)
PY

# --- composite / CRT firmware (merge keys; keep the rest of config.txt) ---
CONFIG=/boot/firmware/config.txt
if [ ! -f "$CONFIG" ]; then
    CONFIG=/boot/config.txt
fi
if [ -f "$CONFIG" ]; then
    sudo cp "$CONFIG" "$CONFIG.bak-tvosc"
    sudo python3 - "$CONFIG" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
wanted = {
    "display_auto_detect": "0",
    "enable_tvout": "1",
    "dtoverlay": "vc4-fkms-v3d",
    "max_framebuffers": "2",
    "disable_fw_kms_setup": "1",
    "arm_64bit": "1",
    "disable_overscan": "1",
}
lines = path.read_text().splitlines()
seen = set()
out = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("#") or "=" not in stripped:
        out.append(line)
        continue
    key = stripped.split("=", 1)[0].strip()
    if key in wanted:
        out.append(f"{key}={wanted[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, value in wanted.items():
    if key not in seen:
        out.append(f"{key}={value}")
path.write_text("\n".join(out) + "\n")
PY
fi

echo
echo "TVOSC kiosk files are installed."
echo "Reboot to start LightDM/labwc:  sudo reboot"
echo "After reboot: tail -f /home/pi/.local/state/crt-mode.log"
echo "On a Pi 4, analog out may not be plughw:0,0. Check: aplay -l"
