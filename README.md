# TVOSC-RPI

Raspberry Pi kiosk for an analog CRT: live audio visuals plus looping USB video.

GPIO 27 toggles **LSaO** and **USB video**. GPIO 22 cycles LSaO visualizers. Composite NTSC 720×480, 60 fields.

Live visuals come from a kiosk build of [Aaron F. Bianchi’s LSaO-visualizer](https://github.com/aaronfbianchi/LSaO-visualizer) (GPL-3). Video is VLC playing files from a USB stick.

## Layout

- `crt-mode-supervisor.py` — GPIO, USB mount, VLC, LSaO process
- `LSaO-visualizer/kiosk.py` — fullscreen live vis
- `packaging/` — optional polkit rule so `pi` can mount USB without a desktop automounter

## Hardware

- Raspberry Pi 3B+ (or 4 with composite enabled)
- USB microphone
- GPIO 27 (mode) and GPIO 22 (affect) to GND
- Composite → CRT (or a composite-to-UHF modulator)
