#!/usr/bin/env bash

# Convert each video directly inside a directory to a CRT-friendly 640x480
# H.264 MP4. Wider or taller material is center-cropped to 4:3 after scaling;
# no letterboxing is added.

set -uo pipefail

usage() {
    printf 'Usage: %s [video-directory]\n' "$(basename "$0")"
    printf 'The directory defaults to the current directory.\n'
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

video_dir=${1:-.}

if [[ ! -d "$video_dir" ]]; then
    printf 'Error: not a directory: %s\n' "$video_dir" >&2
    exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'Error: ffmpeg is not installed or is not in PATH.\n' >&2
    exit 127
fi

converted=0
skipped=0
failed=0
found=0
current_tmp=

cleanup() {
    if [[ -n "$current_tmp" ]]; then
        rm -f -- "$current_tmp"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while IFS= read -r -d '' input; do
    filename=${input##*/}
    extension=${filename##*.}
    # macOS still ships Bash 3.2, which does not support ${value,,}.
    extension=$(printf '%s' "$extension" | LC_ALL=C tr '[:upper:]' '[:lower:]')

    case "$extension" in
        3gp|avi|flv|m2ts|m4v|mkv|mov|mp4|mpeg|mpg|mts|ts|webm|wmv)
            ;;
        *)
            continue
            ;;
    esac

    # Do not recursively convert this script's own output files.
    if [[ "$filename" == SMALL_* ]]; then
        continue
    fi

    ((found += 1))
    stem=${filename%.*}
    output="$video_dir/SMALL_${stem}.mp4"

    if [[ -e "$output" ]]; then
        printf 'Skipping existing output: %s\n' "$output"
        ((skipped += 1))
        continue
    fi

    current_tmp="${output%.mp4}.tmp.$$.mp4"
    printf 'Converting: %s -> %s\n' "$input" "$output"

    if ffmpeg -nostdin -hide_banner -loglevel warning -stats \
        -i "$input" \
        -map 0:v:0 -map '0:a?' -sn -dn \
        -vf 'scale=640:480:force_original_aspect_ratio=increase,crop=640:480,setsar=1' \
        -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p \
        -c:a aac -b:a 128k -ac 2 -ar 48000 \
        -movflags +faststart \
        "$current_tmp"; then
        mv -- "$current_tmp" "$output"
        current_tmp=
        ((converted += 1))
    else
        printf 'Failed: %s\n' "$input" >&2
        rm -f -- "$current_tmp"
        current_tmp=
        ((failed += 1))
    fi
done < <(find "$video_dir" -maxdepth 1 -type f -print0)

if (( found == 0 )); then
    printf 'No supported videos found in: %s\n' "$video_dir"
else
    printf 'Done: %d converted, %d skipped, %d failed.\n' \
        "$converted" "$skipped" "$failed"
fi

(( failed == 0 ))
