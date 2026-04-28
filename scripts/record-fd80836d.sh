#!/usr/bin/env bash
set -euo pipefail

CAM_ID="fd80836d"
RTSP="rtsp://admin:Nutsniper0977@192.168.1.164:554/Streaming/Channels/102"
DIR="$HOME/recordings/$CAM_ID"
SEG_DIR="$DIR/segments"
PATH="/opt/homebrew/bin:$PATH"

mkdir -p "$SEG_DIR"

# Re-encode to H.264 ~400kbps for archive efficiency.
# 6-second segments. Strftime filenames so cleanup is trivial.
# Master m3u8 grows indefinitely (append_list); FastAPI generates sliding-window
# live.m3u8 dynamically from segment mtimes.
exec ffmpeg \
    -loglevel warning \
    -nostats \
    -rtsp_transport tcp \
    -i "$RTSP" \
    -c:v libx264 \
    -preset veryfast \
    -tune zerolatency \
    -profile:v main \
    -b:v 400k \
    -maxrate 600k \
    -bufsize 1200k \
    -g 60 \
    -keyint_min 60 \
    -sc_threshold 0 \
    -an \
    -f hls \
    -hls_time 6 \
    -hls_list_size 0 \
    -hls_flags append_list+independent_segments+omit_endlist+second_level_segment_index \
    -hls_segment_type mpegts \
    -strftime 1 \
    -hls_segment_filename "$SEG_DIR/%Y%m%d_%H%M%S_%%05d.ts" \
    "$DIR/index.m3u8"
