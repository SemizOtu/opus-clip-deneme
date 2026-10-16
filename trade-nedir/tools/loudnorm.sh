#!/usr/bin/env bash
# Two-pass EBU R128 normalisation to -15 LUFS / -1.5 dBTP.  usage: loudnorm.sh in.wav out.wav
set -euo pipefail
J=$(ffmpeg -hide_banner -nostats -i "$1" -af loudnorm=I=-15:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
v() { echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -hide_banner -loglevel error -y -i "$1" -af "loudnorm=I=-15:TP=-1.5:LRA=11:measured_I=$(v input_i):measured_TP=$(v input_tp):measured_LRA=$(v input_lra):measured_thresh=$(v input_thresh):offset=$(v target_offset):linear=true" -ar 48000 -c:a pcm_s24le "$2"
