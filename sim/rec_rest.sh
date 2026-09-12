#!/bin/bash
set -u
cd /home/i/ros2_ws/hero-release/sim
SC=/tmp/claude-1000/-home-i-monoscale/2e1d95d6-8247-4109-8573-5841db9f5f99/scratchpad
./rec_slalom.sh curve_s05_t01 60 0.05 > /tmp/rec_cs05.log 2>&1
python3 $SC/checkbag.py /home/i/hero_bags/curve_s05_t01 2>/dev/null | grep -v INFO
./rec_slalom.sh curve_s20_t01 60 0.20 > /tmp/rec_cs20.log 2>&1
python3 $SC/checkbag.py /home/i/hero_bags/curve_s20_t01 2>/dev/null | grep -v INFO
./rec_straight.sh straight_s8_t01 8.0 20 > /tmp/rec_s8.log 2>&1
python3 $SC/checkbag.py /home/i/hero_bags/straight_s8_t01 2>/dev/null | grep -v INFO
echo "== 전부 완료"
