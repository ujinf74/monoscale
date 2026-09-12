#!/bin/bash
# Slalom on Town01_Opt, which gives 224 m of junction-free straight at 4.0 m of
# lane width. Town10HD has none longer than 60 m and every slalom recorded
# there hit a kerb or a lamppost at 84-88 m.
#
# The first segment is half length. There is no lane keeping: the pilot holds a
# steer for a fixed time, so a run of equal alternating segments leaves the
# heading centred but the *path* offset by the first half cycle. The first
# Town01 attempt started the bag already 9.4 deg off the lane and put the car on
# the pavement 30 m later, into a bench 2.6 m from where it stopped. Halving
# the first segment centres the weave; 4 s segments keep the excursion near
# 0.8 m against the 2 m the lane allows before the kerb.
#   usage: rec_slalom.sh <name> <sim_seconds> <steer> [segment_seconds] [speed_mps]
#
# The speed is the fifth argument and defaults to 2.0, which is what every
# recorded slalom used. It exists to separate the turn term's variable: at a
# fixed 30 Hz, radians per frame and radians per second are proportional and
# cannot be told apart, but radians per metre is yaw rate over speed. Holding
# the steer and doubling the speed holds the curvature and doubles the yaw
# rate, so `s05` at 4.0 m/s has the yaw rate of `s10` at 2.0 and half its
# curvature. Halve `segment_seconds` with the speed to keep the lateral
# excursion where it was -- it goes as the segment's length, and the lamp posts
# leave about 1.2 m.
#
# Town01's lamp posts stand about 2.5 m off the lane centre, so the weave has
# roughly 1.2 m of room. s05 (+0.30 m) and s20 (+1.11 m) clear them; s10 reached
# -1.46 m on 4 s segments and clipped CityLamp133 at 70 m. Shorter segments cut
# the excursion proportionally.
set -u
cd /home/i/ros2_ws/hero-release/sim
env -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH \
    -u PYTHONPATH -u LD_LIBRARY_PATH -u CYCLONEDDS_URI -u RMW_IMPLEMENTATION \
  HERO_OVERLAY_SETUP=/home/i/monoscale/install/setup.bash \
  SENSOR_MAPPING=sensor_mapping_fisheye_live.yaml \
  FIXED_DELTA=0.0166666667 \
  ASSEMBLE_FISHEYE=1 FISHEYE_SPLIT=1 RECORD_FISHEYE_ONLY=1 \
  FISHEYE_ARGS="-p source_reliability:=best_effort -p width:=1280 -p height:=720 ${FISHEYE_ARGS:-}" \
  CARLA_MAP="${CARLA_MAP:-Town01_Opt}" \
  SPAWN_POINT="${SPAWN_POINT:-101.42,199.14,0.600,0.000,0.000,0.00}" \
  SEGMENT_DURATIONS="[$(python3 -c "
import sys
L=float(sys.argv[1]); n=int(90//L)+2
print(','.join(['8.0', f'{L/2:.2f}'] + [f'{L:.2f}']*n))" "${4:-4.0}")]" \
  SEGMENT_STEERS="[$(python3 -c "
import sys
s=float(sys.argv[1]); print('0.000,'+','.join(f'{s if i%2==0 else -s:.3f}' for i in range(int(90//float(sys.argv[2]))+3)))" "$3" "${4:-4.0}")]" \
  SEGMENT_SPEEDS="[$(python3 -c "
import sys
print(','.join([sys.argv[2]]*(int(90//float(sys.argv[1]))+4)))" "${4:-4.0}" "${5:-2.0}")]" \
  SEGMENT_GEARS="[$(python3 -c "
import sys
print(','.join([chr(39)+'drive'+chr(39)]*(int(90//float(sys.argv[1]))+4)))" "${4:-4.0}")]" \
  ./record_run.sh "$1" "${5:-2.0}" "$2"
