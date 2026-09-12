#!/bin/bash
# Straight on Town01_Opt: 224 m of junction-free lane, so 20 s at 8 m/s fits
# where Town10HD ran out at 120 m and straight120_s8 hit at 134.7 m.
#   usage: rec_straight.sh <name> <speed> <sim_seconds>
set -u
cd /home/i/ros2_ws/hero-release/sim
env -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH \
    -u PYTHONPATH -u LD_LIBRARY_PATH -u CYCLONEDDS_URI -u RMW_IMPLEMENTATION \
  HERO_OVERLAY_SETUP=/home/i/monoscale/install/setup.bash \
  SENSOR_MAPPING=sensor_mapping_fisheye_live.yaml \
  FIXED_DELTA=0.0166666667 \
  ASSEMBLE_FISHEYE=1 FISHEYE_SPLIT=1 RECORD_FISHEYE_ONLY=1 \
  FISHEYE_ARGS="-p source_reliability:=best_effort -p width:=1280 -p height:=720" \
  CARLA_MAP="${CARLA_MAP:-Town01_Opt}" \
  SPAWN_POINT="${SPAWN_POINT:-101.42,199.14,0.600,0.000,0.000,0.00}" \
  ./record_run.sh "$1" "$2" "$3"
