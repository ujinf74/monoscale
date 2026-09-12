#!/usr/bin/env bash
# Bring up the bridge, drive the ego on a scripted manoeuvre and record the
# raw streams.
#
# The simulator is deliberately slowed while recording. A bag that dropped
# frames under best_effort back-pressure is not comparable between two
# configurations, and comparing configurations is the only reason this bag
# exists; how the stack behaves at real time is measured online, not here.
#
# usage: record_run.sh <name> <speed_mps> <sim_seconds> [max_real_delta_seconds]
set -u
name=${1:?usage: record_run.sh <name> <speed_mps> <sim_seconds> [max_real_delta]}
speed=${2:?target speed in m/s}
sim_seconds=${3:-30}
max_real_delta=${4:-0.03}
fixed_delta=${FIXED_DELTA:-0.0016666667}
# Town10HD spawn point 102: 120 m of lane ahead without a single degree of
# turn, found by walking the map's waypoints.
spawn_point=${SPAWN_POINT:-"-48.820,-4.795,0.600,0.000,0.000,89.839"}
# The pilot has to keep driving for the whole recording, in sim seconds.
pilot_seconds=$(python3 -c "print(float($sim_seconds) + 30.0)")

here=$(dirname "$(readlink -f "$0")")
root=${HERO_BAG_ROOT:-/home/i/hero_bags}
logs="$here/logs"
mkdir -p "$root" "$logs"
target="$root/$name"
if [ -e "$target" ]; then
  echo "refusing to overwrite $target" >&2
  exit 1
fi

# The ROS setup files read variables they have not set yet, so nounset has to
# stand down for the lines that source them.
set +u
source /opt/ros/humble/setup.bash
# The bridge lives in the Autoware workspace. Relying on the calling shell to
# have it sourced worked until it did not, and the failure surfaces 180 s later
# as "no images" rather than as a missing package.
source /home/i/autoware/install/setup.bash
source "${WORKSPACE_SETUP:-$(dirname "$(dirname "$(readlink -f "$0")")")/install/setup.bash}"
if [ -n "${HERO_OVERLAY_SETUP:-}" ]; then
  source "$HERO_OVERLAY_SETUP"
fi
set -u

# Cameras publish 921,600 byte frames on a best_effort topic, which Cyclone
# splits into UDP datagrams; lose one and the whole sample goes. Measured on
# the bridge with nothing recording, camera_info arrived at 60 Hz on both
# cameras while the rear image arrived at 44.6 -- the two are published one
# line apart in the same function, so the images were being dropped in
# transport. Raising net.core.wmem_max off its 212,992 default took rear to
# 50.1 Hz and this file's fragment size took it to 60.0, matching info exactly.
# Section 12 read the same 44 Hz as a ceiling in the bridge's publish path; it
# was the socket and the fragment size.
#
# Needs net.core.wmem_max and rmem_max at 16 MB (see /etc/sysctl.d).
export CYCLONEDDS_URI=file://$here/config/cyclonedds.xml
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ROS_LOCALHOST_ONLY=1 cannot be used here even though every node is local:
# this kernel's lo carries no MULTICAST flag, so discovery confined to loopback
# never completes and the run times out with no images at all.

cleanup() {
  "$here/stop_stack.sh" >/dev/null 2>&1
  python3 "$here/reset_world.py" >/dev/null 2>&1
}
trap cleanup EXIT

echo "== resetting world"
python3 "$here/reset_world.py"

echo "== bridge"
# Town10HD has no junction-free lane longer than 60 m that is straighter than
# five degrees, so anything that weaves runs into junction furniture: the
# slalom drives all hit a kerb or a lamppost at 84-88 m. CARLA_MAP moves a
# recording to a map with room -- Town01_Opt gives 224 m of straight at 4.0 m
# of lane width, Town04_Opt 236 m.
ros2 launch ioniq_carla_bridge carla_vehicle.launch.py \
  carla_map:="${CARLA_MAP:-Town10HD_Opt}" \
  sensor_mapping_file:="${SENSOR_MAPPING:-sensor_mapping_vio_only.yaml}" \
  fixed_delta_seconds:="$fixed_delta" \
  max_real_delta_seconds:="$max_real_delta" \
  spawn_point:="$spawn_point" \
  > "$logs/$name.bridge.log" 2>&1 &

# Every extrinsic in the kit is measured against one car: the camera heights
# the ground plane solve scales with, the wheelbase the truth is shifted by.
# The bridge takes the type from a launch default and ignores it outright under
# attach_to_existing_ego, so a wrong car spawns quietly and the whole recording
# comes out plausible and wrong. Checked here rather than read back out of a
# log afterwards.
#
# Generous, because this gate sits in front of the wait for images and inherits
# its job of outlasting a map load. At 120 s it expired in the same second the
# ego appeared and killed two recordings outright.
echo "== confirming the vehicle"
expect=${EXPECT_VEHICLE:-vehicle.toyota.prius}
deadline=$((SECONDS + 300))
until python3 - "$expect" <<'CHECK'
import sys
import carla
wanted = sys.argv[1]
try:
    client = carla.Client('localhost', 2000, worker_threads=1)
    # The 5 s default is not enough while the bridge is loading the map.
    client.set_timeout(30.0)
    world = client.get_world()
    # A synchronous world hands snapshots to the client that owns the tick, and
    # the bridge owns it here. Everyone else reads frame 0 and an empty actor
    # list until they wait for one, so without this the gate spends its whole
    # deadline looking at a world with no ego in it -- and the except below
    # makes that indistinguishable from not being connected at all.
    world.wait_for_tick(20.0)
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') != 'ego_vehicle':
            continue
        if actor.type_id != wanted:
            print(f'ego is {actor.type_id}, expected {wanted}', file=sys.stderr)
            sys.exit(2)
        print(actor.type_id)
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
CHECK
do
  status=$?
  if [ $status -eq 2 ]; then
    echo "refusing to record against the wrong vehicle" >&2
    exit 1
  fi
  if [ $SECONDS -gt $deadline ]; then
    echo "no ego after 300 s; see $logs/$name.bridge.log" >&2
    exit 1
  fi
  sleep 3
done

# Sun and obstacles go down only now, after the ego gate has proven the world
# is loaded and ticking. They used to go down before the bridge -- and the
# bridge calls client.load_world() unconditionally, which destroys every actor
# and resets the weather. Since the 1.6 -> 1.8 interface move, every recording
# made with the old ordering came out of an empty world with default light:
# "11 of 11 obstacles placed" was true for about two seconds. Found on 08-12
# by a LiDAR sweep of a bag whose reference swore four cars were in it.
#
# The sun matters for the same comparability reason it always did: leaving it
# to the map default made every recording a different scene.
echo "== sun"
python3 "$here/set_sun.py" \
  --azimuth "${SUN_AZIMUTH:-300}" --altitude "${SUN_ALTITUDE:-45}"

if [ -n "${OBSTACLES:-}" ]; then
  echo "== obstacles"
  layout=${OBSTACLES}
  [ "$layout" = 1 ] && layout=a
  python3 "$here/spawn_obstacles.py" --layout "$layout"
fi

# The cameras publish best_effort; an echo left on the default reliable profile
# never matches them and the wait spins until the deadline.
echo "== waiting for images"
# Wait on a camera the mapping actually spawns. The old default named the VIO
# pair, which is not in enabled_sensors for a fisheye recording -- removing
# those two cameras to save render time made this wait for something that
# would never arrive, and the run died after its 180 s deadline with a
# perfectly healthy bridge.
if [ -n "${ASSEMBLE_FISHEYE:-}" ]; then
  default_wait_topic=/sensing/camera/front/fisheye_a/image_raw
else
  default_wait_topic=/sensing/camera/front/vio/image_raw
fi
wait_image_topic=${WAIT_IMAGE_TOPIC:-$default_wait_topic}
deadline=$((SECONDS + 180))
until timeout 10 ros2 topic echo --once --qos-reliability best_effort \
      "$wait_image_topic" >/dev/null 2>&1; do
  if [ $SECONDS -gt $deadline ]; then
    echo "no images after 180 s; see $logs/$name.bridge.log" >&2
    exit 1
  fi
  sleep 3
done
echo "images flowing"

# The fisheye is assembled from the two pinholes CARLA renders per end, and
# only when the mapping asked for them. Started before the ground truth so the
# first pair it publishes is already inside the recorded window.
if [ -n "${ASSEMBLE_FISHEYE:-}" ]; then
  echo "== fisheye assembler"
  if [ -n "${FISHEYE_SPLIT:-}" ]; then
    ros2 run monoscale_carla fisheye_assembler --ros-args \
      -r __node:=fisheye_assembler_front -p use_sim_time:=true \
      -p "ends:=['front']" -p "base_yaw_deg:=[0.0]" ${FISHEYE_ARGS:-} \
      > "$logs/$name.fisheye-front.log" 2>&1 &
    ros2 run monoscale_carla fisheye_assembler --ros-args \
      -r __node:=fisheye_assembler_rear -p use_sim_time:=true \
      -p "ends:=['rear']" -p "base_yaw_deg:=[180.0]" ${FISHEYE_ARGS:-} \
      > "$logs/$name.fisheye-rear.log" 2>&1 &
  else
    ros2 run monoscale_carla fisheye_assembler --ros-args \
      -p use_sim_time:=true ${FISHEYE_ARGS:-} \
      > "$logs/$name.fisheye.log" 2>&1 &
  fi
fi

echo "== ground truth"
ros2 run monoscale_carla carla_ground_truth --ros-args \
  -p use_sim_time:=true > "$logs/$name.truth.log" 2>&1 &

# The traffic manager is not usable from here: in a synchronous world it only
# applies its commands when the client that owns the tick has put it into
# synchronous mode, and the tick belongs to the bridge. ego_pilot drives
# through the actuation topics the bridge already subscribes to, which also
# makes the manoeuvre repeatable.
echo "== pilot at $speed m/s"
ros2 run monoscale_carla ego_pilot --ros-args \
  -p use_sim_time:=true \
  -p start_delay_sec:=1.0 \
  -p max_throttle:=0.9 \
  -p segment_durations:="${SEGMENT_DURATIONS:-[$pilot_seconds]}" \
  -p segment_gears:="${SEGMENT_GEARS:-['drive']}" \
  -p segment_speeds:="${SEGMENT_SPEEDS:-[$speed]}" \
  -p segment_steers:="${SEGMENT_STEERS:-[0.0]}" \
  > "$logs/$name.pilot.log" 2>&1 &

# Wait for the target speed, not merely for motion. Opening the window as soon
# as the car creeps means most of the scored distance is spent accelerating,
# and a car under throttle rides nose up: the ground plane the estimator
# assumes is no longer the one in front of it, and the scale comes out short.
# At 8 m/s that alone read as a 2.4% scale error.
#
# RECORD_FROM_STANDSTILL=1 skips the gate. The gate itself created a 1.27 s /
# 3.5 m hole at the head of every bag that was read as VIO error for weeks
# (28-33): the estimator boots mid-motion in the bag where the deployed system
# boots parked. A deployment-faithful bag records the standstill and the
# ramp-up, and takes the nose-up scale error as part of reality.
if [ -n "${RECORD_FROM_STANDSTILL:-}" ]; then
  echo "== recording from standstill (speed gate skipped)"
else
gate_speed=$(python3 -c "print(0.9 * float($speed))")
echo "== waiting for $gate_speed m/s"
deadline=$((SECONDS + 150))
until timeout 10 ros2 topic echo --once --qos-reliability best_effort \
        --field longitudinal_velocity /vehicle/status/velocity_status 2>/dev/null \
      | awk -v want="$gate_speed" '$1 + 0 > want { found = 1 } END { exit !found }'; do
  if [ $SECONDS -gt $deadline ]; then
    echo "the ego never reached $gate_speed m/s; see $logs/$name.pilot.log" >&2
    tail -8 "$logs/$name.pilot.log" >&2
    exit 1
  fi
  sleep 2
done
echo "at speed"
fi

# The wall budget is a ceiling, not the duration. max_real_delta_seconds makes
# the bridge sleep so that every tick takes at least that long, so the budget
# below is what the throttle would cost if it were honoured exactly -- and the
# recording used to run for all of it, whether or not the drive was over. The
# waiter watches /clock and stops the recorder as soon as the simulation has
# covered sim_seconds, which is the thing actually being asked for. Doubled
# because the throttle is a floor per tick: real work on top makes ticks
# longer, never shorter.
ratio=$(python3 -c "print(max($max_real_delta / $fixed_delta, 1.0))")
wall=$(python3 -c "print(int($sim_seconds * $ratio * 2 + 30))")
echo "== recording ${sim_seconds}s of sim time (wall ceiling ${wall}s)"
# A bigger cache than the 100 MB default: the recorder was losing 12-15% of the
# image stream on longer runs even with the simulator far below real time,
# which is a queue that empties too slowly rather than a disk that cannot keep
# up.
if [ -n "${RECORD_FISHEYE_ONLY:-}" ]; then
  camera_topics=(
    /sensing/camera/front/fisheye/image_raw
    /sensing/camera/front/fisheye/camera_info
    /sensing/camera/rear/fisheye/image_raw
    /sensing/camera/rear/fisheye/camera_info
  )
  # The two pinholes the assembler remaps into that fisheye. Recorded only when
  # asked: they are what lets the assembly itself be measured rather than
  # assumed, and they are four times the bytes.
  if [ -n "${RECORD_FISHEYE_SOURCES:-}" ]; then
    camera_topics+=(
      /sensing/camera/front/fisheye_a/image_raw
      /sensing/camera/front/fisheye_a/camera_info
      /sensing/camera/front/fisheye_b/image_raw
      /sensing/camera/front/fisheye_b/camera_info
    )
  fi
else
  camera_topics=(
    /sensing/camera/front/vio/image_raw
    /sensing/camera/front/vio/camera_info
    /sensing/camera/rear/vio/image_raw
    /sensing/camera/rear/vio/camera_info
  )
fi
# /tf and /tf_static cost nothing today -- nothing in this sim publishes them,
# so the bags come out the same size -- and they are recorded anyway because the
# odometry's own `mount_from_tf` asks the tree for each camera's mount and falls
# back to its parameters when the tree is silent. Live it uses the tree; from a
# bag it has never had the option, so every replayed number in this stack has
# been taken against the configured extrinsics with no way to notice if the two
# disagreed. The moment anything publishes the kit as TF, replay stops guessing.
ros2 bag record \
  --max-cache-size 1073741824 \
  --output "$target" \
  /clock \
  "${camera_topics[@]}" \
  /sensing/imu/imu_data \
  /carla/ground_truth/pose \
  /tf \
  /tf_static \
  ${RECORD_EXTRA:-} \
  > "$logs/$name.record.log" 2>&1 &
recorder=$!
python3 "$here/wait_sim_seconds.py" "$sim_seconds" "$wall"
# INT, not TERM: the recorder flushes its cache on interrupt and truncates the
# bag on anything harsher.
kill -INT "$recorder" 2>/dev/null
wait "$recorder" 2>/dev/null

echo "== pilot speeds"
grep -E "m/s" "$logs/$name.pilot.log" | tail -8
echo "== bag"
ros2 bag info "$target" 2>&1 | sed -n '1,30p'
