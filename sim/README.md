# 녹화 조건

벤치마크의 bag을 만든 것들이다. **이 디렉터리는 `~/ros2_ws/hero-release/sim`의
사본이다** — 그쪽이 실제로 실행되는 트리이고, 여기 있는 것은 조건을 저장소 안에
남기기 위한 기록이다. 2026-09-12에 뜬 사본.

## 왜 저장소 안에 있는가

두 가지 때문이다.

**재현.** `vision_fisheye.param.yaml`이 담은 마운트 높이와 회전은 여기 있는
`sensor_kit_calibration.yaml`에서 유도된 것이고, 그 파일이 저장소 밖에 있으면
숫자의 출처가 이 기계에만 존재한다. 2026-09-11에 `actor_origin_height_m`을
0.0372에서 0.03635로 고친 것이 바로 그 부류의 변경이다 — 저장소의 설정과 킷의
값이 같은 커밋에서 움직여야 한다.

**테스트.** `monoscale_odometry/test/test_sensor_kit_consistency.py`가 킷을
읽어 `vision_fisheye.param.yaml`의 외부 파라미터와 대조한다. 그 탐색 경로는
`<repo>/sim/ioniq_carla_bridge/config`를 이 기계의 절대경로보다 **먼저** 본다.
사본이 없으면 그 테스트는 실패가 아니라 `pytest.skip`이었다 — 다른 기계에서는
일관성 보장이 조용히 사라지고 있었다는 뜻이다.

## 표류

사본은 원본과 갈라질 수 있다. `test_sim_copy_matches_live.py`가 원본이 있는
기계에서 둘을 대조하고, 없는 기계에서는 건너뛴다. **킷을 고칠 때는 양쪽을
고쳐라.**

## 무엇이 들어 있는가

| | |
| --- | --- |
| `ioniq_carla_bridge/config/sensor_kit_calibration.yaml` | 센서가 액터에 붙는 위치. 오도메트리 외부 파라미터의 출처. |
| `ioniq_carla_bridge/config/sensor_mapping_fisheye_live.yaml` | 벤치가 쓴 센서 킷 — 해상도, FOV, `motion_blur_intensity: 0.0`, `shutter_speed: 200`. |
| `record_run.sh` | 한 드라이브를 찍는다. 브리지를 띄우고, 어안을 조립하고, 파일럿을 돌리고, bag을 쓴다. |
| `rec_slalom.sh` `rec_straight.sh` `rec_rest.sh` | 벤치의 드라이브 종류별 조리법. |
| `reset_world.py` `set_sun.py` `spawn_obstacles.py` `wait_sim_seconds.py` | `record_run.sh`가 부르는 것들. |

## 저장소 밖에 남아 있는 것

여기 담기지 않았고, 없으면 재현이 되지 않는다.

- **CARLA 서버**: `./CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000`.
  **`-quality-level=Low`는 선택이 아니다.** Epic으로 찍으면 같은 드라이브의
  길이 편향이 62배 움직인다(직진 Epic −6.826% 대 Low +0.111%). 노면 텍스처가
  광도 정합이 읽는 바로 그것이기 때문이다.
- **PhysX substep**: `autoware_carla_interface/src/autoware_carla_interface/`
  `carla_autoware.py`의 `settings.max_substep_delta_time = 0.002`. CARLA의
  기본값 0.01은 1/60 s 안에 8.33 ms substep 두 개를 넣고, 보고되는 각속도가
  최대 0.9 deg/s의 턴온 바이어스를 갖는다. 그것이 자이로 바이어스로 읽혔고
  IMU도 브리지도 아니었다. `monoscale_evaluation/bag_gate.py`가 녹화된 bag에서
  그 흔적을 검사한다.
- **`ioniq_carla_bridge` 패키지 본체** (런치, 노드). 여기에는 설정만 담았다.

## 한 드라이브 찍기

```bash
cd ~/ros2_ws/hero-release/sim
./rec_slalom.sh <이름> <초> <조향> [세그먼트초] [속도]
```

`rec_slalom.sh`는 `FISHEYE_ARGS`를 자기 값으로 설정하되 호출자가 내보낸 것을
덧붙인다. 그래서 어안 조립기의 파라미터를 이렇게 넘길 수 있다:

```bash
FISHEYE_ARGS="-p interpolation:=cubic" ./rec_slalom.sh curve_s27_cubic 90 0.265 4.0 1.4
```

2026-09-11까지는 덧붙이지 않고 덮어써서, 호출자가 넘긴 인자가 조용히 버려졌다.

벤치 세트의 bag 이름과 조건은 `src/monoscale_evaluation/README.md`에 있다.
