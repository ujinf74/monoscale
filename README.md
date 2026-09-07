# monoscale

지면평면을 기준으로 **미터 단위 스케일을 얻는 시각-관성 오도메트리**와, 그
과정에서 나온 지면점으로 만드는 **점유격자**다. 스테레오도 라이다도 쓰지
않는다. 스케일은 카메라 장착 높이와 지면평면에서 나오고, 카메라는 1대부터
N대까지 쓸 수 있다. 사전 기록 지도를 쓰지 않고(mapless), 학습 모델도 쓰지
않는다.

차량에 올라가는 것은 전부 C++이다. 이 스택은 Python 추정기로 개발됐고, 그
Python은 궤적이 일치하는 것을 확인한 뒤 이력에만 남기고 트리에서 걷어냈다.
어느 시점을 옮긴 것인지와 무엇을 남기지 않았는지는
[`src/monoscale_core/README.md`](src/monoscale_core/README.md)에 있다.

## 패키지

| 패키지 | 하는 일 |
| --- | --- |
| `monoscale_core` | 추정 그 자체. ROS를 모른다 — 그래서 그래프 없이 시험되고, 나중에 추종기와 한 프로세스로 묶일 수 있다. |
| `monoscale_tracker` | C++ KLT 전단. 이미지에서 특징 궤적만 뽑아 발행한다. |
| `monoscale_odometry` | 위를 감싸는 노드. bag을 직접 재생해 채점하는 `monoscale_replay`도 여기 있다. |
| `monoscale_occupancy_grid_map` | 원본 어안 영상을 평면 스윕해 점유격자를 만든다. CUDA 필요. |
| `monoscale_evaluation` | 참값 대비 채점. 차량에는 올리지 않는다. |
| `monoscale_carla` | CARLA 전용: 카메라 조립, 참값 탭, 주행 스크립트. |

## 토픽

```
이미지 ─┬─→ monoscale_tracker  →  /vision/tracks/<camera>
        │                          └→ monoscale_odometry
        │                                 ├→ /localization/kinematic_state
        │                                 └→ /perception/ground_points
        └────────────────────────→ monoscale_occupancy_grid_map
                  + camera_info               └→ /perception/occupancy_grid_map
                  + /localization/kinematic_state
```

두 갈래가 이미지에서 갈린다. 오도메트리는 특징 궤적만 보고, 점유격자는 원본
영상을 다시 본다.

격자는 **평면 스윕**으로 만든다. 세계-수평 평면을 −0.15 m부터 0.05 m 계단으로
쌓아 올리며 화소마다 어느 높이에서 두 시점이 가장 잘 맞는지를 ZNCC로 재고,
SGM으로 집계해 그 화소의 표면 높이를 정한다. 즉 높이가 **판정 결과**이지 특징
삼각측량의 부산물이 아니다. 자세(롤·피치)는 구독한 오도메트리에서 직접 꺼내
화소별 워프에 넣는다.

이전 구현은 오도메트리가 발행하는 `/perception/ground_points`를 격자에 누적했다.
그 토픽은 지금도 나오지만 격자는 더 이상 쓰지 않는다. 코너를 내주지 않는
장애물은 그 경로에 존재하지 않았고, 깊이 사슬에 2026-09-02 감사가 측정한 결함
다섯 개가 있었다 — 그중 가장 큰 것은 워프가 피치·롤을 0으로 고정한 것으로,
프레임 공통 오차의 94 %였다.

격자는 0.1 m 해상도의 고정 60 x 60 m이고 첫 포즈에 앵커된다. 참값 대비 채점은
`approach_hd60_occ_b`에서 G1(차량을 자유라 함) 0, G2(막힌 곳을 자유라 함) 3,
G3(커버) 0.831, G4(오탐) 29, 경로유령 0이다. 파이썬 참조 구현이 같은 조건에서
G2 2 / G3 0.829 / G4 39이므로, 커버와 오탐에서는 이쪽이 앞선다.

## 빌드와 실행

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install
source install/setup.bash

ros2 launch monoscale_odometry odometry.launch.py
```

오도메트리에는 CUDA가 필요 없다. `monoscale_tracker`의 광류에 GPU 경로가
있지만(`use_cuda`), cudaoptflow가 있는 OpenCV를 만났을 때만 빌드되고 기본값은
꺼져 있다.

**점유격자에는 필요하다.** 평면 스윕의 CPU 경로는 키프레임 674장에 28분이라
배포할 수 없고, CUDA 경로는 키프레임당 0.2초다. 커널은
`monoscale_fast/src/sweep_kernels.cu`를 CMake가 직접 컴파일해 파이썬 참조
구현과 같은 산술을 공유한다. 그 파일이나 nvcc가 없으면 CPU 경로로 조용히
물러나므로, 빌드가 성공했다고 배포 가능한 것은 아니다 — 첫 실행에서
`cuda backend: available=1`을 확인할 것.

## 카메라 대수

`camera_names`가 대수를 정한다. 기본값은 차량에 달린 두 대다.

추정기와 트래커가 이름이 다르다: 추정기는 `camera_names`, 트래커는 `cameras`를
읽는다. 이미지 토픽도 마찬가지로 추정기는 카메라 이름으로 합성한
`<이름>_image_topic`을, 트래커는 `image_topics` 배열을 읽는다. 배포에서는
`deployment.param.yaml`이 트래커 쪽을 채운다.

```yaml
camera_names: ['front', 'rear']       # 추정기
cameras: ['front', 'rear']            # 트래커
front.k: [...]                      # 3x3, 행 우선
front.rotation_base_from_camera: [...]   # 3x3
front.translation_base_from_camera: [x, y, z]
front_image_topic: /sensing/camera/front/image_raw
```

이름을 하나 더 넣으면 그 이름으로 같은 파라미터를 읽고, 프레임은 스탬프가
가장 가깝게 모이는 조합으로 정렬된다. 한 대만 쓰면 카메라끼리의 불일치라는
신호가 사라지므로 `single_camera_variance`가 그 자리를 대신한다 — 돌긴 하지만
두 대보다 눈에 띄게 나쁘다.

## 테스트

```bash
colcon test --packages-select monoscale_core
colcon test-result --all
```

기하, 앵커맵, 필터, 관성, 자세, 그리고 합성 주행 위의 추정기 전체까지
132개다. 채점 패키지의 Python 테스트는 `python3 -m pytest src/monoscale_evaluation/test`.

## 기록된 주행에 대고 채점하기

```bash
ros2 run monoscale_odometry monoscale_replay <bag> \
  --params src/monoscale_odometry/config/vision_fisheye.param.yaml \
  --set track_topic_prefix:=/vision/tracks
```

bag을 직접 읽어 라이브러리를 녹화 순서대로 돌린다. 시계도 난수도 읽지 않으므로
같은 bag은 데스크톱에서든 Orin에서든 같은 궤적을 낸다 — 회귀가 보이는 것은
그 때문이다.
