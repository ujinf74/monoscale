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
| `monoscale_occupancy_grid_map` | 지면점을 격자에 누적한다. |
| `monoscale_evaluation` | 참값 대비 채점. 차량에는 올리지 않는다. |
| `monoscale_carla` | CARLA 전용: 카메라 조립, 참값 탭, 주행 스크립트. |

## 토픽

```
이미지  →  monoscale_tracker         →  /vision/tracks/<camera>
        →  monoscale_odometry        →  /localization/kinematic_state
                                        /perception/ground_points
        →  monoscale_occupancy_grid_map  →  /perception/occupancy_grid_map
```

`/perception/ground_points`는 map 프레임의 PointCloud2다. 점마다 `label`
(지면 0, 장애물 1)과 그 점을 본 카메라 위치(`origin_x`, `origin_y`)를 싣기
때문에, 격자 노드는 외부 파라미터도 TF도 필요 없다. 격자를 다시 손봐도
오도메트리를 다시 돌릴 필요가 없다는 뜻이기도 하다. 실제로 매핑 주기를 어떻게
두든 궤적은 비트 단위로 같다 — 지도는 자세 루프의 일부가 아니라 부산물이다.

격자에는 범위가 없다. 셀은 관측된 자리에만 타일로 잡히고, 발행되는 것은 차
주변에서 실제로 증거가 있는 타일의 바운딩 박스라 크기와 원점이 매 프레임
변한다. 예전에는 시작 지점을 중심으로 한 60 m 고정 창이었는데, 주차 조작에는
맞았지만 도로를 달리면 그 밖의 점을 조용히 버렸다 — 08-18 측정에서 차가
x = 30.35 m에 있을 때 들어오는 지면점의 49 %가 버려지고 있었다. 타일 구조와
그 값들은 `ioniq-autopark-ws`의 `pointcloud_ogm_accumulator`에서 왔고, 그쪽은
LiDAR 참조로 검증된 것이다. 근거는 `config/occupancy.param.yaml`에 있다.

## 빌드와 실행

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install
source install/setup.bash

ros2 launch monoscale_odometry odometry.launch.py
```

CUDA는 필요 없다. `monoscale_tracker`의 광류에 GPU 경로가 있지만
(`use_cuda`), cudaoptflow가 있는 OpenCV를 만났을 때만 빌드되고 기본값은
꺼져 있다.

## 카메라 대수

`camera_names`가 대수를 정한다. 기본값은 차량에 달린 두 대다.

```yaml
camera_names: ['front', 'rear']
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

기하, 앵커맵, 필터, 관성, 자세, 격자, 그리고 합성 주행 위의 추정기 전체까지
118개다. 채점 패키지의 Python 테스트는 `python3 -m pytest src/monoscale_evaluation/test`.

## 기록된 주행에 대고 채점하기

```bash
ros2 run monoscale_odometry monoscale_replay <bag> \
  --params src/monoscale_odometry/config/vision_fisheye.param.yaml \
  --set track_topic_prefix:=/vision/tracks
```

bag을 직접 읽어 라이브러리를 녹화 순서대로 돌린다. 시계도 난수도 읽지 않으므로
같은 bag은 데스크톱에서든 Orin에서든 같은 궤적을 낸다 — 회귀가 보이는 것은
그 때문이다.
