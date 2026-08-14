# monoscale

지면평면을 기준으로 **미터 단위 스케일을 얻는 시각-관성 오도메트리**와, 그
과정에서 나온 지면점으로 만드는 **점유격자**다. 스테레오도 라이다도 쓰지
않는다. 스케일은 카메라 장착 높이와 지면평면에서 나오고, 카메라는 1대부터
N대까지 쓸 수 있다. 사전 기록 지도를 쓰지 않고(mapless), 학습 모델도 쓰지
않는다.

## 패키지

| 패키지 | 하는 일 |
| --- | --- |
| `monoscale_tracker` | C++ KLT 전단. 이미지에서 특징 궤적만 뽑아 발행한다. |
| `monoscale_fast` | 위 추종기가 쓰는 FAST 검출 커널(CUDA, CPU 대체 있음). |
| `monoscale_odometry` | 궤적과 IMU를 융합해 자세와 지면점을 낸다. |
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
오도메트리를 다시 돌릴 필요가 없다는 뜻이기도 하다.

## 빌드와 실행

```bash
source /opt/ros/humble/setup.bash
export PATH=/usr/local/cuda/bin:$PATH      # CUDA 커널을 쓸 때만
colcon build --base-paths src --symlink-install
source install/setup.bash

ros2 launch monoscale_odometry odometry.launch.py
```

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
python3 -m pytest src/*/test -q
```
