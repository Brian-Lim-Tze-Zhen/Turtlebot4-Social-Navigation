# replay_AB — re-run of head-on ablation groups A and B

## Why
The original A–E bags (Aug 14–15 2026) snapshotted a fixed file as
config_used.yaml (social_nav2_fast_person_test.yaml, md5 c4a4c85e...),
not the launched ablation YAML. A–D therefore have no machine proof of
their config. E is verified (its file matched the snapshot).
This folder re-runs A and B with the original code and a recorder that
snapshots what actually runs.

## Restored code (src/social_perception/social_perception/)
| File | Source | md5 |
|---|---|---|
| human_kf_predictor.py | src/ (unchanged since Aug) | aff5d2bde40d9baecc6ba33476375ffd |
| predicted_person_cloud_node.py | bags/b_critweight20_t06/cloud_node_used.py (ELLIPSE_B 1.00, never committed) | 5c6a874ff1e0d250773a6fff21d36430 |
| move_person_oneway.py | bags/b_critweight20_t06/mover_used.py | 24f5ff3ef3f50fa933d877ed66eeb969 |
| yolo_detector.py | git 0451ced^ (before track_id=-1 filter) | differs from current only by that filter |

## Configs
- A: config/ablation/social_nav2_ablation_A_nolayer.yaml (identical to B; A = cloud node not started)
- B: config/ablation/social_nav2_ablation_B_critweight20.yaml

## Launch
ros2 launch /root/thesis_social_navigation_ws/launch/headon_scenario.py ws_root:=/root/thesis_social_navigation_ws/replay_AB enable_predicted_cloud:=false   (A)
ros2 launch /root/thesis_social_navigation_ws/launch/headon_scenario.py ws_root:=/root/thesis_social_navigation_ws/replay_AB enable_predicted_cloud:=true    (B)

## Record
replay_AB/record_trial_headon.sh <bag_name> <params_file> /root/thesis_social_navigation_ws/replay_AB <true|false>
Bag names: a_nolayer_r2_tNN, b_critweight20_r2_tNN (written to bags/).

## Scenario
TO CONFIRM against the August protocol: robot spawn, pre-drive pose, goal,
person start/end and speed.
