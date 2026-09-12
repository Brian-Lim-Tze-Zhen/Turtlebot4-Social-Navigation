# Day Summary — 2026-07-04 (Full Session)

## What We Did Today

### Step 1 — KF Prediction Accuracy Evaluation ✅

**Pipeline fixes:**
- `move_person_gazebo.py` fully rewritten:
  - Ground truth publisher added (`/person_ground_truth` as `PoseArray`)
  - Yaw fix: `+π/2` offset so model faces direction of travel (x: 3→5 = +π/2, x: 5→3 = −π/2)
  - dt clamp ceiling raised to `update_dt * 3.0` (was silently halving speed at 0.5s dt)
  - 1.5s endpoint pause before reversing (more realistic + helps KF settle velocity)
  - Threaded `set_model_pose` calls (non-blocking)
- `yolo_detector.py`: `max_depth` 5.0→6.0, `min_depth_pixels` 20→50, model path updated to `models/`, bbox fields appended to message
- Installed `pandas` into `/root/venv`, added to Dockerfile

**Evaluation results (Step 1 final):**

| Metric | KF Prediction | Naive Baseline |
|---|---|---|
| Mean error | 0.289 m | 0.402 m |
| Median error | 0.199 m | 0.402 m |
| Max error | 0.863 m | 0.863 m |
| Improvement over naive | **28.1%** | — |
| KF better than naive | **69.0%** of predictions | — |
| Matched samples | 984 / 1072 | — |
| Velocity bias | **−5.5%** | — |
| Prediction horizon | 2.0 s | — |

**Plots generated (saved to `eval_plots/`):**
- `prediction_error_track_1.png` — KF vs naive error over time + predicted vs actual scatter
- `track_1.png` — X position over time + signed X velocity over time
- `speed_comparison_track_1.png` — KF speed vs GT speed + trajectory context

**Key finding:** Direction-reversal moments produce ~0.5m mean error vs 0.244m during steady walking. Overshoot is conservative (robot avoids zone person never enters) — not a safety risk.

**Consistent colour scheme applied across all 3 plots:**
- Current/filtered position → Blue `#2196F3`
- Predicted position → Orange `#FF6F00`
- Ground truth → Black `#000000`
- KF error / KF speed → Green `#2E7D32`
- Naive baseline → Grey `#9E9E9E`

---

### Step 2 — Costmap Visual Comparison ✅

- Local costmap size temporarily changed to 6×6m for screenshots (reverted to 3×3m after)
- **Screenshot 1** (`costmap_with_prediction.png`): ellipse extending ahead of person in walking direction — `social_nav2.yaml` active
- **Screenshot 2** (`costmap_without_prediction.png`): symmetric blob at current position only — `social_nav2_no_predicted_cloud.yaml` active, no KF/cloud nodes running

---

### PowerPoint Presentation — Step 1 ✅

6-slide deck created (`step1_kf_evaluation.pptx`):
1. Title + experimental setup
2. KF tracking quality (placeholder for `track_1.png`)
3. Prediction accuracy vs naive baseline (placeholder for `prediction_error_track_1.png` left panel)
4. Prediction overshoot — known limitation (placeholder for right panel)
5. Velocity estimate accuracy (placeholder for `speed_comparison_track_1.png`)
6. Summary with key numbers

Speaker notes filled in for all slides.

---

### Repository Reorganisation ✅

**New folder structure:**
```
ros2_ws/
├── config/
├── eval_plots/          ← evaluation output plots
├── eval_scripts/        ← evaluate_prediction_accuracy.py, compare_true_vs_filtered_speed.py, plot_kf_log.py
├── maps/
├── models/              ← yolov8n.pt, yolov8s.pt
├── simulation_models/
├── src/
├── CLAUDE.md
├── debug_notes_2026-06-24.md
└── run_sim.sh
```

**Deleted:**
- `ros2_ws/worlds/` — stale duplicate of `simulation_models/worlds/`
- `ros2_ws/YOLO ByteTrack Position_screenshot_26.05.2026.png` — old screenshot
- `plot_trajectory_offset_figure.py` — replaced by better plots
- Old plot PNGs from wrong location inside `social_perception/`

**`Thesis_Procedure` updated:**
- World references changed from `two_human` to `empty_human`
- Ablation launch fixed to use `social_nav2_no_predicted_cloud.yaml`
- All commands changed from `ros2 run` to `python3` with full paths
- Eval scripts paths updated to `eval_scripts/`
- `compute_min_distance.py` documented as live ROS2 node (not post-processing script)
- 3 new troubleshooting rows added

---

### Git Commit ✅

Commit: `eval: reorganise scripts, fix move_person_gazebo, update yolo depth params`
Push: `main` → `origin/main` (c980e7d)

---

## Pending / Next Session

- [ ] **Step 3**: Run ablation trials — `social_nav2.yaml` vs `social_nav2_no_predicted_cloud.yaml`
  - Drop `compute_min_distance.py` into `eval_scripts/` (file drafted, not yet in repo)
  - Run multiple trials per config, collect min-distance CSVs
  - Generate box plot + comparison table
- [ ] Insert plot PNGs into `step1_kf_evaluation.pptx` placeholder boxes
- [ ] Insert costmap screenshots into Step 2 slides (slides not yet created for Step 2)
- [ ] Consider adding stop/pause confidence gating to prediction pipeline (deferred)
- [ ] MobileCLIP facing-classification retest (pending clean Gazebo server restart)
- [ ] Merge `feature/group-formation-detection` branch after cleanup
- [ ] Remove debug saves (`debug_clip_crop.png`, `debug_full_frame.png`) from `group_formation_detector.py`
- [ ] Delete stray file at repo root: `'sify_facing() with real MobileCLIP-S1 inference"'`
