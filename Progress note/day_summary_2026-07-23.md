# Day Summary — 2026-07-23

## Goal for the session
Resume `feature/group-formation-detection` branch, get the MobileCLIP-S1 facing-classification path actually running end-to-end for the first time (VLM confirmation of "conversation" detection between two people).

## Starting state
- Branch had the full detection pipeline written (geometry-based conversation/queue detection, MobileCLIP-S1 loading and inference), but the VLM confirmation path was **never actually reachable** — `classify_facing()` always returned `None` immediately because the bbox field it depended on was never populated anywhere upstream.

## What got fixed, in order

### 1. Wired bbox data through the full pipeline (real blocker, not previously flagged)
- **Discovery:** `yolo_detector.py` never captured/forwarded bbox corners; `human_kf_predictor.py` never had a bbox field at all in `/predicted_person_positions`. `group_formation_detector.py` was written against an assumption the upstream pipeline had never actually implemented.
- **A second discovery mid-fix:** `human_kf_predictor.py`'s `/predicted_person_positions` already used field `[9]` for the rotation-gate flag (load-bearing for `predicted_person_cloud_node.py`'s head-on collision fix). My first draft of the fix would have silently overwritten that field with bbox data and broken the rotation gate. Caught before it shipped by re-checking the actual message consumer, not just the message producer.
- **Fix implemented:**
  - `yolo_detector.py`: appends `x1,y1,x2,y2` as 4 trailing fields on `/person_positions_map`.
  - `human_kf_predictor.py`: parses that bbox; republishes it as **field [10]** on `/predicted_person_positions` (field [9] stays the rotation-gate flag, untouched); publishes `"none"` for bbox specifically while coasting (no fresh detection), by design — matches the "fresh bbox only" semantics `group_formation_detector.py`'s docstring already assumed.
  - `group_formation_detector.py`: parsing corrected to read bbox from field `[10]`.
  - Caught and fixed two missing-comma bugs during this (field concatenation bugs, e.g. `"0"+"none"` → `"0none"`) via careful re-verification of pasted files rather than assuming edits landed correctly.

### 2. Built `conversation_test.sdf`
- New static Gazebo world, two `person_standing` models 1.0m apart, posed to face each other using the yaw-offset convention verified in `move_person_gazebo2.py`'s comments (`pose_yaw = heading + π/2`), extended from X-axis (originally verified) to Y-axis (this session) and visually confirmed correct in Gazebo.

### 3. Diagnosed and fixed a camera framing / detection-confidence tradeoff
- Found: getting close enough for YOLO to detect both people confidently cut off their heads (camera VFOV ≈ 56.8°, mount height 0.244m — computed from live `camera_info`, cross-checked against `tf_static` chain).
- Found: backing up far enough for full-body framing dropped YOLO confidence below `conf=0.80`, causing intermittent single-person-only detection. Confirmed via live diagnostic (temporarily lowered conf to 0.30, then settled on 0.70) that this was a real confidence/distance tradeoff, not something else.

### 4. Found and fixed the actual root cause of wrong facing-classification (main result of the session)
Root-caused via several data-driven tests, not guesswork:
- **Removed a candidate explanation (framing):** tested headless, partial-head, and full-body crops — classification stayed wrong across all three (~0.65-0.70 favoring the wrong prompt) until much later findings explained why.
- **Removed a candidate explanation (mirror/orientation invariance):** tested original crop vs. horizontally-flipped version — scores barely moved (<0.03 delta), ruling out orientation-cue blindness as the driver.
- **Removed a candidate explanation (camera viewing angle):** tested broadside, on-axis, and 45° viewing angles — no clean trend, ruled out as the dominant factor.
- **Found the real cause (prompt-embedding bias):** tested the original 3-way prompt set against a real, unambiguous photo of two people clearly facing and talking — "far apart" *still* won (0.627 vs. 0.235), proving the bias was in the prompts themselves, not image content or domain gap.
- **Fix:** replaced the original unbalanced 3-prompt set with a minimal-contrast binary pair (`"two people facing each other"` / `"two people not facing each other"`). Verified correct on both the real photo (0.587) and the actual synthetic Gazebo crop (0.564) — first time this ever classified correctly.

### 5. Characterized and gated a residual framing-sensitivity
- A controlled 6-point distance sweep (1.40m → 3.49m) showed the facing-score cleanly and reliably degrades as distance decreases, flipping to "not facing" at 1.40m.
- Found the real proxy variable: not bbox height (initially assumed, disproven by data — taller boxes occurred at *closer*, worse-framed distances) but **bbox `y1` (top-edge clipping)**. Clean binary split across all 6 trials: `y1=0` always bad/borderline, `y1≥5` always good.
- Implemented `TOP_CLIP_MARGIN_PX=5` gate: skips VLM confirmation entirely (stays "inconclusive") when either person's bbox is clipped at the frame top, rather than risking a confidently-wrong classification. Verified both ends: far-range still classifies correctly, close-range now silently skips instead of flipping wrong.
- **Derived sensing constraint for the thesis:** minimum ~3.0m standoff distance required for reliable conversation-facing classification with this camera (56.8° VFOV, 0.244m mount height) and person model (~1.8m estimated visual height, taller than the 1.5m collision-cylinder proxy that initially under-estimated it).

## Bugs found and NOT yet fixed (flagged, not blocking, logged to memory)
1. **Sticky-confirmation gap:** `detect_groups()` rebuilds `groups` from scratch every 0.3s cycle with no memory of prior VLM confirmation. Once the robot gets within the ~3m framing-gate boundary, a previously-confirmed conversation pair silently drops out of `/social_groups` — the zone would vanish exactly when the robot is closest, the opposite of the intended protective behavior. Not yet fixed; needs a per-pair `confirmed` state that persists on cheap geometry alone once VLM-confirmed.
2. **Track-ID re-identification gap:** every layer of the pipeline (`human_kf_predictor.py`, `group_formation_detector.py`, and the planned `social_group_cloud_node.py`) keys its state by `track_id`. A brief (~2s) camera occlusion causes ByteTrack to assign a new ID on re-detection, which cascades into a full state reset at every layer — KF covariance reset, conversation duration timer reset to zero, future costmap zone flicker. Proposed direction: lidar leg-detection as a continuous position anchor to re-associate track IDs across occlusion, explicitly scoped as *not* solving the 3m distance constraint or facing/identity content — a separate, substantial piece of architecture, not a quick fix.
3. Debug crop save (`debug_clip_crop.png`) was reinstated for this session's diagnosis and is currently still active in `classify_facing()` — needs removing again before merge.

## Also carried over, untouched this session
- Two-person threading scenario's ellipse-heading-swing issue (rotation gate + low-speed atan2 instability) from the prior session — not investigated tonight.
- Queue scenario (Step 3 of the original session goal) — not started.

## Files changed
- `yolo_detector.py`
- `human_kf_predictor.py`
- `group_formation_detector.py`
- `simulation_models/worlds/conversation_test.sdf` (new)
