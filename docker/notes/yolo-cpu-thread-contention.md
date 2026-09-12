# Fix: YOLO CPU thread contention degrading simulation real-time factor

## Symptom

With the full stack running (Gazebo GUI + Nav2 + RViz + social_perception
nodes including yolo_detector), Gazebo's simulation real_time_factor (RTF)
was unstable, oscillating roughly between 0.07 and 1.0, averaging around
0.3 (i.e. the simulation ran at roughly 30% of real speed on average).

## Investigation

CPU usage (`htop`) showed all cores at 70-80% with no single dominant
process, and a cluster of ~8-10 near-identical PIDs all belonging to a
single `yolo_detector` process — confirmed via `ps -eo pid,ppid,nlwp,cmd`
that one PID had `NLWP` (thread count) in the dozens. This pattern —
many threads, no GPU on this machine (YOLO/PyTorch running on CPU only)
— pointed at PyTorch's default CPU thread pool oversubscribing the
available cores.

**Important finding from later isolation testing:** YOLO/thread
contention was NOT the dominant cause of the RTF problem. Running
`gz sim -s -r <world>.sdf` (Gazebo server only, no GUI, no Nav2, no
perception nodes at all) gave a perfectly stable RTF near 1.0. Adding
Gazebo's GUI client back (still nothing else running) reintroduced the
same 0.07-1.0 oscillation. **Gazebo's own GUI rendering, not YOLO, is
the primary cause of RTF instability on this hardware** (confirmed:
Intel integrated GPU via Mesa/Iris, hardware acceleration genuinely
active — this is a real compute ceiling from sharing CPU+GPU silicon,
not a misconfiguration). See the headless launch files
(`sim_headless.launch.py` / `turtlebot4_gz_headless.launch.py`) for the
actual fix to that primary cause.

**This fix (capping YOLO's thread pools) is a secondary, smaller
contributor** — real, but not the main story. With Gazebo's GUI
disabled (headless mode), this fix matters less. With the GUI enabled
(normal development/visual sessions), this fix reduces some of the
additional CPU contention YOLO adds on top of the GUI's own load, but
will NOT by itself bring RTF back to 1.0 while the GUI is running.

## Root cause (of the YOLO-specific contribution)

`torch.set_num_threads(N)`, called in Python, only configures PyTorch's
own intra-op scheduling layer. It does **not** configure the underlying
BLAS/OpenMP libraries (MKL, OpenBLAS) that PyTorch's CPU backend
actually dispatches matrix-multiply work to — those libraries read
their own thread count from environment variables (`OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) at process/library
initialization, which can happen before `torch.set_num_threads()` is
ever called in your script, and is not affected by it regardless of
timing.

Practical symptom of this: adding `torch.set_num_threads(2)` to
`yolo_detector.py` alone produced only a partial improvement (CPU usage
dropped somewhat, but load average actually increased temporarily in
one test, and the dominant contention pattern in `htop` persisted) —
confirming the Python-side call alone was insufficient.

## Fix

1. **In code** (`yolo_detector.py`, near the top, before any model
   loading):
   ```python
   import torch
   torch.set_num_threads(2)
   ```

2. **In the environment** (set before any Python process starts, since
   BLAS libraries read these at their own init time): added to
   `/root/.bashrc` via the Dockerfile, so every shell in the container
   has them set automatically:
   ```bash
   export OMP_NUM_THREADS=2
   export MKL_NUM_THREADS=2
   export OPENBLAS_NUM_THREADS=2
   export NUMEXPR_NUM_THREADS=2
   ```

Both are needed together — the Python call alone is insufficient; the
environment variables alone would leave PyTorch's own scheduling layer
unconstrained (though in practice the BLAS-level cap is the larger
effect of the two).

## Caveats / things to watch for in the future

- **`2` is a starting guess, not a measured optimum.** It was chosen as
  "low enough to reduce contention, not so low that YOLO inference
  itself crawls." If YOLO's own per-frame inference time feels too slow
  after this change, try `3` or `4` and re-measure — there's a genuine
  tradeoff between giving YOLO more threads (faster per-frame) and
  leaving more headroom for everything else running concurrently
  (Gazebo, Nav2, RViz).
- **This does not fix the dominant RTF problem** (Gazebo's GUI
  rendering). Don't expect RTF to return to ~1.0 from this fix alone
  while running with the GUI. Use the headless launch setup for actual
  evaluation runs where RTF stability matters.
- If you ever add MobileCLIP or another CPU-bound model to the
  pipeline, the same BLAS thread-pool oversubscription risk applies to
  it too — check `htop` for the same "many near-identical PIDs from one
  process" pattern, and consider whether it needs the same environment
  variable treatment, or whether the existing `.bashrc`-level caps
  already cover it (they should, since they're process-wide environment
  variables, not specific to `yolo_detector.py`).

## Update 2026-07-29: both hypotheses in this file were measured and are wrong

Everything above describes two candidate causes for the RTF problem:
YOLO's CPU thread contention (this file's original subject, later
demoted to "secondary contributor") and Gazebo's GUI rendering (later
promoted to "the primary cause"). Both were measured directly on
2026-07-29. **Neither is the dominant cause.**

### Measurement method

`/clock` publishes once per Gazebo physics step. `empty_human.sdf` sets
`max_step_size` to 0.003, so **RTF 1.0 corresponds to ~333 Hz on
`/clock`**. This gives a direct, one-command RTF probe:

```bash
ros2 topic hz /clock

```

An equivalent probe, when the perception pipeline is running:
`predicted_person_cloud_node.py` publishes on a 10 Hz sim-time timer,
so `ros2 topic hz /predicted_person_cloud` divided by 10 is the RTF.

Note the two probes have **different denominators** (333 vs 10). Mixing
them up produces nonsense — this was briefly done during the session
and made a 0.43 RTF look like 0.14.

### Results

| Configuration | RTF |
|---|---|

| Full stack, Gazebo GUI enabled | ~0.39 |
| Full stack, headless (`headless:=true`) | ~0.41 |
| Headless, `yolo_detector.py` killed | ~0.43 |

- **Disabling the GUI is worth ~5%**, not the large effect claimed
  above. The earlier "GUI is the primary cause" conclusion came from
  comparing a bare `gz sim -s -r <world>.sdf` against the full stack —
  which changes far more than just the GUI.

- **Killing YOLO entirely is worth ~2 percentage points**, i.e. nothing.
  The 249% CPU figure for `yolo_detector.py` is real but irrelevant.

### Why the reasoning was wrong

Gazebo's physics loop is **single-threaded**. Its speed is bounded by
how fast one core can step it, not by how many cores are free. YOLO
spreading across other cores was never competing for the resource that
actually gates the simulation. Capping YOLO's thread pools protected
something that could not benefit from the protection.

This is the core misconception worth remembering: *high CPU usage by
process A does not imply process B is being starved*, when B is
single-threaded and the machine has spare cores. `htop` showing all
cores busy is not evidence of contention with a single-threaded
consumer.

### What the remaining ~0.57 actually is

By elimination: the robot's own simulation cost. `gz_ros2_control`,
the sensor plugins (RPLidar, OAK-D RGB + depth), and the ros_gz bridge
traffic. Rendering the depth and RGB streams every physics step is the
most likely single contributor.

**This was deliberately not pursued further.** Reducing it means
lowering camera resolution or sensor update rates, which changes what
the perception pipeline sees, which breaks comparability with all
existing recorded bags. That is a much larger decision than a launch
flag and should not be made mid-experiment.

### What to keep, and what to revisit

**Keep headless mode** (`headless:=true`, see
`launch_overrides/sim.launch.py` and `notes/` on that change). Although
the throughput gain is small, the **jitter reduction is large and
real**:

| | mean rate | min | max | std dev |
|---|---|---|---|---|

| GUI | 3.9 Hz | 0.000 s | 0.422 s | 0.078 |
| headless | 4.1 Hz | 0.198 s | 0.295 s | 0.024 |

Std dev dropped ~3x and the `min: 0.000s` bursts disappeared entirely.
Under the GUI the simulation stalled and caught up in clumps; headless
it steps evenly. Bursty stepping is exactly what causes control loops
to miss their rate, and it was measurably inflating run durations in
the slower (ablation) condition — 38–54 s under the GUI vs 34–40 s
headless across both conditions.

**Revisit the thread caps.** `OMP_NUM_THREADS=2` and friends (Dockerfile
section 11) plus `torch.set_num_threads(2)` in `yolo_detector.py` were
set to protect an RTF that they do not protect. On a 16-thread machine
this is likely leaving YOLO inference speed on the table for no benefit.

Suggested next step, **not yet performed**: baseline
`ros2 topic hz /person_positions_map` at 2 threads, then try 6 and
re-measure, watching `/clock` to confirm nothing else degrades. Open a
fresh shell after editing — BLAS libraries read these variables at
process start, so an existing terminal keeps the old values and would
show a misleading "no change" result.

Do NOT change this mid-experiment: faster inference means more frequent
detections into the KF, which may alter prediction behaviour and
confound any in-progress trial set.

### Impact on recorded results

None. All conclusions in the evaluation rest on RTF-safe geometric
metrics (`min_distance`, `path_ratio`, `commit_dist`), which are
computed from positions rather than timestamps and are unaffected by
simulation speed. `analyse_avoidance.py` separates these from
RTF-sensitive metrics (`duration_s`, `mean_speed`) precisely for this
reason.
