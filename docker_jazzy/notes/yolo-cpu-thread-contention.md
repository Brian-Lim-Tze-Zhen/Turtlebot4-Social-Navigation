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
