# Fix: diagnostic_updater / controller_manager ABI mismatch

## Symptom

On the very first launch after a fresh image build, the Gazebo server
process crashes immediately with:

```
gz sim server: symbol lookup error: /opt/ros/jazzy/lib/libcontroller_manager.so:
undefined symbol: _ZN18diagnostic_updater7UpdaterC1ESt10shared_ptrIN6rclcpp15node_interfaces17NodeBaseInterfaceEES1_INS3_18NodeClockInterfaceEES1_INS3_20NodeLoggingInterfaceEES1_INS3_23NodeParametersInterfaceEES1_INS3_19NodeTimersInterfaceEES1_INS3_19NodeTopicsInterfaceEEdh
```

This kills the Gazebo **server** outright (not just a node), which cascades:

- `gz_ros_control` never finishes loading `controller_manager`
- the diff-drive / wheel hardware components never activate
- `odom -> base_link` TF is never published
- Nav2's costmaps time out waiting for transforms
- RViz cannot resolve `map`-frame content (including the loaded map)

So "the map won't show in RViz" was a downstream symptom of this crash,
not a map-server or RViz config problem.

## Root cause

The mangled symbol is the constructor for `diagnostic_updater::Updater`.
`ros-jazzy-controller-manager` was built against one version of
`diagnostic_updater`'s constructor signature; the `diagnostic_updater`
package actually installed in the image was an older build whose
constructor ABI didn't match.

Confirmed via build timestamps in `apt list --installed`:

```
ros-jazzy-controller-manager    4.45.2-1noble.20260615.164916   (built 2026-06-15)
ros-jazzy-diagnostic-updater    4.2.6-1noble.20260412.045349    (built 2026-04-12)
```

Two months apart. The ROS apt repo had moved `controller-manager`
forward without a correspondingly fresh `diagnostic-updater` being
pulled in by the same `apt install` transaction at image build time —
both are nominally "jazzy" packages, but package-level ABI compatibility
isn't guaranteed just because the ROS distro name matches.

## Fix

Explicitly re-run an upgrade of `diagnostic-updater` immediately after
installing the packages that depend on `controller-manager`
(`gz-ros2-control`, `ros2controlcli`), so apt resolves it against
whatever is current in the same build:

```dockerfile
RUN apt update && apt install -y --only-upgrade \
    ros-jazzy-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*
```

Verified working: after this, `diagnostic-updater` upgraded from
`4.2.6` (April build) to `4.2.7` (June 15 build), matching
`controller-manager`'s build date, and the symbol lookup error did
not recur. `controller_manager` loaded cleanly, both wheel hardware
components initialized and activated, and TF began flowing correctly
between `odom` and `base_link`.

## Caveats / things to watch for in the future

- This fix addresses *today's* specific version skew. If the ROS apt
  repo rotates again and reintroduces a different mismatch (e.g.
  `controller-manager` gets bumped further ahead of some other
  dependency), the same class of crash could recur in a different form.
- If this starts happening again after a rebuild, check build
  timestamps the same way (`apt list --installed | grep -E
  "diagnostic-updater|controller-manager"`) before assuming the fix
  has "stopped working" — it may just need re-pointing at a newer
  diagnostic-updater version.
