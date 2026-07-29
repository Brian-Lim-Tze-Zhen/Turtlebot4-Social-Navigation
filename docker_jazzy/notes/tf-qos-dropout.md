# Fix: rosbag2 silently recording no `/tf` due to QoS durability mismatch

## Symptom

Roughly 50% of recorded evaluation runs contained no `map -> odom`
transform in `/tf`, making `min_distance` uncomputable in
`eval_scripts/analyse_avoidance.py` (the script warns and skips the
metric). The failure looked random: the same trial protocol, run the
same way, would sometimes produce a usable bag and sometimes not.

Critically, `ros2 bag record` reported **success** in both cases. The
bag was created, other topics recorded normally, and the run appeared
to complete cleanly. The loss was only discovered at analysis time.

## Investigation

Initially ruled out, in order:

- CPU starvation / RTF collapse dropping messages
- Config differences between the full and ablation conditions
- Duplicate AMCL nodes publishing conflicting transforms
- Duplicate RViz instances
- AMCL's `update_min_d` / `update_min_a` suppressing publication while
  the robot was stationary

None of these explained the pattern. The AMCL threshold hypothesis was
the most plausible of the five — but `nav2_amcl` republishes its last
transform on every laser callback regardless of whether the filter
updated, so a stationary robot alone does not stop `map -> odom`.

The actual cause was caught live, in the startup output of a recording
that was being watched rather than backgrounded:

```
[WARN] [rosbag2_recorder]: New publisher discovered on topic '/tf',
offering incompatible QoS. No messages will be sent to it.
Last incompatible policy: DURABILITY_QOS_POLICY

[WARN] [rosbag2_recorder]: A new publisher for subscribed topic /tf was
found offering RMW_QOS_POLICY_DURABILITY_VOLATILE, but rosbag2 already
subscribed requesting RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL.
Messages from this new publisher will not be recorded.

```

## Root cause

`rosbag2` does not impose a fixed QoS profile on a subscribed topic.
It **infers** one from the publishers that exist at the moment it
subscribes, then holds that profile for the lifetime of the recording.

`/tf` has many publishers in this stack (AMCL, the Gazebo bridge,
`robot_state_publisher`, the controller stack). If rosbag2 happens to
subscribe while a `TRANSIENT_LOCAL` publisher is the only one present,
it requests `TRANSIENT_LOCAL`. Every publisher that appears afterwards
offering `VOLATILE` — which includes AMCL — is then QoS-incompatible,
and **none of its messages are recorded**.

This is why the failure looked random. It depended entirely on the
ordering between `ros2 bag record` starting and the various `/tf`
publishers coming up, which varied run to run depending on how quickly
each step of the manual trial protocol was performed.

It also explains why the failures clustered in ablation runs: those
were recorded in a separate block, with a slightly different rhythm of
launching and recording than the full-config runs.

## Fix

Force `VOLATILE` durability on `/tf` explicitly, so rosbag2 stops
inferring and every publisher is compatible.

`eval_scripts/tf_qos_override.yaml`:

```yaml
/tf:
  durability: volatile
  reliability: reliable
  history: keep_all

```

Add to every `ros2 bag record` invocation:

```bash
ros2 bag record \
  --qos-profile-overrides-path /root/thesis_social_navigation_ws/eval_scripts/tf_qos_override.yaml \
  --topics /person_ground_truth /odom /amcl_pose /tf /tf_static /plan /predicted_person_positions \
  -o bags/<name>

```

`/tf_static` is deliberately NOT overridden. It is genuinely
transient-local by design (late-joining subscribers must receive the
static transforms published before they connected), and rosbag2's
default handling of it is correct.

Verified: ten consecutive trials recorded with the override, zero
missing `map -> odom`, zero discarded runs. Prior to the fix the
failure rate was roughly 50%.

## Caveats / things to watch for in the future

- **Watch the startup output of `ros2 bag record`.** The incompatible-QoS
  warning is the only signal this is happening. It scrolls past quickly
  among the normal "Subscribed to topic" lines, and the recording
  otherwise behaves as if nothing is wrong. If bags are ever recorded
  in a backgrounded terminal or piped to a log, grep the log for
  `incompatible QoS` before trusting the data.

- **The same class of bug can affect any multi-publisher topic**, not
  just `/tf`. If another topic ever shows up empty or partial in a bag
  despite being listed in `--topics`, check QoS before checking
  anything else.

- **Bags recorded before this fix are not automatically invalid.** If
  `analyse_avoidance.py` computes `min_distance` without warning, the
  `map -> odom` transform was present and the run is usable. The fix
  prevents a failure mode; it does not change what is recorded when the
  failure does not occur. The following bags predate the override and
  were verified to analyse cleanly:
  `headon_full_t01/t02/t03`, `headon_ablation_clean_t01/t02/t03`,
  `headon_full_hl_t01`, `headon_ablation_hl_t01`.

- **Cost of this bug:** roughly half of all evaluation runs recorded
  before 2026-07-29 were unusable, and the cause was mis-attributed to
  CPU starvation and AMCL behaviour for some time before being found.
  It was found only by watching a recording start in the foreground.
