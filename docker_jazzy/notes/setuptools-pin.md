# Fix: setuptools version pin for `colcon build --symlink-install`

## Symptom

Building the workspace with `colcon build --symlink-install` fails on
the `social_perception` package (an `ament_python` package) with:

```
usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
   or: setup.py --help [cmd1 cmd2 ...]
   or: setup.py --help-commands
   or: setup.py cmd --help
error: option --editable not recognized
```

Building the same package *without* `--symlink-install` (plain
`colcon build`) works fine.

## Why `--symlink-install` matters

For Python (`ament_python`) packages, plain `colcon build` *copies*
source files into the install space. Every edit to a `.py` file
requires a rebuild before `ros2 run`/`ros2 launch` will see the change.

`--symlink-install` creates symlinks from the install space back to
the source tree instead of copying. Edits to source files take effect
immediately on the next `ros2 run`, with no rebuild step — a much
faster edit/test loop for active development on Python nodes (e.g.
the perception pipeline).

This has no effect on compiled (C++) packages, which always require
an actual rebuild regardless of this flag.

## Root cause

`colcon`'s `ament_python` build step still relies on the legacy
`setup.py develop --editable` install path to create the symlinked
install. Modern `setuptools` (roughly 66+, and especially the latest
releases at time of writing) has deprecated/removed support for the
`--editable` flag in that code path, causing the error above.

The first attempted fix — pinning to an arbitrary older version
(`setuptools<66`, which resolved to `65.7.0`) — went too far in the
other direction. That version's bundled `pkg_resources` calls
`pkgutil.ImpImporter`, an attribute that existed through Python 3.11
but was **removed in Python 3.12**, which this image runs. Result:

```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'.
Did you mean: 'zipimporter'?
```

So there's a compatibility window: too new breaks `--editable`,
too old breaks under Python 3.12. There's also a ceiling imposed by
`colcon-core` itself (surfaced via pip's dependency-conflict warning
when testing versions):

```
colcon-core 0.20.1 requires setuptools<80,>=30.3.0
torch 2.12.0 requires setuptools<82
```

## Fix

```dockerfile
RUN pip install "setuptools==70.0.0" --break-system-packages
```

`70.0.0` was tested directly in this image and confirmed to satisfy
all three constraints simultaneously:
- New enough for Python 3.12 (no `pkgutil.ImpImporter` dependency)
- Old enough to still support the legacy `--editable` path colcon needs
- Within `colcon-core`'s `<80` ceiling

After pinning, `colcon build --symlink-install` completed cleanly:
```
Starting >>> social_perception
Finished <<< social_perception [0.76s]
Summary: 1 package finished [0.82s]
```

## Caveats / things to watch for in the future

- This is a narrow compatibility window specific to this combination
  of Python 3.12, this colcon/colcon-core version, and this ROS distro.
  It is not guaranteed to remain valid if any of those move (e.g. a
  future base image ships a newer colcon-core with a different cap,
  or a different Python minor version).
- If this starts failing again after a rebuild, the diagnostic
  approach is: check which of the two error signatures comes back.
  - `AttributeError: ... ImpImporter` → current pin is too *old*, try
    a newer version.
  - `error: option --editable not recognized` → current pin is too
    *new*, try an older version.
  Bisect from `70.0.0` in the appropriate direction within the
  `>=66, <80` window.
- This is a developer-convenience fix (faster iteration on Python
  perception code), not a correctness fix. If it ever becomes
  troublesome to maintain across rebuilds, falling back to plain
  `colcon build` (no symlink) is a safe, fully functional alternative
  — it just means rebuilding after each Python source edit.
