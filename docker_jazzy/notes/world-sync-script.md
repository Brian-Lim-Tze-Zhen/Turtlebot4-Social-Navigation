# Fix: world / model sync helper script

## Symptom

Custom Gazebo worlds and models placed in the bind-mounted workspace
are not reliably picked up by `turtlebot4_gz_bringup`'s launch file
when referenced via `model://` URIs under `ros2 launch`, even when
`GZ_SIM_RESOURCE_PATH` is set correctly in the environment.

(Confirmed empirically in this project — the env var alone was not
sufficient to make the launch file resolve custom resources.)

## Fix

`setup_gazebo_worlds.sh` copies custom worlds/models from the
bind-mounted workspace into `turtlebot4_gz_bringup`'s own `worlds/`
directory directly, sidestepping the unreliable resource-path
resolution entirely.

It's installed into the image at `/usr/local/bin/setup_gazebo_worlds.sh`
and invoked from `.bashrc`, so it re-syncs automatically:

- every time a new shell is opened in the container
- after `docker compose build` recreates the image
- after workspace files change between container restarts

See the script itself for the exact list of files/models it syncs and
the copy logic.

## Caveats

- If a new custom world or model is added to the workspace and isn't
  showing up in Gazebo, check this script's contents first — it may
  need an entry added for the new file before it will be picked up.
