FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy



# ---------------------------------------------------
# Base tools
# ---------------------------------------------------
RUN apt update && apt install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    nano \
    python3-pip \
    python3-opencv \
    python3-colcon-common-extensions \
    python3-rosdep \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Python venv for YOLO / Ultralytics
# ---------------------------------------------------
RUN apt update && apt install -y python3-venv && \
    python3 -m venv /root/venv && \
    /root/venv/bin/pip install --upgrade pip && \
    /root/venv/bin/pip install --no-cache-dir \
        "numpy==1.26.4" \
        ultralytics \
        "lap>=0.5.12"

# Make ROS use venv packages
ENV PYTHONPATH=/root/venv/lib/python3.12/site-packages:$PYTHONPATH

# ---------------------------------------------------
# Pre-download Gazebo Fuel models
# ---------------------------------------------------
RUN mkdir -p /root/.gz/fuel && \
    gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/Warehouse || true && \
    gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/person_standing || true

# ---------------------------------------------------
# ROS / Gazebo packages
# ---------------------------------------------------
RUN apt update && apt install -y \
    ros-jazzy-nav2-bringup \
    ros-jazzy-nonpersistent-voxel-layer \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros2controlcli \
    && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------
# TurtleBot4 packages
# ---------------------------------------------------
RUN apt update && apt install -y \
    ros-jazzy-turtlebot4-desktop \
    ros-jazzy-turtlebot4-simulator \
    ros-jazzy-irobot-create-nodes \
    && rm -rf /var/lib/apt/lists/*

# Fix simulated RPLidar constant range issue
RUN sed -i 's/<render_engine>ogre<\/render_engine>/<render_engine>ogre2<\/render_engine>/g' \
    /opt/ros/jazzy/share/irobot_create_description/urdf/create3.urdf.xacro
# ---------------------------------------------------
# ROS dependency tools
# ---------------------------------------------------
RUN rosdep init || true
RUN rosdep update

# ---------------------------------------------------
# Workspace
# ---------------------------------------------------
WORKDIR /root/thesis_social_navigation_ws

# ---------------------------------------------------
# Default ROS setup
# ---------------------------------------------------
ENV GZ_GUI_PLUGIN_PATH=/opt/ros/jazzy/share/turtlebot4_gz_gui_plugins/lib:$GZ_GUI_PLUGIN_PATH

# ---------------------------------------------------
# THESIS FIX: copy the world-sync helper script into the image.
# It re-syncs custom worlds/models from the bind-mounted
# workspace into turtlebot4_gz_bringup's worlds/ directory,
# since that package's launch file does not reliably honor
# GZ_SIM_RESOURCE_PATH for model:// URIs under `ros2 launch`
# (confirmed empirically -- see script header for details).
# Runs every time a new shell starts, so it stays in sync
# even after `docker compose build` recreates the image, and
# even if workspace files change between container restarts.
# ---------------------------------------------------
COPY setup_gazebo_worlds.sh /usr/local/bin/setup_gazebo_worlds.sh
RUN chmod +x /usr/local/bin/setup_gazebo_worlds.sh

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo 'if [ -f /root/thesis_social_navigation_ws/install/setup.bash ]; then source /root/thesis_social_navigation_ws/install/setup.bash; fi' >> /root/.bashrc && \
    echo '/usr/local/bin/setup_gazebo_worlds.sh' >> /root/.bashrc

# ---------------------------------------------------
# Default command
# ---------------------------------------------------


CMD ["bash"]