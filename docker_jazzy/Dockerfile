# =====================================================
# Thesis Social Navigation — ROS 2 Jazzy + Gazebo + Nav2
# Base: TurtleBot4 simulation stack on Ubuntu Noble
#
# Fix rationale lives in notes/*.md — see pointers below.
# =====================================================

FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy


# -----------------------------------------------------
# 1. Base system tools
# -----------------------------------------------------
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


# -----------------------------------------------------
# 2. FIX: pin setuptools for colcon symlink installs
#    -> notes/setuptools-pin.md
# -----------------------------------------------------
RUN pip install "setuptools==70.0.0" --break-system-packages


# -----------------------------------------------------
# 3. Python venv for YOLO / Ultralytics
#    (kept separate from system Python to avoid
#    polluting the ROS Python environment)
#
#    FIX: setuptools pinned here too, matching the system
#    pin in section 2. PYTHONPATH below exposes this venv's
#    site-packages globally, so colcon (a Python program)
#    would otherwise pick up whichever setuptools "wins" the
#    PYTHONPATH order -- if they disagree, colcon's behavior
#    becomes version-dependent on which one shadows the other.
#    -> notes/setuptools-pin.md
# -----------------------------------------------------
RUN apt update && apt install -y python3-venv && \
    python3 -m venv /root/venv && \
    /root/venv/bin/pip install --upgrade pip && \
    /root/venv/bin/pip install "setuptools==70.0.0" && \
    /root/venv/bin/pip install --no-cache-dir \
        "numpy==1.26.4" \
        ultralytics \
        "lap>=0.5.12"

# Make ROS use venv packages
ENV PYTHONPATH=/root/venv/lib/python3.12/site-packages:$PYTHONPATH


# -----------------------------------------------------
# 4. Pre-download Gazebo Fuel models
#    (avoids a slow/flaky download on first sim launch)
# -----------------------------------------------------
RUN mkdir -p /root/.gz/fuel && \
    gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/Warehouse || true && \
    gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/person_standing || true


# -----------------------------------------------------
# 5. ROS / Gazebo control + navigation packages
# -----------------------------------------------------
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


# -----------------------------------------------------
# 6. FIX: force-sync diagnostic_updater with controller_manager
#    -> notes/diagnostic-updater-abi.md
# -----------------------------------------------------
RUN apt update && apt install -y --only-upgrade \
    ros-jazzy-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------
# 7. TurtleBot4 packages
# -----------------------------------------------------
RUN apt update && apt install -y \
    ros-jazzy-turtlebot4-desktop \
    ros-jazzy-turtlebot4-simulator \
    ros-jazzy-irobot-create-nodes \
    && rm -rf /var/lib/apt/lists/*

# Fix simulated RPLidar constant-range issue (ogre -> ogre2)
RUN sed -i 's/<render_engine>ogre<\/render_engine>/<render_engine>ogre2<\/render_engine>/g' \
    /opt/ros/jazzy/share/irobot_create_description/urdf/create3.urdf.xacro


# -----------------------------------------------------
# 8. ROS dependency tools
# -----------------------------------------------------
RUN rosdep init || true
RUN rosdep update


# -----------------------------------------------------
# 9. Workspace
# -----------------------------------------------------
WORKDIR /root/thesis_social_navigation_ws

ENV GZ_GUI_PLUGIN_PATH=/opt/ros/jazzy/share/turtlebot4_gz_gui_plugins/lib:$GZ_GUI_PLUGIN_PATH


# -----------------------------------------------------
# 10. FIX: world/model sync helper
#     -> notes/world-sync-script.md
# -----------------------------------------------------
COPY setup_gazebo_worlds.sh /usr/local/bin/setup_gazebo_worlds.sh
RUN chmod +x /usr/local/bin/setup_gazebo_worlds.sh

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo 'if [ -f /root/thesis_social_navigation_ws/install/setup.bash ]; then source /root/thesis_social_navigation_ws/install/setup.bash; fi' >> /root/.bashrc && \
    echo '/usr/local/bin/setup_gazebo_worlds.sh' >> /root/.bashrc


# -----------------------------------------------------
# 11. FIX: cap CPU thread pools for PyTorch/BLAS
#     -> notes/yolo-cpu-thread-contention.md
#
#     torch.set_num_threads() alone (set in yolo_detector.py)
#     only limits PyTorch's own intra-op thread pool -- it does
#     NOT limit the underlying BLAS/OpenMP libraries (MKL,
#     OpenBLAS) that PyTorch's CPU backend actually calls into
#     for the heavy matrix-multiply work. Those libraries read
#     their thread count from environment variables at process
#     start, before torch.set_num_threads() ever runs in Python,
#     so the Python-side call alone was insufficient.
#
#     Without these caps, YOLO inference (no GPU on this machine)
#     spawns enough threads to saturate every CPU core, starving
#     Gazebo's physics loop and Nav2/RViz of CPU time. Confirmed
#     empirically: real_time_factor dropped to ~0.07-0.3 with
#     YOLO running uncapped, vs. a stable ~1.0 with Gazebo's GUI
#     disabled and no YOLO running at all (see notes file).
#
#     Values of 2 are a starting point, not a proven optimum --
#     re-tune against your actual hardware/workload if YOLO
#     inference itself feels too slow with this cap in place.
# -----------------------------------------------------
RUN echo 'export OMP_NUM_THREADS=2' >> /root/.bashrc && \
    echo 'export MKL_NUM_THREADS=2' >> /root/.bashrc && \
    echo 'export OPENBLAS_NUM_THREADS=2' >> /root/.bashrc && \
    echo 'export NUMEXPR_NUM_THREADS=2' >> /root/.bashrc


# -----------------------------------------------------
# Default command
# -----------------------------------------------------
CMD ["bash"]