FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp


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
        
ENV PYTHONPATH=/root/venv/lib/python3.12/site-packages:$PYTHONPATH

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
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo 'if [ -f /root/thesis_social_navigation_ws/install/setup.bash ]; then source /root/thesis_social_navigation_ws/install/setup.bash; fi' >> /root/.bashrc

# ---------------------------------------------------
# Default command
# ---------------------------------------------------
CMD ["bash"]