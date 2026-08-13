cmake_minimum_required(VERSION 3.8)
project(social_critic)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_mppi_controller REQUIRED)

add_library(social_critic SHARED src/social_critic.cpp)

target_include_directories(social_critic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(social_critic
  rclcpp
  rclcpp_lifecycle
  pluginlib
  std_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  nav2_costmap_2d
  nav2_mppi_controller)

# Tells pluginlib where to find the class declaration. The base package
# is nav2_mppi_controller because CriticFunction lives there.
pluginlib_export_plugin_description_file(nav2_mppi_controller social_critic_plugin.xml)

install(TARGETS social_critic
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include/)

ament_export_include_directories(include)
ament_export_libraries(social_critic)
ament_export_dependencies(
  rclcpp rclcpp_lifecycle pluginlib std_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs nav2_costmap_2d nav2_mppi_controller)

ament_package()
