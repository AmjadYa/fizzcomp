cmake_minimum_required(VERSION 3.0.2)
project(controller)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

## Find Qt5 components
find_package(Qt5 REQUIRED COMPONENTS Widgets)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS geometry_msgs rospy sensor_msgs std_msgs
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
)

## Install Python scripts
catkin_install_python(PROGRAMS
  src/controller_gui.py  # Adjust the path if needed
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
