cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(pointnet VERSION 1.0.0 LANGUAGES CXX)
list(APPEND CMAKE_PREFIX_PATH "/home/nagaharish/Downloads/cpp/HighFive/build/")
# Files
set(SOURCES src/train.cpp
            src/model.cpp
            src/input_transform.cpp
)

set(HEADERS include/train.h
            include/model.h
            include/input_transform.h
)

set(EXECUTABLE_NAME pointnet)
find_package(HighFive REQUIRED)
find_package(Torch REQUIRED)
add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME}  HighFive)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
)
