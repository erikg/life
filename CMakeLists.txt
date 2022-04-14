cmake_minimum_required(VERSION 3.12)
#project(life LANGUAGES C CXX CUDA)

project(life LANGUAGES C CXX)
find_package(SDL2 REQUIRED)
#find_package(OpenGL REQUIRED)

set(SRCS cell.cxx life.cxx main.c disp.c)
include_directories("${CMAKE_SOURCE_DIR}" "${SDL2_INCLUDE_DIRS}")
link_libraries(${SDL2_LIBRARIES})
add_executable(life ${SRCS})
