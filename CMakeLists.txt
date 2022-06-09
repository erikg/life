cmake_minimum_required(VERSION 3.12)

find_package(SDL2 REQUIRED)

project(life LANGUAGES C CXX CUDA)
set(SRCS cell.cxx life.cu main.c disp.c)
#project(life LANGUAGES C CXX)
#set(SRCS cell.cxx life.cxx main.c disp.c)

include_directories("${CMAKE_SOURCE_DIR}" "${SDL2_INCLUDE_DIRS}")
link_libraries(${SDL2_LIBRARIES})
add_executable(life ${SRCS})
set_property(TARGET life PROPERTY CUDA_ARCHITECTURES 50 60 72)
