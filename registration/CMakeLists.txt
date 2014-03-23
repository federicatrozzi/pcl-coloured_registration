cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(finalProva)

find_package(PCL 1.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(-Wno-deprecated -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

add_executable (coloredregistration coloredregistration.cpp)
target_link_libraries (coloredregistration ${PCL_LIBRARIES})