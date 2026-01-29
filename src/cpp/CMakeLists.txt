cmake_minimum_required(VERSION 3.10)
project(perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(perception SHARED geometry.cpp map.cpp bindings.cpp)
