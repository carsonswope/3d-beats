cmake_minimum_required(VERSION 3.10)
project(CPP_GROUPING CXX)
set(CMAKE_CXX_STANDARD 14)

add_library(CppGrouping STATIC grouping.h grouping.cpp)
set_target_properties(CppGrouping PROPERTIES LINKER_LANGUAGE CXX)

add_executable(CppGrouping_Test grouping_test.cpp)
target_link_libraries(CppGrouping_Test LINK_PUBLIC CppGrouping)
