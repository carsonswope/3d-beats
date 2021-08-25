cmake_minimum_required(VERSION 3.10)
project(CPP_GROUPING CXX)
set(CMAKE_CXX_STANDARD 14)

# cython always links to dynamic runtime, and so we have to too!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD")

add_library(CppGrouping STATIC grouping.h grouping.cpp)
set_target_properties(CppGrouping PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET CppGrouping PROPERTY MSVC_RUNTIME_LIBRARY MultiThreadedDLL)

# add_executable(CppGrouping_Test grouping_test.cpp)
# target_link_libraries(CppGrouping_Test LINK_PUBLIC CppGrouping)
