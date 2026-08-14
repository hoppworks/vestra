cmake_minimum_required(VERSION 3.18)
project(vestra_cpp_pr2_oracle LANGUAGES C CXX)

# This harness deliberately embeds the pinned depth-anything.cpp PR checkout as
# a subproject.  It never modifies that checkout: only this build directory and
# the supplied fixture/output files are written.
if(NOT DEFINED CPP_PR2_SOURCE)
    message(FATAL_ERROR "Set CPP_PR2_SOURCE to a pristine depth-anything.cpp PR #2 checkout")
endif()
if(NOT EXISTS "${CPP_PR2_SOURCE}/src/stream.cpp")
    message(FATAL_ERROR "CPP_PR2_SOURCE does not contain src/stream.cpp: ${CPP_PR2_SOURCE}")
endif()

set(DA_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(DA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${CPP_PR2_SOURCE}" cpp-pr2)

add_executable(vestra_cpp_stream_fixture_dump stream_fixture_dump.cpp)
target_link_libraries(vestra_cpp_stream_fixture_dump PRIVATE depthanything)
target_include_directories(vestra_cpp_stream_fixture_dump PRIVATE
    "${CPP_PR2_SOURCE}/src"
    "${CPP_PR2_SOURCE}/include")

# End-to-end counterpart to the model-free fixture harness. It invokes the
# pinned C++ C API over an already-decoded, lexically ordered frame directory
# and writes raw stream evidence without a server, browser, or resampling layer.
add_executable(vestra_cpp_capi_stream_dump capi_stream_dump.cpp)
target_link_libraries(vestra_cpp_capi_stream_dump PRIVATE depthanything)
target_include_directories(vestra_cpp_capi_stream_dump PRIVATE
    "${CPP_PR2_SOURCE}/include")

# Deterministic diagnostic evidence for the exact PR #2 closed-trajectory
# regression. It only writes a VPS1 fixture; it never modifies or links to the
# pinned C++ source tree.
add_executable(vestra_cpp_closed_loop_fixture make_closed_loop_fixture.cpp)
target_include_directories(vestra_cpp_closed_loop_fixture PRIVATE
    "${CPP_PR2_SOURCE}/tests")
