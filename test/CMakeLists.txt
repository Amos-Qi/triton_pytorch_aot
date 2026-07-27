# Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Standalone, CPU-only unit tests for the pure CUDA-graph bucketing helpers
# (src/cuda_graph_bucketing.h). Deliberately independent of the main backend
# build: no torch / Triton / CUDA, so it compiles and runs on any host CI.
#
# Usage:
#   cmake -S test -B build-test
#   cmake --build build-test -j
#   ctest --test-dir build-test --output-on-failure

cmake_minimum_required(VERSION 3.18)
project(cuda_graph_bucketing_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
)
# Keep googletest's CRT linkage aligned with ours on all platforms.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(cuda_graph_bucketing_test cuda_graph_bucketing_test.cc)
target_include_directories(
  cuda_graph_bucketing_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(cuda_graph_bucketing_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cuda_graph_bucketing_test)
