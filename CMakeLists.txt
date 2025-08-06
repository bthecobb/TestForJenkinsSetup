cmake_minimum_required(VERSION 3.10)
project(PushTestJenkins)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add GoogleTest
add_subdirectory(C:/googletest/install googletest)

# Find all test files
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/*.cpp")

# Remove main.cpp (it should not be compiled twice)
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_SOURCE_DIR}/main.cpp")

# Create test executable
add_executable(runTests ${TEST_SOURCES})

# Link GoogleTest libraries
target_link_libraries(runTests PRIVATE gtest_main)

# Enable Testing
enable_testing()
add_test(NAME AllTests COMMAND runTests)

add_test(NAME SortingTest COMMAND runTests)
