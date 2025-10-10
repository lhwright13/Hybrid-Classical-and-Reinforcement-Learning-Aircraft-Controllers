cmake_minimum_required(VERSION 3.15)
project(aircraft_controls VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" ON)
option(BUILD_TESTS "Build C++ unit tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Find Python and pybind11 for Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "Python3 version: ${Python3_VERSION}")
    message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
    message(STATUS "pybind11 version: ${pybind11_VERSION}")
endif()

# Core C++ library
add_library(aircraft_controls_core
    cpp/src/pid_controller.cpp
    # Add more source files as they are created
)

target_include_directories(aircraft_controls_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Set library properties
set_target_properties(aircraft_controls_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "cpp/include/pid_controller.h"
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(aircraft_controls_bindings
        cpp/bindings/bindings.cpp
    )

    target_link_libraries(aircraft_controls_bindings PRIVATE
        aircraft_controls_core
    )

    # Install bindings to project root so Python can import them
    set_target_properties(aircraft_controls_bindings PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# C++ tests (optional, using Google Test)
if(BUILD_TESTS)
    enable_testing()
    # Google Test integration would go here
    # For now, we'll use pytest for all testing
endif()

# Install targets
install(TARGETS aircraft_controls_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Aircraft Controls Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "============================================")
message(STATUS "")
