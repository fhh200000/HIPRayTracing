# @file CMakeLists.txt
#
# CMake config file for generic platform.
# SPDX-License-Identifier: WTFPL
#

cmake_minimum_required(VERSION 3.21)
project(HIPRayTracing)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/3rdparty/CMakeModules")
include_directories(Include/)

add_compile_definitions("-DLOGLEVEL_ENABLED=0")

message(CHECK_START "Finding HIP")
find_package(hip REQUIRED)
find_package(rocrand REQUIRED)

get_filename_component(hip_root ${hip_INCLUDE_DIR} DIRECTORY)
get_filename_component(rocrand_root ${rocRAND_INCLUDE_DIR} DIRECTORY)

message(CHECK_PASS "HIP ${hip_VERSION} at ${hip_root}, rocRAND ${rocrand_VERSION} at ${rocrand_root}")
include_directories(${hip_INCLUDE_DIR})
include_directories(${rocRAND_INCLUDE_DIR})

enable_language(HIP)


set(window_implementation "Auto"
    CACHE STRING "switch between window implemetation, if supported")
if (WIN32)
    set (window_implementation_file Window/Win32Window.cpp)
elseif (UNIX)
    message(CHECK_START "Finding Wayland")
    find_package(Wayland COMPONENTS Client)
    if (Wayland_FOUND)
        set (wayland_test_message "${Wayland_VERSION}, at ${Wayland_LIBRARIES}")
        if ((${window_implementation} STREQUAL "Auto") OR (${window_implementation} STREQUAL "Vulkan"))
            message(CHECK_PASS "${wayland_test_message}, selected")
            set (window_implementation_file Window/WaylandWindow.cpp)
            include_directories(Wayland_INCLUDE_DIRS)
        else()
            message(CHECK_PASS "${wayland_test_message}, not used because Window Implemetation is ${window_implementation}")
        endif()
    endif()
endif()

add_executable(HIPRayTracing Device.cpp Kernel.cpp Main.cpp ${window_implementation_file})
target_link_libraries(HIPRayTracing PRIVATE hip::host ${rocrand_LIBRARIES})

set_source_files_properties(Kernel.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(Device.cpp PROPERTIES LANGUAGE HIP)
