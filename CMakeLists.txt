cmake_minimum_required(VERSION 3.16)
cmake_policy(VERSION 3.16)

############################################
# Check whether TT_METAL_HOME and ARCH_NAME
# environemnt variables are set
############################################

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME environment variable is not set. Please do:\n\t1) export TT_METAL_HOME=<path_to_tt_metal>\n\t2) source <path_to_tt_metal>/setup.sh")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME environment variable is not set. Please do:\n\texport ARCH_NAME=<arch_name>")
endif()

############################################
# Project setup
############################################
include($ENV{TT_METAL_HOME}/cmake/compilers.cmake)

if (DEFINED ENV{CMAKE_C_COMPILER} AND DEFINED ENV{CMAKE_CXX_COMPILER})
    message(STATUS "Setting C and C++ compiler from environment variables")
    set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
    set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
endif()

if (CMAKE_CXX_COMPILER AND CMAKE_C_COMPILER)
    message(STATUS "Using specifed C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Using specifed C compiler: ${CMAKE_C_COMPILER}")
else()
    message(STATUS "No C or C++ compiler specified, defaulting to Clang-17")
    FIND_AND_SET_CLANG17()
endif()

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(FATAL_ERROR "CMake generation is not allowed within source directory!! Please set a build folder with '-B'!!")
endif()

# Set project after compiler setup
project(tt-llk-example LANGUAGES CXX C)
CHECK_COMPILERS()

############################################################################################################################
# Setting build type flags
#   Will default to Release build unless manually set with -DCMAKE_BUILD_TYPE
############################################################################################################################
# Define valid build types
set(VALID_BUILD_TYPES Debug Release RelWithDebInfo CI)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release build is the default" FORCE)
endif()

# Check if the specified build type is valid
list(FIND VALID_BUILD_TYPES ${CMAKE_BUILD_TYPE} _build_type_index)

if(_build_type_index EQUAL -1)
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}. Valid options are: ${VALID_BUILD_TYPES}")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG=DEBUG")
set(CMAKE_CXX_FLAGS_CI "-O3 -DDEBUG=DEBUG")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default values for variables/options
set(UMD_HOME "${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd")

option(ENABLE_CODE_TIMERS "Enable code timers" OFF)
option(TT_METAL_VERSIM_DISABLED "Disable TT_METAL_VERSIM" ON)
option(ENABLE_TRACY "Enable Tracy Profiling" OFF)


option(ENABLE_BUILD_TIME_TRACE "Enable build time trace (Clang only -ftime-trace)" OFF)
if(ENABLE_BUILD_TIME_TRACE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "Adding compile option: -ftime-trace")
        add_compile_options("-ftime-trace")
    else()
        message(FATAL "ENABLE_BUILD_TIME_TRACE is only supported with Clang")
    endif()
endif()

include(GNUInstallDirs)
set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
set(CMAKE_INSTALL_LIBDIR "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_INSTALL_BINDIR "${PROJECT_BINARY_DIR}/tmp/bin")
set(CMAKE_INSTALL_INCLUDEDIR "${PROJECT_BINARY_DIR}/tmp/include")
set(CMAKE_INSTALL_DATAROOTDIR "${PROJECT_BINARY_DIR}/tmp/share")

############################################################################################################################
# Find all required libraries to build
############################################################################################################################
SET(OLD_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
SET(PROJECT_SOURCE_DIR $ENV{TT_METAL_HOME})
include($ENV{TT_METAL_HOME}/cmake/dependencies.cmake)
SET(PROJECT_SOURCE_DIR ${OLD_PROJECT_SOURCE_DIR})

# bro is there a more clean way to do this??? wtf

############################################################################################################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propogate to the rest of tt_metal, tt_eager, etc.
############################################################################################################################
add_library(stdlib INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(stdlib INTERFACE c++ c++abi)
    target_compile_options(stdlib INTERFACE -stdlib=libc++)
else()
    target_link_libraries(stdlib INTERFACE stdc++)
endif()

add_library(metal_common_libs INTERFACE)
target_link_libraries(metal_common_libs INTERFACE
    dl z pthread atomic hwloc numa stdlib # system libraries, hwloc has no cmake support, find_package won't find it
)

# Note on flags:
#   DFMT_HEADER_ONLY must be for every target or else they won't interact with the header only fmt as intended
#   ttnn and tt_lib will break if built with LTO, so leaving -fno-lto in compile options
add_library(linker_flags INTERFACE)

add_library(compiler_warnings INTERFACE)
target_compile_options(compiler_warnings INTERFACE -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter)

# add additional compile warning flags depending on the compiler
ADJUST_COMPILER_WARNINGS()

add_library(compiler_flags INTERFACE)
target_link_libraries(compiler_flags INTERFACE compiler_warnings stdlib)
target_compile_options(compiler_flags INTERFACE -mavx2 -fPIC -DFMT_HEADER_ONLY -fvisibility-inlines-hidden -fno-lto)

if(TT_METAL_VERSIM_DISABLED)
    target_compile_options(compiler_flags INTERFACE -DTT_METAL_VERSIM_DISABLED)
endif()
if(ENABLE_CODE_TIMERS)
    target_compile_options(compiler_flags INTERFACE -DTT_ENABLE_CODE_TIMERS)
endif()
if(ENABLE_TRACY)
    target_compile_options(compiler_flags INTERFACE -DTRACY_ENABLE -fno-omit-frame-pointer)
    target_link_options(linker_flags INTERFACE -rdynamic)
endif()

add_library(metal_header_directories INTERFACE)
target_include_directories(metal_header_directories INTERFACE $ENV{TT_METAL_HOME}/tt_metal/hw/inc)
target_include_directories(metal_header_directories SYSTEM INTERFACE ${reflect_SOURCE_DIR})
foreach(lib ${BoostPackages})
    target_include_directories(metal_header_directories INTERFACE ${Boost${lib}_SOURCE_DIR}/include)
endforeach()

if ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
    target_include_directories(metal_header_directories INTERFACE tt_metal/hw/inc/wormhole
        tt_metal/hw/inc/wormhole/wormhole_b0_defines
        ${UMD_HOME}/device/wormhole
        ${UMD_HOME}/src/firmware/riscv/wormhole
    )
else()
    target_compile_options(compiler_flags INTERFACE -DDISABLE_ISSUE_3487_FIX)
    target_include_directories(metal_header_directories INTERFACE
        tt_metal/hw/inc/$ENV{ARCH_NAME}
        ${UMD_HOME}/device/$ENV{ARCH_NAME}
        ${UMD_HOME}/src/firmware/riscv/$ENV{ARCH_NAME}
    )
endif()

############################################################################################################################
# Build subdirectories
############################################################################################################################
if(ENABLE_TRACY)
    include($ENV{TT_METAL_HOME}/cmake/tracy.cmake)
endif()

############################################################################################################################
# Our actual project
############################################################################################################################
add_executable(tt-llk-example 
                main.cpp
)

target_include_directories(tt-llk-example PRIVATE
                            metal_header_directories
                            ${reflect_SOURCE_DIR}
                            $ENV{TT_METAL_HOME}
                            $ENV{TT_METAL_HOME}/build/lib
                            $ENV{TT_METAL_HOME}/tt_metal
                            $ENV{TT_METAL_HOME}/tt_metal/common
                            $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
                            $ENV{TT_METAL_HOME}/tt_metal/hw/inc/$ENV{ARCH_NAME}
                            $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
                            $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/$ENV{ARCH_NAME}
                            $ENV{TT_METAL_HOME}/tt_metal/detail
                            $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon
                            $ENV{TT_METAL_HOME}/tt_metal/include
                            $ENV{TT_METAL_HOME}/tt_metal/third_party/
                            $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
                            $ENV{TT_METAL_HOME}/tt_metal/third_party/pybind11/include
)

target_link_directories(tt-llk-example PRIVATE
    $ENV{TT_METAL_HOME}/build/lib
)

target_link_libraries(tt-llk-example PRIVATE
                        stdlib
                        metal_common_libs
                        linker_flags
                        compiler_flags
                        tt_metal 
                        yaml-cpp
)
