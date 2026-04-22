cmake_minimum_required(VERSION 3.10)

# Make sure this matches ./NAM/version.h!
project(NAM VERSION 0.4.0)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use libc++ on macOS, system default on Linux
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	include_directories(SYSTEM /usr/local/include)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	link_libraries(stdc++fs)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
	message(FATAL_ERROR "Unrecognized Platform!")
endif()

# STRATUS: Eigen is located at shared/eigen (sibling to NeuralAmpModelerCore)
set(NAM_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories(SYSTEM "${NAM_DEPS_PATH}/eigen")

# Build the specialized A2 fast-path WaveNet (A2 standard + A2 nano). When ON,
# models whose config matches the A2 shape signature are routed to a hand-optimized
# WaveNet implementation instead of the generic one. When OFF, all models go
# through the generic path.
# ============================================================
# STRATUS: Build NAM as static library
# ============================================================
# NAM's original CMakeLists only builds executables in tools/ subdirectory.
# We need the library for firmware integration, so we build it explicitly.

file(GLOB_RECURSE NAM_SOURCES NAM/*.cpp NAM/*.c)
file(GLOB_RECURSE NAM_HEADERS NAM/*.h)

add_library(NeuralAmpModelerCore STATIC ${NAM_SOURCES} ${NAM_HEADERS})

target_include_directories(NeuralAmpModelerCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/NAM
    ${NAM_DEPS_PATH}/eigen
)

target_compile_features(NeuralAmpModelerCore PUBLIC cxx_std_17)

# Match optimization flags from Stratus firmware
if (NOT MSVC)
    target_compile_options(NeuralAmpModelerCore PRIVATE
        -O3 -fPIC -ftree-vectorize -ffast-math -funroll-loops
        -Wall -Wextra
    )
endif()

# Embedded/single-core optimizations
target_compile_definitions(NeuralAmpModelerCore PRIVATE
    EIGEN_DONT_PARALLELIZE  # Disable Eigen threading (Stratus is single-core audio)
    NDEBUG                  # Disable assertions in release builds
    NAM_USE_INLINE_GEMM     # Hand-optimized GEMM kernels (~3x faster than Eigen on ARM)
    NAM_ENABLE_A2_FAST      # A2 fast-path WaveNet for nano (ch=3) and standard (ch=8) models
)

# STRATUS: Disable building NAM's internal tools and tests
# They conflict with NAM_SAMPLE_FLOAT and we have our own benchmarking system
# add_subdirectory(tools)

#file(MAKE_DIRECTORY build/tools)

#add_custom_target(copy_tools ALL
#	${CMAKE_COMMAND} -E copy "$<TARGET_FILE:tools>" tools/
#	DEPENDS tools
#)
