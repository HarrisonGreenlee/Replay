cmake_minimum_required(VERSION 3.18)
project(GPUCPUSim LANGUAGES C CXX CUDA)

# Build settings
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find dependencies
find_package(OpenMP REQUIRED)
find_package(LibXml2 REQUIRED)

# Include paths
include_directories(
    ${LibXml2_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncls/ncls/src     # intervaldb.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Eigen             # Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}                   # headers like temporal_contact_matrix.h
)

# Source files
set(SOURCES
    ncls/ncls/src/intervaldb.c
    temporal_contact_matrix.c
    gpu_cpu_temporal_sim.cu
)

# Build executable
add_executable(replay ${SOURCES})

# Add macro to disable Python.h in intervaldb
target_compile_definitions(replay PRIVATE BUILD_C_LIBRARY)

# Link dependencies
target_link_libraries(replay
    PRIVATE
        ${LibXml2_LIBRARIES}
        OpenMP::OpenMP_CXX
        cudart
        cusparse
        cublas
        curand
)

# MSVC-specific settings
if (MSVC)
    target_link_options(replay PRIVATE /DEBUG /NODEFAULTLIB:libcmtd.lib)
endif()
