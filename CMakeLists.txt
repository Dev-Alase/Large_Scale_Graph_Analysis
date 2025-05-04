cmake_minimum_required(VERSION 3.18)
project(GraphX CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Add include directories
include_directories(include)

# Add the executable
add_executable(graphx src/main.cu)

# Set CUDA architecture
set_target_properties(graphx PROPERTIES
    CUDA_ARCHITECTURES "75"
)

# Link against CUDA libraries
target_link_libraries(graphx PRIVATE CUDA::cudart)
