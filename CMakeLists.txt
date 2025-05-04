cmake_minimum_required(VERSION 3.10)\nproject(GraphX)\nfind_package(CUDA REQUIRED)\nset(CMAKE_CXX_STANDARD 14)\ninclude_directories(include)\ncuda_add_executable(graphx src/main.cu)
