cmake_minimum_required(VERSION 3.16)
project(parallel_grid)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(parallel_grid
        main.cpp
        algorithms/heat_diffusion.cpp
        algorithms/matrix_multiplication.cpp
        communication/blocking.cpp
        communication/nonblocking.cpp
        communication/neighbor_exchange.cpp
        utils/matrix.cpp
        utils/timer.cpp
        utils/file_loader.cpp
)

target_include_directories(parallel_grid PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(parallel_grid ${MPI_CXX_LIBRARIES})