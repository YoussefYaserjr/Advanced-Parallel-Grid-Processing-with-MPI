# Advanced Parallel Grid Processing with MPI



A C++ MPI-based parallel computing project demonstrating core parallel algorithms and communication patterns.

## Features

### Algorithms
- **Heat Diffusion** -- Parallel 2D Jacobi stencil with row-wise decomposition across MPI ranks. Supports both blocking and non-blocking halo exchange.
- **Matrix Multiplication** -- Parallel row-partitioned matrix multiplication using `MPI_Scatterv`/`MPI_Gatherv` with matrix broadcast.

### Communication Demos
- **Blocking Exchange** -- Parity-ordered `MPI_Send`/`MPI_Recv` with deadlock prevention
- **Non-blocking Exchange** -- `MPI_Isend`/`MPI_Irecv` with `MPI_Waitall`
- **Neighbor Exchange** -- 1D and 2D exchange via `MPI_Sendrecv`
- **Ring Communication** -- Token passing around a ring of processes
- **Pipeline Communication** -- Value flowing through a chain of processes
- **Deadlock Demo** -- Demonstrates deadlock scenario and its fix
- **Process Groups** -- Splitting communicators via `MPI_Comm_split`

## Prerequisites

- **MPI Implementation** -- [MS-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (Windows), [MPICH](https://www.mpich.org/), or [OpenMPI](https://www.open-mpi.org/)
- **CMake** >= 3.16
- **C++17** compatible compiler (MSVC, GCC, Clang)
- **Ninja** (optional, recommended)

## Building

### Option 1: Command Line (CMake + Ninja)

```bash
cmake -B build -G Ninja
cmake --build build
```

### Option 2: Command Line (CMake + default generator)

```bash
cmake -B build
cmake --build build --config Release
```



## Running

### Basic Usage

```bash
mpirun -np <num_processes> ./parallel_grid [algorithm] [options]
```

> **Note:** On Windows with MS-MPI, use `mpiexec` instead of `mpirun`.

### Interactive Mode

Run without arguments to get an interactive menu:

```bash
mpirun -np 4 ./parallel_grid
```

### Command-Line Options

| Algorithm | Command | Options |
|---|---|---|
| Heat Diffusion | `heat` | `--grid <N>` grid size (default 512), `--iter <I>` iterations (default 200), `--nonblocking` use non-blocking exchange |
| Matrix Multiplication | `matmul` | `--size <N>` random NxN size (default 256), `--verify` verify against sequential |
| Both | `both` (default) | Combines heat + matmul settings |
| Deadlock Demo | `deadlock_demo` | No additional options |
| Ring Demo | `ring` | No additional options |
| Pipeline Demo | `pipeline` | No additional options |

### Examples

```bash
# Heat diffusion on 256x256 grid, 100 iterations, 8 processes
mpirun -np 8 ./parallel_grid heat --grid 256 --iter 100

# Matrix multiplication 512x512 with verification
mpirun -np 4 ./parallel_grid matmul --size 512 --verify

# Non-blocking halo exchange
mpirun -np 4 ./parallel_grid heat --grid 512 --iter 200 --nonblocking

# Communication demo
mpirun -np 6 ./parallel_grid ring
```

```

## Output

- **Heat Diffusion** produces `heat_output.csv` containing the final temperature grid, suitable for visualization in Excel or Python.
- **Matrix Multiplication** prints timing results and optionally verifies correctness against a sequential implementation.
- All algorithms report wall-clock timing via the built-in `Timer` class.

## MPI Concepts Demonstrated

| Concept | MPI Functions Used |
|---|---|
| Data Decomposition | Row-wise partitioning of grids and matrices |
| Broadcast | `MPI_Bcast` |
| Scatter/Gather | `MPI_Scatterv`, `MPI_Gatherv` |
| Reduction | `MPI_Reduce` |
| Blocking Communication | `MPI_Send`, `MPI_Recv` |
| Non-blocking Communication | `MPI_Isend`, `MPI_Irecv`, `MPI_Waitall` |
| Send-Receive | `MPI_Sendrecv` |
| Process Groups | `MPI_Comm_split` |
| Synchronization | `MPI_Barrier` |
