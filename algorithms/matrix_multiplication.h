#pragma once
#include <mpi.h>
#include "../utils/matrix.h"
#include <string>

/*
 * Parallel Matrix Multiplication  (C = A × B)
 * ---------------------------------------------
 * Strategy: Row-wise distribution of A across ranks.
 *           B is broadcast to all ranks (since every rank needs all of B).
 *
 * Supports loading A and B from files (big-data requirement).
 */

struct MatMulConfig {
    std::string fileA;    // path to matrix A file (empty → generate random)
    std::string fileB;    // path to matrix B file (empty → generate random)
    int         randomN = 256;  // size for random generation
    bool        verify  = false; // verify result with sequential multiply (slow)
};

// Root rank loads/generates A and B, distributes, multiplies in parallel.
// Returns C on rank 0, empty matrix on other ranks.
Matrix runMatrixMultiplication(const MatMulConfig& cfg, MPI_Comm comm);
