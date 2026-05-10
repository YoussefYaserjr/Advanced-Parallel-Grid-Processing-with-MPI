#pragma once
#include <mpi.h>
#include "../utils/matrix.h"

/*
 * Parallel Heat Diffusion (Jacobi Stencil)
 * -----------------------------------------
 * Domain: NxN grid.  Each MPI rank owns a horizontal slab (row-wise partition).
 * Ghost rows are exchanged with neighbours every iteration via blocking or
 * non-blocking sends (selected at runtime).
 *
 * Update rule:  T[i][j] = 0.25 * (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1])
 */

struct HeatConfig {
    int    gridSize   = 512;   // NxN
    int    iterations = 200;
    double alpha      = 0.25;  // diffusion coefficient weight
    bool   useNonBlocking = false;
};

// Run heat diffusion on the current MPI rank.
// Root (rank 0) returns the full assembled grid; others return empty.
Matrix runHeatDiffusion(const HeatConfig& cfg, MPI_Comm comm);
