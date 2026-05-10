/*
 * Advanced Parallel Grid Processing with MPI
 * ============================================
 * Fayoum University – Parallel Computing Project 2026
 *
 * Usage:
 *   mpirun -np <P> ./parallel_grid [algorithm] [options]
 *
 *   algorithm:
 *     heat          – Heat Diffusion (stencil)
 *     matmul        – Matrix Multiplication
 *     both          – Run both sequentially (default)
 *     deadlock_demo – Show deadlock scenario and fix
 *     ring          – Ring communication demo
 *     pipeline      – Pipeline communication demo
 *
 *   options (for heat):
 *     --grid <N>       grid size (default 512)
 *     --iter <I>       iterations (default 200)
 *     --nonblocking    use non-blocking halo exchange
 *
 *   options (for matmul):
 *     --fileA <path>   matrix A file
 *     --fileB <path>   matrix B file
 *     --size  <N>      random NxN matrix size (default 256)
 *     --verify         verify result against sequential (only for small N)
 *
 *   options (general):
 *     --generate-data  generate test matrix files in ./data/
 */

#include <mpi.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>

#include "algorithms/heat_diffusion.h"
#include "algorithms/matrix_multiplication.h"
#include "communication/blocking.h"
#include "communication/nonblocking.h"
#include "utils/file_loader.h"
#include "utils/timer.h"

// ─── argument helpers ────────────────────────────────────────────────────────

static bool hasArg(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; i++)
        if (key == argv[i]) return true;
    return false;
}

static std::string getArg(int argc, char** argv, const std::string& key,
                          const std::string& def = "") {
    for (int i = 1; i < argc - 1; i++)
        if (key == argv[i]) return argv[i + 1];
    return def;
}

// ─── deadlock demonstration ──────────────────────────────────────────────────
/*
 * DEADLOCK SCENARIO
 * -----------------
 * Consider 2 ranks, both executing:
 *
 *     MPI_Send(buf, N, MPI_DOUBLE, other, tag, comm);  // BLOCKS until received
 *     MPI_Recv(buf, N, MPI_DOUBLE, other, tag, comm, &status);
 *
 * If MPI_Send uses synchronous mode (or the message is too large to buffer),
 * BOTH ranks block on Send waiting for the other to call Recv — which never
 * happens because both are stuck in Send.  → DEADLOCK.
 *
 * FIX: use MPI_Sendrecv (or Isend/Irecv + Waitall, or alternating ordering).
 */
static void deadlockDemo(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2) {
        if (rank == 0) std::cout << "[DeadlockDemo] Need at least 2 processes.\n";
        return;
    }

    if (rank == 0) {
        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout << "║        DEADLOCK DEMONSTRATION                ║\n";
        std::cout << "╠══════════════════════════════════════════════╣\n";
        std::cout << "║ Scenario: Both ranks call MPI_Send first.    ║\n";
        std::cout << "║ If messages exceed buffer → DEADLOCK.        ║\n";
        std::cout << "║                                              ║\n";
        std::cout << "║ We demonstrate the FIXED version using       ║\n";
        std::cout << "║ MPI_Sendrecv which avoids the race condition. ║\n";
        std::cout << "╚══════════════════════════════════════════════╝\n\n";
    }
    MPI_Barrier(comm);

    // Only ranks 0 and 1 participate
    if (rank > 1) return;
    int other = 1 - rank;

    const int N = 1024;
    std::vector<double> send(N, (double)rank);
    std::vector<double> recv(N, 0.0);

    // ── FIXED version: MPI_Sendrecv ──────────────────────────────────────────
    MPI_Status status;
    MPI_Sendrecv(send.data(), N, MPI_DOUBLE, other, 99,
                 recv.data(), N, MPI_DOUBLE, other, 99,
                 comm, &status);

    std::cout << "[Rank " << rank << "] Received " << N
              << " doubles from rank " << other
              << " (first value = " << recv[0] << ")\n";

    MPI_Barrier(comm);
    if (rank == 0)
        std::cout << "[DeadlockDemo] Fixed version completed successfully.\n\n";
}

// ─── process group demo ──────────────────────────────────────────────────────
static void processGroupDemo(MPI_Comm worldComm) {
    int rank, size;
    MPI_Comm_rank(worldComm, &rank);
    MPI_Comm_size(worldComm, &size);

    // Split into two groups: even ranks and odd ranks
    int color = rank % 2;
    MPI_Comm subComm;
    MPI_Comm_split(worldComm, color, rank, &subComm);

    int subRank, subSize;
    MPI_Comm_rank(subComm, &subRank);
    MPI_Comm_size(subComm, &subSize);

    // Each sub-group does its own reduction
    double myVal = (double)(rank + 1);
    double subSum = 0.0;
    MPI_Reduce(&myVal, &subSum, 1, MPI_DOUBLE, MPI_SUM, 0, subComm);

    if (subRank == 0) {
        std::cout << "[ProcessGroup] " << (color == 0 ? "Even" : "Odd ")
                  << " group (" << subSize << " ranks): sum = " << subSum << "\n";
    }

    MPI_Comm_free(&subComm);
}

// ─── ring demo ───────────────────────────────────────────────────────────────
static void ringDemo(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    double myToken = (double)(rank + 1) * 10.0;
    double total   = ringCommunication(myToken, comm);

    if (rank == 0)
        std::cout << "[RingDemo] Sum of all tokens = " << total << "\n";
}

// ─── pipeline demo ───────────────────────────────────────────────────────────
static void pipelineDemo(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double result = pipelineCommunication(100.0, comm);
    if (rank == 0)
        std::cout << "[PipelineDemo] Final value after pipeline = " << result
                  << "  (expected " << 100.0 + size*(size-1)/2.0 << ")\n";
}

// ─── interactive menu (rank 0 only, then broadcast choices) ─────────────────

struct UserChoices {
    int  algo;          // 1=heat 2=matmul 3=both 4=deadlock 5=ring 6=pipeline
    int  gridSize;
    int  iterations;
    int  useNonBlocking;
    int  matSize;
    int  verify;
};

static UserChoices askUser() {
    UserChoices c{};

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║   Advanced Parallel Grid Processing with MPI         ║\n";
    std::cout << "║   Fayoum University – Parallel Computing 2026        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Select an option:                                   ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║  [1] Heat Diffusion          (Category A – Grid)     ║\n";
    std::cout << "║  [2] Matrix Multiplication   (Category B – Data)     ║\n";
    std::cout << "║  [3] Both Algorithms                                 ║\n";
    std::cout << "║  [4] Deadlock Demo                                   ║\n";
    std::cout << "║  [5] Ring Communication Demo                         ║\n";
    std::cout << "║  [6] Pipeline Communication Demo                     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << "  Your choice: ";
    std::cin >> c.algo;

    // ── Heat options ─────────────────────────────────────────────────────────
    if (c.algo == 1 || c.algo == 3) {
        std::cout << "\n--- Heat Diffusion Settings ---\n";

        std::cout << "  Grid size N (NxN grid, e.g. 256 or 512): ";
        std::cin >> c.gridSize;

        std::cout << "  Number of iterations (e.g. 100 or 200): ";
        std::cin >> c.iterations;

        int nb = 0;
        std::cout << "  Use non-blocking communication? (0=No  1=Yes): ";
        std::cin >> nb;
        c.useNonBlocking = nb;
    } else {
        c.gridSize      = 256;
        c.iterations    = 100;
        c.useNonBlocking = 0;
    }

    // ── MatMul options ───────────────────────────────────────────────────────
    if (c.algo == 2 || c.algo == 3) {
        std::cout << "\n--- Matrix Multiplication Settings ---\n";

        std::cout << "  Matrix size N (NxN random matrix, e.g. 128 or 256): ";
        std::cin >> c.matSize;

        int v = 0;
        std::cout << "  Verify result against sequential? (0=No  1=Yes, slow for N>256): ";
        std::cin >> v;
        c.verify = v;
    } else {
        c.matSize = 256;
        c.verify  = 0;
    }

    return c;
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ── Only rank 0 shows the menu and reads input ───────────────────────────
    UserChoices choices{};
    if (rank == 0) {
        std::cout << "\n  Running with " << size << " MPI process(es)\n";
        choices = askUser();
    }

    // ── Broadcast all choices to every rank so they agree ────────────────────
    MPI_Bcast(&choices, sizeof(UserChoices), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    Timer timer;

    // ── Process groups (always shown) ────────────────────────────────────────
    if (rank == 0) std::cout << "\n--- Process Group Demo (MPI_Comm_split) ---\n";
    processGroupDemo(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Algorithm dispatch ───────────────────────────────────────────────────

    if (choices.algo == 4) {
        deadlockDemo(MPI_COMM_WORLD);

    } else if (choices.algo == 5) {
        if (rank == 0) std::cout << "\n--- Ring Communication Demo ---\n";
        ringDemo(MPI_COMM_WORLD);

    } else if (choices.algo == 6) {
        if (rank == 0) std::cout << "\n--- Pipeline Communication Demo ---\n";
        pipelineDemo(MPI_COMM_WORLD);

    } else {

        // ── Heat Diffusion ───────────────────────────────────────────────────
        if (choices.algo == 1 || choices.algo == 3) {
            HeatConfig hcfg;
            hcfg.gridSize       = choices.gridSize;
            hcfg.iterations     = choices.iterations;
            hcfg.useNonBlocking = (choices.useNonBlocking == 1);

            if (rank == 0) std::cout << "\n--- Running Heat Diffusion ---\n";
            MPI_Barrier(MPI_COMM_WORLD);
            timer.start("HeatDiffusion");
            runHeatDiffusion(hcfg, MPI_COMM_WORLD);
            double heatMs = timer.stop("HeatDiffusion");
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0)
                std::cout << "[Main] Heat Diffusion total wall time: " << heatMs << " ms\n";
        }

        // ── Matrix Multiplication ────────────────────────────────────────────
        if (choices.algo == 2 || choices.algo == 3) {
            MatMulConfig mcfg;
            mcfg.randomN = choices.matSize;
            mcfg.verify  = (choices.verify == 1);

            if (rank == 0) std::cout << "\n--- Running Matrix Multiplication ---\n";
            MPI_Barrier(MPI_COMM_WORLD);
            timer.start("MatMul");
            runMatrixMultiplication(mcfg, MPI_COMM_WORLD);
            double matMs = timer.stop("MatMul");
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0)
                std::cout << "[Main] Matrix Multiplication total wall time: " << matMs << " ms\n";
        }

        // ── Always show comm demos after main algorithms ─────────────────────
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n--- Communication Pattern Demos ---\n";
        ringDemo(MPI_COMM_WORLD);
        pipelineDemo(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n";
        deadlockDemo(MPI_COMM_WORLD);
    }

    // ── Timing summary ───────────────────────────────────────────────────────
    if (rank == 0) timer.report();

    MPI_Finalize();
    return 0;
}