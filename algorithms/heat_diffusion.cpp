#include "heat_diffusion.h"
#include "../communication/blocking.h"
#include "../communication/nonblocking.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

// ─── helpers ────────────────────────────────────────────────────────────────

static void applyBoundary(Matrix& g, int N) {
    // Fixed hot boundary on top, cold on bottom; insulated sides
    for (int j = 0; j < N; j++) { g[0][j] = 100.0; g[N-1][j] = 0.0; }
    for (int i = 0; i < N; i++) { g[i][0] = g[i][1]; g[i][N-1] = g[i][N-2]; }
}

// ─── main entry ─────────────────────────────────────────────────────────────

Matrix runHeatDiffusion(const HeatConfig& cfg, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int N = cfg.gridSize;

    // Row decomposition – handle uneven sizes
    int baseRows = N / size;
    int extra    = N % size;
    int myRows   = baseRows + (rank < extra ? 1 : 0);
    int myStart  = rank * baseRows + std::min(rank, extra);

    // Local grid with 2 ghost rows (top and bottom)
    int localH = myRows + 2;  // +2 ghost rows
    Matrix local(localH, std::vector<double>(N, 0.0));
    Matrix next (localH, std::vector<double>(N, 0.0));

    // Initialise interior: simple gradient
    for (int i = 1; i <= myRows; i++) {
        int globalRow = myStart + i - 1;
        for (int j = 0; j < N; j++)
            local[i][j] = 100.0 * (1.0 - (double)globalRow / (N - 1));
    }

    int above = (rank > 0)      ? rank - 1 : MPI_PROC_NULL;
    int below = (rank < size-1) ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        std::cout << "[HeatDiffusion] Grid=" << N << "x" << N
                  << "  procs=" << size
                  << "  iters=" << cfg.iterations
                  << "  mode=" << (cfg.useNonBlocking ? "non-blocking" : "blocking")
                  << "\n";
    }

    for (int iter = 0; iter < cfg.iterations; iter++) {

        // ── ghost-row exchange ──────────────────────────────────────────────
        std::vector<double> sendTop(N), sendBot(N), recvTop(N), recvBot(N);

        for (int j = 0; j < N; j++) {
            sendTop[j] = local[1][j];          // first real row → send up
            sendBot[j] = local[myRows][j];     // last  real row → send down
        }

        if (cfg.useNonBlocking) {
            exchangeGhostRowsNonBlocking(sendTop, sendBot, recvTop, recvBot,
                                         above, below, comm);
        } else {
            exchangeGhostRowsBlocking(sendTop, sendBot, recvTop, recvBot,
                                      above, below, comm);
        }

        for (int j = 0; j < N; j++) {
            local[0][j]         = recvTop[j];   // ghost top
            local[myRows+1][j]  = recvBot[j];   // ghost bot
        }

        // Apply fixed boundaries at global edges
        if (rank == 0)
            for (int j = 0; j < N; j++) local[0][j] = 100.0;   // hot top ghost
        if (rank == size - 1)
            for (int j = 0; j < N; j++) local[myRows+1][j] = 0.0; // cold bot ghost

        // ── stencil update ──────────────────────────────────────────────────
        for (int i = 1; i <= myRows; i++) {
            for (int j = 1; j < N - 1; j++) {
                next[i][j] = cfg.alpha *
                    (local[i-1][j] + local[i+1][j] + local[i][j-1] + local[i][j+1]);
            }
            // insulated sides
            next[i][0]   = next[i][1];
            next[i][N-1] = next[i][N-2];
        }
        std::swap(local, next);
    }

    // ── gather results on rank 0 ────────────────────────────────────────────
    // Flatten local real rows (exclude ghost rows)
    std::vector<double> myData(myRows * N);
    for (int i = 0; i < myRows; i++)
        for (int j = 0; j < N; j++)
            myData[i * N + j] = local[i + 1][j];

    // Gather row counts and offsets
    std::vector<int> recvCounts(size), displs(size);
    MPI_Gather(&myRows, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, comm);

    Matrix result;
    std::vector<double> allData;

    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < size; r++) {
            displs[r] = total;
            recvCounts[r] *= N;
            total += recvCounts[r];
        }
        allData.resize(total);
    }

    MPI_Gatherv(myData.data(), myRows * N, MPI_DOUBLE,
                allData.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        result = flatToMatrix(allData, N, N);

        // ── Correctness checks ───────────────────────────────────────────────
        std::cout << "\n========================================\n";
        std::cout << "  HEAT DIFFUSION – RESULTS\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(4);

        // 1. Boundary values
        double topAvg = 0.0, botAvg = 0.0;
        for (int j = 0; j < N; j++) { topAvg += result[0][j]; botAvg += result[N-1][j]; }
        topAvg /= N;  botAvg /= N;

        std::cout << "  Top edge avg    : " << topAvg
                  << (std::fabs(topAvg - 100.0) < 1.0 ? "  ✓ (~100 C hot)" : "  ✗ WRONG") << "\n";
        std::cout << "  Bottom edge avg : " << botAvg
                  << (std::fabs(botAvg) < 1.0 ? "  ✓ (~0 C cold)" : "  ✗ WRONG") << "\n";
        std::cout << "  Center T["<<N/2<<"]["<<N/2<<"] : " << result[N/2][N/2]
                  << "  (expected ~40-60 C)\n";

        // 2. Gradient check: temperature should decrease from top to bottom
        bool monotone = true;
        for (int j = 0; j < N && monotone; j++)
            for (int i = 0; i < N-1 && monotone; i++)
                if (result[i][j] + 2.0 < result[i+1][j])  // allow small tolerance
                    monotone = false;
        std::cout << "  Top→Bot gradient: " << (monotone ? "✓ Decreasing (correct)" : "✗ Non-monotone") << "\n";

        // 3. No NaN / Inf
        bool hasNaN = false;
        for (int i = 0; i < N && !hasNaN; i++)
            for (int j = 0; j < N && !hasNaN; j++)
                if (!std::isfinite(result[i][j])) hasNaN = true;
        std::cout << "  NaN/Inf check   : " << (hasNaN ? "✗ Found invalid values!" : "✓ All values finite") << "\n";

        std::cout << "========================================\n\n";

        // ── Print a small ASCII preview ──────────────────────────────────────
        std::cout << "  Temperature grid preview (every " << std::max(1,N/8) << " rows/cols):\n";
        std::cout << "  Row\\Col";
        int step = std::max(1, N / 8);
        for (int j = 0; j < N; j += step) std::cout << std::setw(8) << j;
        std::cout << "\n";
        for (int i = 0; i < N; i += step) {
            std::cout << "  [" << std::setw(4) << i << "] ";
            for (int j = 0; j < N; j += step)
                std::cout << std::setw(8) << std::setprecision(2) << result[i][j];
            std::cout << "\n";
        }
        std::cout << "\n";

        // ── Save CSV (open with Excel to visualise as a heat map) ───────────
        std::ofstream csv("heat_output.csv");
        if (csv.is_open()) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    csv << std::fixed << std::setprecision(4) << result[i][j];
                    if (j < N-1) csv << ",";
                }
                csv << "\n";
            }
            std::cout << "  [Saved] heat_output.csv  ← open in Excel and apply\n"
                         "          a colour scale to see the heat map!\n\n";
        }
    }

    return result;
}