#include "matrix_multiplication.h"
#include "../utils/file_loader.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>

// ─── sequential helper (for verification only) ──────────────────────────────
static Matrix seqMul(const Matrix& A, const Matrix& B) {
    int M = A.size(), K = B.size(), N = B[0].size();
    Matrix C = createMatrix(M, N, 0.0);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// ─── main entry ─────────────────────────────────────────────────────────────
Matrix runMatrixMultiplication(const MatMulConfig& cfg, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Dimensions broadcast from rank 0
    int dims[3] = {0, 0, 0}; // M, K, N
    std::vector<double> flatA, flatB;

    if (rank == 0) {
        Matrix A, B;
        if (!cfg.fileA.empty() && !cfg.fileB.empty()) {
            A = loadMatrix(cfg.fileA);
            B = loadMatrix(cfg.fileB);
        } else {
            srand(42);
            int N = cfg.randomN;
            A = randomMatrix(N, N);
            B = randomMatrix(N, N);
            std::cout << "[MatMul] Generated random " << N << "x" << N << " matrices\n";
        }
        dims[0] = A.size();
        dims[1] = A[0].size();  // K
        dims[2] = B[0].size();  // N
        if ((int)B.size() != dims[1])
            throw std::runtime_error("Matrix dimension mismatch: A cols != B rows");

        flatA = matrixToFlat(A);
        flatB = matrixToFlat(B);
    }

    MPI_Bcast(dims, 3, MPI_INT, 0, comm);
    int M = dims[0], K = dims[1], N = dims[2];

    if (rank == 0) {
        std::cout << "[MatMul] A=" << M << "x" << K
                  << "  B=" << K << "x" << N
                  << "  procs=" << size << "\n";
    }

    // Broadcast B to all ranks
    flatB.resize(K * N);
    MPI_Bcast(flatB.data(), K * N, MPI_DOUBLE, 0, comm);
    Matrix B = flatToMatrix(flatB, K, N);

    // Distribute rows of A (handle uneven)
    int baseRows = M / size;
    int extra    = M % size;
    int myRows   = baseRows + (rank < extra ? 1 : 0);
    int myStart  = rank * baseRows + std::min(rank, extra);

    // Scatterv A rows
    std::vector<int> sendCounts(size), displs(size);
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < size; r++) {
            int rRows = baseRows + (r < extra ? 1 : 0);
            sendCounts[r] = rRows * K;
            displs[r]     = off;
            off += sendCounts[r];
        }
    }

    std::vector<double> myFlatA(myRows * K);
    MPI_Scatterv(flatA.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 myFlatA.data(), myRows * K, MPI_DOUBLE, 0, comm);

    Matrix myA = flatToMatrix(myFlatA, myRows, K);

    // Local multiply: myC = myA × B
    Matrix myC = createMatrix(myRows, N, 0.0);
    for (int i = 0; i < myRows; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                myC[i][j] += myA[i][k] * B[k][j];

    // Gather result on rank 0
    std::vector<double> myFlatC = matrixToFlat(myC);

    std::vector<int> recvCounts(size), recvDispls(size);
    if (rank == 0) {
        int off = 0;
        for (int r = 0; r < size; r++) {
            int rRows = baseRows + (r < extra ? 1 : 0);
            recvCounts[r] = rRows * N;
            recvDispls[r] = off;
            off += recvCounts[r];
        }
    }

    std::vector<double> allFlatC;
    if (rank == 0) allFlatC.resize(M * N);

    MPI_Gatherv(myFlatC.data(), myRows * N, MPI_DOUBLE,
                allFlatC.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, comm);

    Matrix result;
    if (rank == 0) {
        result = flatToMatrix(allFlatC, M, N);
        std::cout << "[MatMul] Done. C[0][0]=" << result[0][0]
                  << "  C[M/2][N/2]=" << result[M/2][N/2] << "\n";

        if (cfg.verify && M <= 256) {
            // Reconstruct A for verification
            Matrix A = flatToMatrix(flatA, M, K);
            Matrix ref = seqMul(A, B);
            bool ok = matricesEqual(result, ref, 1e-4);
            std::cout << "[MatMul] Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
        }
    }
    return result;
}
