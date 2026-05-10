#include "nonblocking.h"
#include <iostream>

void exchangeGhostRowsNonBlocking(
    const std::vector<double>& sendTop,
    const std::vector<double>& sendBot,
    std::vector<double>&       recvTop,
    std::vector<double>&       recvBot,
    int above, int below, MPI_Comm comm)
{
    int N = (int)sendTop.size();
    MPI_Request reqs[4];
    MPI_Status  stats[4];
    int nReqs = 0;

    // Post all receives first (best practice for non-blocking)
    if (above != MPI_PROC_NULL)
        MPI_Irecv(recvTop.data(), N, MPI_DOUBLE, above, 1, comm, &reqs[nReqs++]);
    if (below != MPI_PROC_NULL)
        MPI_Irecv(recvBot.data(), N, MPI_DOUBLE, below, 0, comm, &reqs[nReqs++]);

    // Then post sends
    if (above != MPI_PROC_NULL)
        MPI_Isend(sendTop.data(), N, MPI_DOUBLE, above, 0, comm, &reqs[nReqs++]);
    if (below != MPI_PROC_NULL)
        MPI_Isend(sendBot.data(), N, MPI_DOUBLE, below, 1, comm, &reqs[nReqs++]);

    // Wait for all transfers to complete
    MPI_Waitall(nReqs, reqs, stats);
}

double pipelineCommunication(double startValue, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double value = startValue;
    MPI_Status status;

    if (size == 1) return value + rank;

    if (rank == 0) {
        // Stage 0: send to next
        value += rank;
        MPI_Request req;
        MPI_Isend(&value, 1, MPI_DOUBLE, 1, 20, comm, &req);
        MPI_Wait(&req, &status);
    } else {
        // Receive from previous stage
        MPI_Recv(&value, 1, MPI_DOUBLE, rank - 1, 20, comm, &status);
        value += rank;
        if (rank < size - 1) {
            MPI_Request req;
            MPI_Isend(&value, 1, MPI_DOUBLE, rank + 1, 20, comm, &req);
            MPI_Wait(&req, &status);
        }
    }

    // Broadcast final result from last rank to all
    MPI_Bcast(&value, 1, MPI_DOUBLE, size - 1, comm);
    return value;
}
