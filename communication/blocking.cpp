#include "blocking.h"
#include <iostream>

void exchangeGhostRowsBlocking(
    const std::vector<double>& sendTop,
    const std::vector<double>& sendBot,
    std::vector<double>&       recvTop,
    std::vector<double>&       recvBot,
    int above, int below, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int N = (int)sendTop.size();
    MPI_Status status;

    // Parity-based ordering prevents deadlock:
    // Even ranks send first; odd ranks receive first.
    if (rank % 2 == 0) {
        // Send top row up, receive ghost from above
        if (above != MPI_PROC_NULL) {
            MPI_Send(sendTop.data(), N, MPI_DOUBLE, above, 0, comm);
            MPI_Recv(recvTop.data(), N, MPI_DOUBLE, above, 1, comm, &status);
        }
        // Send bottom row down, receive ghost from below
        if (below != MPI_PROC_NULL) {
            MPI_Send(sendBot.data(), N, MPI_DOUBLE, below, 1, comm);
            MPI_Recv(recvBot.data(), N, MPI_DOUBLE, below, 0, comm, &status);
        }
    } else {
        // Odd: receive first, then send
        if (above != MPI_PROC_NULL) {
            MPI_Recv(recvTop.data(), N, MPI_DOUBLE, above, 1, comm, &status);
            MPI_Send(sendTop.data(), N, MPI_DOUBLE, above, 0, comm);
        }
        if (below != MPI_PROC_NULL) {
            MPI_Recv(recvBot.data(), N, MPI_DOUBLE, below, 0, comm, &status);
            MPI_Send(sendBot.data(), N, MPI_DOUBLE, below, 1, comm);
        }
    }
}

double ringCommunication(double myToken, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int right = (rank + 1) % size;
    int left  = (rank - 1 + size) % size;

    double accumulated = myToken;
    double recv_val;
    MPI_Status status;

    // Pass the token around the ring (size-1 hops)
    for (int hop = 0; hop < size - 1; hop++) {
        // Even/odd alternation to prevent deadlock in the ring
        if (rank % 2 == 0) {
            MPI_Send(&accumulated, 1, MPI_DOUBLE, right, 10, comm);
            MPI_Recv(&recv_val,    1, MPI_DOUBLE, left,  10, comm, &status);
        } else {
            MPI_Recv(&recv_val,    1, MPI_DOUBLE, left,  10, comm, &status);
            MPI_Send(&accumulated, 1, MPI_DOUBLE, right, 10, comm);
        }
        accumulated += recv_val;
    }
    return accumulated;
}
