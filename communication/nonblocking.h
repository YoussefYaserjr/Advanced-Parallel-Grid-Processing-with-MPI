#pragma once
#include <mpi.h>
#include <vector>

/*
 * Non-blocking ghost-row exchange using MPI_Isend / MPI_Irecv.
 *
 * All four transfers are posted simultaneously, then MPI_Waitall
 * ensures completion before the caller uses the received data.
 *
 * Advantage over blocking: computation can overlap communication
 * (interior cells can be updated while halos are in flight).
 */
void exchangeGhostRowsNonBlocking(
    const std::vector<double>& sendTop,
    const std::vector<double>& sendBot,
    std::vector<double>&       recvTop,
    std::vector<double>&       recvBot,
    int above, int below, MPI_Comm comm);

/*
 * Pipeline communication demo.
 * Rank 0 starts with a value; it flows through 0→1→2→…→(size-1).
 * Each rank adds its rank number and forwards.
 * Returns the final value on the last rank (broadcast to all via MPI_Bcast).
 */
double pipelineCommunication(double startValue, MPI_Comm comm);
