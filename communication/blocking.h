#pragma once
#include <mpi.h>
#include <vector>

/*
 * Blocking ghost-row exchange using MPI_Send / MPI_Recv.
 *
 * To AVOID deadlock we use alternating send-first / recv-first based on rank
 * parity (odd/even ordering). This is a textbook deadlock-prevention pattern:
 *
 *   Even ranks: send up first, then receive from above; send down, then recv from below
 *   Odd  ranks: recv from above first, then send up; recv from below, then send down
 *
 * WITHOUT this ordering all ranks might MPI_Send simultaneously → deadlock
 * if the MPI implementation uses synchronous mode under the hood.
 */
void exchangeGhostRowsBlocking(
    const std::vector<double>& sendTop,
    const std::vector<double>& sendBot,
    std::vector<double>&       recvTop,
    std::vector<double>&       recvBot,
    int above, int below, MPI_Comm comm);

/*
 * Ring communication using blocking sends.
 * Each rank sends a token to its right neighbour and receives from left.
 * Returns the accumulated sum of tokens from all ranks.
 */
double ringCommunication(double myToken, MPI_Comm comm);
