#include "neighbor_exchange.h"
#include <iostream>

/*
 * Neighbour exchange using MPI_Sendrecv — a single call that avoids
 * the classic deadlock scenario of naive paired MPI_Send/MPI_Recv.
 *
 * MPI_Sendrecv internally manages ordering and buffering, so it is
 * safe even when all processes call it simultaneously.
 */
void neighborExchange(
    const std::vector<double>& sendData,
    std::vector<double>&       recvData,
    int dest, int src, int tag, MPI_Comm comm)
{
    int N = (int)sendData.size();
    recvData.resize(N);
    MPI_Status status;

    MPI_Sendrecv(
        sendData.data(), N, MPI_DOUBLE, dest, tag,
        recvData.data(), N, MPI_DOUBLE, src,  tag,
        comm, &status);
}

/*
 * 2D Neighbour exchange — each rank sends its local block edges to all
 * four cardinal neighbours and receives ghost edges in return.
 * Uses MPI_Sendrecv for each direction.
 */
void neighborExchange2D(
    const std::vector<double>& sendUp,
    const std::vector<double>& sendDown,
    const std::vector<double>& sendLeft,
    const std::vector<double>& sendRight,
    std::vector<double>&       recvUp,
    std::vector<double>&       recvDown,
    std::vector<double>&       recvLeft,
    std::vector<double>&       recvRight,
    int up, int down, int left, int right,
    MPI_Comm comm)
{
    int rowN = (int)sendUp.size();
    int colN = (int)sendLeft.size();
    MPI_Status status;

    recvUp.resize(rowN);
    recvDown.resize(rowN);
    recvLeft.resize(colN);
    recvRight.resize(colN);

    // Up/Down exchange
    MPI_Sendrecv(sendUp.data(),   rowN, MPI_DOUBLE, up,   30,
                 recvDown.data(), rowN, MPI_DOUBLE, down, 30, comm, &status);
    MPI_Sendrecv(sendDown.data(), rowN, MPI_DOUBLE, down, 31,
                 recvUp.data(),   rowN, MPI_DOUBLE, up,   31, comm, &status);

    // Left/Right exchange
    MPI_Sendrecv(sendLeft.data(),  colN, MPI_DOUBLE, left,  32,
                 recvRight.data(), colN, MPI_DOUBLE, right, 32, comm, &status);
    MPI_Sendrecv(sendRight.data(), colN, MPI_DOUBLE, right, 33,
                 recvLeft.data(),  colN, MPI_DOUBLE, left,  33, comm, &status);
}
