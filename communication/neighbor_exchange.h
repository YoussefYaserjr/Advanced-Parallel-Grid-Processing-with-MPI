#pragma once
#include <mpi.h>
#include <vector>

// Simple 1D neighbour exchange using MPI_Sendrecv
void neighborExchange(
    const std::vector<double>& sendData,
    std::vector<double>&       recvData,
    int dest, int src, int tag, MPI_Comm comm);

// 2D neighbour exchange — four directions
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
    MPI_Comm comm);
