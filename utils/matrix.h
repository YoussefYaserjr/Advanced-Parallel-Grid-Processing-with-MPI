#pragma once
#include <vector>
#include <string>

using Matrix = std::vector<std::vector<double>>;

Matrix createMatrix(int rows, int cols, double initVal = 0.0);
Matrix randomMatrix(int rows, int cols, double minVal = 0.0, double maxVal = 100.0);
void printMatrix(const Matrix& m, const std::string& label = "");
Matrix flatToMatrix(const std::vector<double>& flat, int rows, int cols);
std::vector<double> matrixToFlat(const Matrix& m);
bool matricesEqual(const Matrix& a, const Matrix& b, double tol = 1e-6);
