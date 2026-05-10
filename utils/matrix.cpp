#include "matrix.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

Matrix createMatrix(int rows, int cols, double initVal) {
    return Matrix(rows, std::vector<double>(cols, initVal));
}

Matrix randomMatrix(int rows, int cols, double minVal, double maxVal) {
    Matrix m(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i][j] = minVal + (double)rand() / RAND_MAX * (maxVal - minVal);
    return m;
}

void printMatrix(const Matrix& m, const std::string& label) {
    if (!label.empty()) std::cout << label << ":\n";
    int rows = m.size();
    int cols = rows > 0 ? m[0].size() : 0;
    int printRows = std::min(rows, 8);
    int printCols = std::min(cols, 8);
    for (int i = 0; i < printRows; i++) {
        for (int j = 0; j < printCols; j++)
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << m[i][j];
        if (cols > printCols) std::cout << " ...";
        std::cout << "\n";
    }
    if (rows > printRows) std::cout << "...\n";
    std::cout << "[" << rows << " x " << cols << "]\n";
}

Matrix flatToMatrix(const std::vector<double>& flat, int rows, int cols) {
    Matrix m(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i][j] = flat[i * cols + j];
    return m;
}

std::vector<double> matrixToFlat(const Matrix& m) {
    int rows = m.size();
    int cols = rows > 0 ? m[0].size() : 0;
    std::vector<double> flat(rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            flat[i * cols + j] = m[i][j];
    return flat;
}

bool matricesEqual(const Matrix& a, const Matrix& b, double tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); j++)
            if (std::fabs(a[i][j] - b[i][j]) > tol) return false;
    }
    return true;
}
