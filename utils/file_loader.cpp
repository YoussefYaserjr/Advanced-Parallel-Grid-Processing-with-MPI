#include "file_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

Matrix loadMatrix(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int rows, cols;
    f >> rows >> cols;
    if (rows <= 0 || cols <= 0)
        throw std::runtime_error("Invalid matrix dimensions in file: " + filename);

    Matrix m(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (!(f >> m[i][j]))
                throw std::runtime_error("Unexpected end of file: " + filename);

    std::cout << "[FileLoader] Loaded " << rows << "x" << cols
              << " matrix from '" << filename << "'\n";
    return m;
}

void saveMatrix(const Matrix& m, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot write file: " + filename);

    int rows = m.size();
    int cols = rows > 0 ? (int)m[0].size() : 0;
    f << rows << " " << cols << "\n";
    f << std::fixed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            f << m[i][j];
            if (j < cols - 1) f << " ";
        }
        f << "\n";
    }
    std::cout << "[FileLoader] Saved " << rows << "x" << cols
              << " matrix to '" << filename << "'\n";
}

void generateMatrixFile(const std::string& filename, int rows, int cols,
                        double minVal, double maxVal) {
    std::ofstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot write file: " + filename);

    f << rows << " " << cols << "\n";
    f << std::fixed;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double v = minVal + (double)rand() / RAND_MAX * (maxVal - minVal);
            f << v;
            if (j < cols - 1) f << " ";
        }
        f << "\n";
    }
    std::cout << "[FileLoader] Generated " << rows << "x" << cols
              << " matrix file '" << filename << "'\n";
}
