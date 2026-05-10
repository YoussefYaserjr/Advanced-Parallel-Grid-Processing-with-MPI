#pragma once
#include "matrix.h"
#include <string>

// Load a matrix from a text file.
// Format: first line is "rows cols", then rows lines of cols doubles.
Matrix loadMatrix(const std::string& filename);

// Save a matrix to a text file in the same format.
void saveMatrix(const Matrix& m, const std::string& filename);

// Generate large random matrix files for testing
void generateMatrixFile(const std::string& filename, int rows, int cols,
                        double minVal = 0.0, double maxVal = 10.0);
