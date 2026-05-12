// 3D 25-point Stencil 正确性对比测试

#include <iostream>
#include <chrono>
#include "stencil_3d_25point.h"

using namespace std;

const double TOLERANCE = 1e-10;

struct CompareResult {
    double maxDiff;
    int mismatchCount;
    bool passed;
};

CompareResult compareGrids3D(double* grid1, double* grid2, int depth, int rows, int cols) {
    CompareResult result = {0.0, 0, true};
    int plane_size = rows * cols;

    for (int k = 1; k < depth - 1; k++) {
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                int idx = k * plane_size + i * cols + j;
                double diff = fabs(grid1[idx] - grid2[idx]);
                if (diff > result.maxDiff) result.maxDiff = diff;
                if (diff > TOLERANCE) {
                    result.mismatchCount++;
                    result.passed = false;
                }
            }
        }
    }
    return result;
}

void printCompareResult(const CompareResult& result, const string& testName) {
    cout << "[" << testName << "] " << (result.passed ? "PASS" : "FAIL")
         << " 最大差异: " << result.maxDiff << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << " 3D 25-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int DEPTH = 32, ROWS = 64, COLS = 64, ITERATIONS = 10;
    size_t grid_size = DEPTH * ROWS * COLS;

    double* g1 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g3 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g4 = (double*)aligned_alloc(64, grid_size * sizeof(double));

    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    initializeGrid3D(g3, DEPTH, ROWS, COLS);

    stencil3D_25point_scalar(g1, g2, DEPTH, ROWS, COLS, 1);
    stencil3D_25point_sme(g3, g4, DEPTH, ROWS, COLS, 1);

    CompareResult r = compareGrids3D(g2, g4, DEPTH, ROWS, COLS);
    printCompareResult(r, "单次迭代");

    cout << "最终平均温度: " << computeAverage3D(g4, DEPTH, ROWS, COLS) << endl;

    free(g1); free(g2); free(g3); free(g4);
    return r.passed ? 0 : 1;
}
