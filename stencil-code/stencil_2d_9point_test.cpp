// 2D 9-point Stencil 正确性对比测试

#include <iostream>
#include <chrono>
#include "stencil_2d_9point.h"

using namespace std;

const double TOLERANCE = 1e-10;

struct CompareResult {
    double maxDiff;
    int mismatchCount;
    bool passed;
};

CompareResult compareGrids2D(double* grid1, double* grid2, int rows, int cols) {
    CompareResult result = {0.0, 0, true};

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int idx = i * cols + j;
            double diff = fabs(grid1[idx] - grid2[idx]);
            if (diff > result.maxDiff) result.maxDiff = diff;
            if (diff > TOLERANCE) {
                result.mismatchCount++;
                result.passed = false;
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
    cout << " 2D 9-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int ROWS = 64, COLS = 64;
    size_t grid_size = ROWS * COLS;

    double* g1 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g3 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g4 = (double*)aligned_alloc(64, grid_size * sizeof(double));

    initializeGrid2D(g1, ROWS, COLS);
    initializeGrid2D(g3, ROWS, COLS);

    stencil2D_9point_scalar(g1, g2, ROWS, COLS, 1);
    stencil2D_9point_sme(g3, g4, ROWS, COLS, 1);

    CompareResult r = compareGrids2D(g2, g4, ROWS, COLS);
    printCompareResult(r, "单次迭代");

    cout << "最终平均温度: " << computeAverage2D(g4, ROWS, COLS) << endl;

    free(g1); free(g2); free(g3); free(g4);
    return r.passed ? 0 : 1;
}
