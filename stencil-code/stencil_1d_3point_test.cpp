// 1D 3-point Stencil 正确性对比测试

#include <iostream>
#include <chrono>
#include "stencil_1d_3point.h"

using namespace std;

const double TOLERANCE = 1e-10;

struct CompareResult {
    double maxDiff;
    int mismatchCount;
    bool passed;
};

CompareResult compareGrids1D(double* grid1, double* grid2, int size) {
    CompareResult result = {0.0, 0, true};

    for (int i = 1; i < size - 1; i++) {
        double diff = fabs(grid1[i] - grid2[i]);
        if (diff > result.maxDiff) result.maxDiff = diff;
        if (diff > TOLERANCE) {
            result.mismatchCount++;
            result.passed = false;
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
    cout << " 1D 3-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int SIZE = 1024;
    size_t grid_size = SIZE;

    double* g1 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g3 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g4 = (double*)aligned_alloc(64, grid_size * sizeof(double));

    initializeGrid1D(g1, SIZE);
    initializeGrid1D(g3, SIZE);

    stencil1D_3point_scalar(g1, g2, SIZE, 1);
    stencil1D_3point_sme(g3, g4, SIZE, 1);

    CompareResult r = compareGrids1D(g2, g4, SIZE);
    printCompareResult(r, "单次迭代");

    cout << "最终平均温度: " << computeAverage1D(g4, SIZE) << endl;

    free(g1); free(g2); free(g3); free(g4);
    return r.passed ? 0 : 1;
}
