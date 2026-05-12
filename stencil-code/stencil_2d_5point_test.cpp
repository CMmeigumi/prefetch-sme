// 2D 5-point Stencil 正确性对比测试

#include <iostream>
#include <iomanip>
#include "stencil_2d_5point.h"

using namespace std;

const double TOLERANCE = 1e-10;

struct CompareResult {
    double maxDiff;
    int mismatchCount;
    int totalPoints;
    bool passed;
};

CompareResult compareGrids2D(double* grid1, double* grid2, int rows, int cols) {
    CompareResult result = {0.0, 0, 0, true};

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int idx = i * cols + j;
            result.totalPoints++;
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

void printSamplePoints(double* grid, const string& label, int rows, int cols) {
    cout << "  " << label << " 采样点 (前5个内点):" << endl;
    int count = 0;
    for (int i = 1; i < rows - 1 && count < 5; i++) {
        for (int j = 1; j < cols - 1 && count < 5; j++) {
            int idx = i * cols + j;
            cout << "    [" << i << "," << j << "] = " << fixed << setprecision(6) << grid[idx] << endl;
            count++;
        }
    }
}

void printCompareResult(const CompareResult& result, const string& testName) {
    cout << "[" << testName << "] " << (result.passed ? "PASS" : "FAIL") << endl;
    cout << "  总计算点数: " << result.totalPoints << endl;
    cout << "  最大差异: " << scientific << result.maxDiff << endl;
    cout << "  不匹配点数: " << result.mismatchCount << " / " << result.totalPoints << endl;
    cout << "  容差阈值: " << scientific << TOLERANCE << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << " 2D 5-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int ROWS = 16, COLS = 16;
    size_t grid_size = ROWS * COLS;

    double* g1 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g3 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g4 = (double*)aligned_alloc(64, grid_size * sizeof(double));

    cout << "网格大小: " << ROWS << " x " << COLS << endl << endl;

    initializeGrid2D(g1, ROWS, COLS);
    initializeGrid2D(g3, ROWS, COLS);

    stencil2D_5point_scalar(g1, g2, ROWS, COLS, 1);
    stencil2D_5point_sme(g3, g4, ROWS, COLS, 1);

    cout << "【标量版本计算后】" << endl;
    printSamplePoints(g2, "标量", ROWS, COLS);
    cout << "  平均: " << fixed << setprecision(6) << computeAverage2D(g2, ROWS, COLS) << endl << endl;

    cout << "【SME版本计算后】" << endl;
    printSamplePoints(g4, "SME", ROWS, COLS);
    cout << "  平均: " << fixed << setprecision(6) << computeAverage2D(g4, ROWS, COLS) << endl << endl;

    CompareResult r = compareGrids2D(g2, g4, ROWS, COLS);
    printCompareResult(r, "单次迭代");

    free(g1); free(g2); free(g3); free(g4);
    return r.passed ? 0 : 1;
}
