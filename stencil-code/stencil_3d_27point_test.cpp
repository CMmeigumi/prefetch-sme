-// 3D 27-point Stencil 正确性对比测试

#include <iostream>
#include <iomanip>
#include "stencil_3d_27point.h"

using namespace std;

const double TOLERANCE = 1e-10;

struct CompareResult {
    double maxDiff;
    int mismatchCount;
    int totalPoints;
    bool passed;
};

CompareResult compareGrids3D(double* grid1, double* grid2, int depth, int rows, int cols, int stride) {
    CompareResult result = {0.0, 0, 0, true};
    int plane_size = rows * cols;

    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += stride) {
                int idx = k * plane_size + i * cols + j;
                result.totalPoints++;
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

void printSamplePoints(double* grid, const string& label, int depth, int rows, int cols, int stride) {
    cout << "  " << label << " 采样点 (前5个计算点):" << endl;
    int count = 0;
    for (int k = 1; k < depth - 1 && count < 5; k += stride) {
        for (int i = 1; i < rows - 1 && count < 5; i += stride) {
            for (int j = 1; j < cols - 1 && count < 5; j += stride) {
                int idx = k * rows * cols + i * cols + j;
                cout << "    [" << k << "," << i << "," << j << "] = " << fixed << setprecision(6) << grid[idx] << endl;
                count++;
            }
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
    cout << " 3D 27-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int DEPTH = 8, ROWS = 16, COLS = 16;
    size_t grid_size = DEPTH * ROWS * COLS;

    double* g1 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g3 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g4 = (double*)aligned_alloc(64, grid_size * sizeof(double));

    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl << endl;

    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    initializeGrid3D(g3, DEPTH, ROWS, COLS);

    cout << "【初始化后】" << endl;
    printSamplePoints(g1, "标量", DEPTH, ROWS, COLS, 1);
    cout << "  初始平均: " << fixed << setprecision(6) << computeAverage3D(g1, DEPTH, ROWS, COLS) << endl << endl;

    stencil3D_27point_scalar(g1, g2, DEPTH, ROWS, COLS, 1);
    stencil3D_27point_sme(g3, g4, DEPTH, ROWS, COLS, 1);

    cout << "【stride=1 标量版本计算后】" << endl;
    printSamplePoints(g2, "标量", DEPTH, ROWS, COLS, 1);
    cout << "  平均: " << fixed << setprecision(6) << computeAverage3D(g2, DEPTH, ROWS, COLS) << endl << endl;

    cout << "【stride=1 SME版本计算后】" << endl;
    printSamplePoints(g4, "SME", DEPTH, ROWS, COLS, 1);
    cout << "  平均: " << fixed << setprecision(6) << computeAverage3D(g4, DEPTH, ROWS, COLS) << endl << endl;

    CompareResult r1 = compareGrids3D(g2, g4, DEPTH, ROWS, COLS, 1);
    printCompareResult(r1, "stride=1");

    if (r1.passed) {
        cout << "  结果: 标量与SME版本输出完全一致!" << endl;
    } else {
        cout << "  警告: 标量与SME版本输出存在差异!" << endl;
    }

    cout << endl;

    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    initializeGrid3D(g3, DEPTH, ROWS, COLS);

    stencil3D_27point_scalar(g1, g2, DEPTH, ROWS, COLS, 2);
    stencil3D_27point_sme(g3, g4, DEPTH, ROWS, COLS, 2);

    cout << "【stride=2 标量版本计算后】" << endl;
    printSamplePoints(g2, "标量", DEPTH, ROWS, COLS, 2);
    cout << "  平均: " << fixed << setprecision(6) << computeAverage3D(g2, DEPTH, ROWS, COLS) << endl << endl;

    cout << "【stride=2 SME版本计算后】" << endl;
    printSamplePoints(g4, "SME", DEPTH, ROWS, COLS, 2);
    cout << "  平均: " << fixed << setprecision(6) << computeAverage3D(g4, DEPTH, ROWS, COLS) << endl << endl;

    CompareResult r2 = compareGrids3D(g2, g4, DEPTH, ROWS, COLS, 2);
    printCompareResult(r2, "stride=2");

    if (r2.passed) {
        cout << "  结果: 标量与SME版本输出完全一致!" << endl;
    } else {
        cout << "  警告: 标量与SME版本输出存在差异!" << endl;
    }

    free(g1); free(g2); free(g3); free(g4);
    return (r1.passed && r2.passed) ? 0 : 1;
}
