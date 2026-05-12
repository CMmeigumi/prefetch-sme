// 3D 27-point Stencil 正确性对比测试
// 同时调用标量版本和SME版本，验证功能一致性

#include <iostream>
#include <chrono>
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

CompareResult compareGrids3D(double* grid1, double* grid2, int depth, int rows, int cols) {
    CompareResult result = {0.0, 0, 0, true};
    int plane_size = rows * cols;

    for (int k = 1; k < depth - 1; k++) {
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                int idx = k * plane_size + i * cols + j;
                result.totalPoints++;
                double diff = fabs(grid1[idx] - grid2[idx]);
                if (diff > result.maxDiff) {
                    result.maxDiff = diff;
                }
                if (diff > TOLERANCE) {
                    result.mismatchCount++;
                    result.passed = false;
                }
            }
        }
    }
    return result;
}

void printSamplePoints(double* grid, const string& label, int depth, int rows, int cols) {
    cout << "  " << label << " 采样点 (前5个内点):" << endl;
    int count = 0;
    for (int k = 1; k < depth - 1 && count < 5; k++) {
        for (int i = 1; i < rows - 1 && count < 5; i++) {
            for (int j = 1; j < cols - 1 && count < 5; j++) {
                int idx = k * rows * cols + i * cols + j;
                cout << "    [" << k << "," << i << "," << j << "] = " << fixed << setprecision(6) << grid[idx] << endl;
                count++;
            }
        }
    }
}

void printCompareResult(const CompareResult& result, const string& testName) {
    cout << "[" << testName << "] ";
    if (result.passed) {
        cout << "PASS";
    } else {
        cout << "FAIL";
    }
    cout << endl;
    cout << "  总计算点数: " << result.totalPoints << endl;
    cout << "  最大差异: " << scientific << result.maxDiff << endl;
    cout << "  不匹配点数: " << result.mismatchCount << " / " << result.totalPoints << endl;
    cout << "  容差阈值: " << scientific << TOLERANCE << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "  3D 27-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int DEPTH = 8;
    const int ROWS = 16;
    const int COLS = 16;
    const int ITERATIONS = 5;

    size_t grid_size = DEPTH * ROWS * COLS;

    double* grid_scalar    = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* new_grid_scalar = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* grid_sme       = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* new_grid_sme   = (double*)aligned_alloc(64, grid_size * sizeof(double));

    int totalPassed = 0;
    int totalTests = 0;

    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
    cout << "总元素数: " << grid_size << endl << endl;

    cout << "========== 测试1：单次迭代精确对比 ==========" << endl << endl;

    {
        totalTests++;
        int stride = 1;
        initializeGrid3D(grid_scalar, DEPTH, ROWS, COLS);
        initializeGrid3D(grid_sme, DEPTH, ROWS, COLS);

        cout << "【初始化后】" << endl;
        printSamplePoints(grid_scalar, "标量版本", DEPTH, ROWS, COLS);
        cout << "  初始平均: " << fixed << setprecision(6) << computeAverage3D(grid_scalar, DEPTH, ROWS, COLS) << endl << endl;

        stencil3D_27point_scalar(grid_scalar, new_grid_scalar, DEPTH, ROWS, COLS, stride);
        stencil3D_27point_sme(grid_sme, new_grid_sme, DEPTH, ROWS, COLS, stride);

        cout << "【标量版本计算后】" << endl;
        printSamplePoints(new_grid_scalar, "标量", DEPTH, ROWS, COLS);
        cout << "  平均: " << fixed << setprecision(6) << computeAverage3D(new_grid_scalar, DEPTH, ROWS, COLS) << endl << endl;

        cout << "【SME版本计算后】" << endl;
        printSamplePoints(new_grid_sme, "SME", DEPTH, ROWS, COLS);
        cout << "  平均: " << fixed << setprecision(6) << computeAverage3D(new_grid_sme, DEPTH, ROWS, COLS) << endl << endl;

        CompareResult result = compareGrids3D(new_grid_scalar, new_grid_sme, DEPTH, ROWS, COLS);
        printCompareResult(result, "单次迭代 stride=1");
        if (result.passed) {
            totalPassed++;
            cout << "  结果: 标量与SME版本输出完全一致!" << endl;
        } else {
            cout << "  警告: 标量与SME版本输出存在差异!" << endl;
        }
        cout << endl;
    }

    cout << "========== 测试2：多次迭代对比 ==========" << endl << endl;

    {
        totalTests++;
        int stride = 1;

        initializeGrid3D(grid_scalar, DEPTH, ROWS, COLS);
        initializeGrid3D(grid_sme, DEPTH, ROWS, COLS);

        for (int iter = 0; iter < ITERATIONS; iter++) {
            stencil3D_27point_scalar(grid_scalar, new_grid_scalar, DEPTH, ROWS, COLS, stride);
            swap(grid_scalar, new_grid_scalar);
            stencil3D_27point_sme(grid_sme, new_grid_sme, DEPTH, ROWS, COLS, stride);
            swap(grid_sme, new_grid_sme);
        }

        cout << "迭代 " << ITERATIONS << " 次后:" << endl;
        cout << "  标量版本平均: " << fixed << setprecision(6) << computeAverage3D(grid_scalar, DEPTH, ROWS, COLS) << endl;
        cout << "  SME版本平均:   " << fixed << setprecision(6) << computeAverage3D(grid_sme, DEPTH, ROWS, COLS) << endl << endl;

        CompareResult result = compareGrids3D(grid_scalar, grid_sme, DEPTH, ROWS, COLS);
        printCompareResult(result, "多次迭代 stride=1");
        if (result.passed) {
            totalPassed++;
        }
        cout << endl;
    }

    cout << "========== 测试3：stride=2 间隔计算 ==========" << endl << endl;

    {
        totalTests++;
        int stride = 2;

        initializeGrid3D(grid_scalar, DEPTH, ROWS, COLS);
        initializeGrid3D(grid_sme, DEPTH, ROWS, COLS);

        stencil3D_27point_scalar(grid_scalar, new_grid_scalar, DEPTH, ROWS, COLS, stride);
        stencil3D_27point_sme(grid_sme, new_grid_sme, DEPTH, ROWS, COLS, stride);

        cout << "stride=" << stride << " 时，两版本计算的是相同的稀疏点集" << endl << endl;

        CompareResult result = compareGrids3D(new_grid_scalar, new_grid_sme, DEPTH, ROWS, COLS);
        printCompareResult(result, "stride=2");
        if (result.passed) {
            totalPassed++;
        }
        cout << endl;
    }

    free(grid_scalar);
    free(new_grid_scalar);
    free(grid_sme);
    free(new_grid_sme);

    cout << "========================================" << endl;
    cout << "           最终测试总结" << endl;
    cout << "========================================" << endl;
    cout << "通过: " << totalPassed << " / " << totalTests << endl;

    if (totalPassed == totalTests) {
        cout << endl;
        cout << "  [成功] SME intrinsic 版本功能正确！" << endl;
        cout << "  标量版本与 SME 版本输出一致。" << endl;
    } else {
        cout << endl;
        cout << "  [失败] 存在 " << (totalTests - totalPassed) << " 个测试未通过！" << endl;
    }
    cout << "========================================" << endl;

    return (totalPassed == totalTests) ? 0 : 1;
}
