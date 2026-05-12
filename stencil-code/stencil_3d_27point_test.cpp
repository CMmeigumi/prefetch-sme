// 3D 27-point Stencil 正确性对比测试
// 同时调用标量版本和SME版本，验证功能一致性

#include <iostream>
#include <chrono>
#include "stencil_3d_27point.h"

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

void printCompareResult(const CompareResult& result, const string& testName) {
    cout << "[" << testName << "] ";
    if (result.passed) {
        cout << "PASS  最大差异: " << result.maxDiff << " (容差: " << TOLERANCE << ")" << endl;
    } else {
        cout << "FAIL  最大差异: " << result.maxDiff
             << "  不匹配点数: " << result.mismatchCount
             << " (容差: " << TOLERANCE << ")" << endl;
    }
}

int main() {
    cout << "========================================" << endl;
    cout << "  3D 27-point Stencil 正确性对比测试" << endl;
    cout << "========================================" << endl << endl;

    const int DEPTH = 32;
    const int ROWS = 64;
    const int COLS = 64;
    const int ITERATIONS = 10;

    size_t grid_size = DEPTH * ROWS * COLS;

    double* grid_scalar    = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* new_grid_scalar = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* grid_sme       = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* new_grid_sme   = (double*)aligned_alloc(64, grid_size * sizeof(double));

    int totalPassed = 0;
    int totalTests = 0;

    cout << "========== 正确性验证：单次迭代精确对比 ==========" << endl << endl;

    {
        totalTests++;
        int stride = 1;
        initializeGrid3D(grid_scalar, DEPTH, ROWS, COLS);
        initializeGrid3D(grid_sme, DEPTH, ROWS, COLS);

        stencil3D_27point_scalar(grid_scalar, new_grid_scalar, DEPTH, ROWS, COLS, stride);
        stencil3D_27point_sme(grid_sme, new_grid_sme, DEPTH, ROWS, COLS, stride);

        CompareResult result = compareGrids3D(new_grid_scalar, new_grid_sme, DEPTH, ROWS, COLS);
        printCompareResult(result, "单次迭代 stride=1");
        if (result.passed) totalPassed++;
        cout << endl;
    }

    cout << "========== 性能测试：多次迭代对比 ==========" << endl << endl;

    cout << "【测试】stride = 1 (全量计算)" << endl;
    cout << "----------------------------------------" << endl;

    {
        totalTests++;
        int stride = 1;

        initializeGrid3D(grid_scalar, DEPTH, ROWS, COLS);
        initializeGrid3D(grid_sme, DEPTH, ROWS, COLS);

        cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
        cout << "迭代次数: " << ITERATIONS << endl;
        cout << "stride: " << stride << endl;
        cout << "初始平均温度: " << computeAverage3D(grid_scalar, DEPTH, ROWS, COLS) << endl;

        auto start = chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; iter++) {
            stencil3D_27point_scalar(grid_scalar, new_grid_scalar, DEPTH, ROWS, COLS, stride);
            swap(grid_scalar, new_grid_scalar);
        }
        auto end = chrono::high_resolution_clock::now();
        auto dur_scalar = chrono::duration_cast<chrono::milliseconds>(end - start);

        start = chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; iter++) {
            stencil3D_27point_sme(grid_sme, new_grid_sme, DEPTH, ROWS, COLS, stride);
            swap(grid_sme, new_grid_sme);
        }
        end = chrono::high_resolution_clock::now();
        auto dur_sme = chrono::duration_cast<chrono::milliseconds>(end - start);

        CompareResult result = compareGrids3D(grid_scalar, grid_sme, DEPTH, ROWS, COLS);

        cout << "标量版本执行时间: " << dur_scalar.count() << " ms" << endl;
        cout << "SME 版本执行时间: " << dur_sme.count() << " ms" << endl;
        if (dur_sme.count() > 0) {
            cout << "加速比: " << (double)dur_scalar.count() / dur_sme.count() << "x" << endl;
        }
        printCompareResult(result, "多次迭代 stride=1");
        if (result.passed) totalPassed++;
        cout << endl;
    }

    cout << "【测试】stride = 2 (间隔计算)" << endl;
    cout << "----------------------------------------" << endl;

    {
        totalTests++;
        int stride = 2;

        initializeGrid3D(grid_scalar, DEPTH, ROWS, COLS);
        initializeGrid3D(grid_sme, DEPTH, ROWS, COLS);

        cout << "stride: " << stride << endl;

        auto start = chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; iter++) {
            stencil3D_27point_scalar(grid_scalar, new_grid_scalar, DEPTH, ROWS, COLS, stride);
            swap(grid_scalar, new_grid_scalar);
        }
        auto end = chrono::high_resolution_clock::now();
        auto dur_scalar = chrono::duration_cast<chrono::milliseconds>(end - start);

        start = chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; iter++) {
            stencil3D_27point_sme(grid_sme, new_grid_sme, DEPTH, ROWS, COLS, stride);
            swap(grid_sme, new_grid_sme);
        }
        end = chrono::high_resolution_clock::now();
        auto dur_sme = chrono::duration_cast<chrono::milliseconds>(end - start);

        CompareResult result = compareGrids3D(grid_scalar, grid_sme, DEPTH, ROWS, COLS);

        cout << "标量版本执行时间: " << dur_scalar.count() << " ms" << endl;
        cout << "SME 版本执行时间: " << dur_sme.count() << " ms" << endl;
        if (dur_sme.count() > 0) {
            cout << "加速比: " << (double)dur_scalar.count() / dur_sme.count() << "x" << endl;
        }
        cout << "注意: stride>1 时标量和SME计算的点集可能不同" << endl;
        printCompareResult(result, "多次迭代 stride=2");
        if (result.passed) totalPassed++;
        cout << endl;
    }

    free(grid_scalar);
    free(new_grid_scalar);
    free(grid_sme);
    free(new_grid_sme);

    cout << "========================================" << endl;
    cout << "           最终正确性总结" << endl;
    cout << "========================================" << endl;
    cout << "通过: " << totalPassed << " / " << totalTests << endl;

    if (totalPassed == totalTests) {
        cout << endl;
        cout << "  SME intrinsic 版本功能正确！" << endl;
        cout << "  标量版本与 SME 版本输出一致，" << endl;
        cout << "  最大差异在容差 " << TOLERANCE << " 以内。" << endl;
    } else {
        cout << endl;
        cout << "  存在 " << (totalTests - totalPassed) << " 个测试未通过！" << endl;
        cout << "  请检查 SME intrinsic 实现。" << endl;
    }
    cout << "========================================" << endl;

    return (totalPassed == totalTests) ? 0 : 1;
}