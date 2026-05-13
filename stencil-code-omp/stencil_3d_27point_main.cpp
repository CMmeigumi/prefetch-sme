// 3D 27-point Stencil 主函数 - 对比标量和SME实现

#include "stencil_3d_27point.h"
#include <iostream>
#include <cstdlib>

#ifdef RUN_MAIN
int main() {
    std::cout << "3D 27-point Stencil 测试" << std::endl;
    const int DEPTH = 128, ROWS = 512, COLS = 512;

    double* g1 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));

    initializeGrid3D(g1, DEPTH, ROWS, COLS);

    std::cout << "标量版本测试 (stride=1)..." << std::endl;
    stencil3D_27point_omp(g1, g2, DEPTH, ROWS, COLS, 1);
    double avg = computeAverage3D(g2, DEPTH, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    std::cout << "标量版本测试 (stride=2)..." << std::endl;
    stencil3D_27point_omp(g1, g2, DEPTH, ROWS, COLS, 2);
    avg = computeAverage3D(g2, DEPTH, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

#ifdef __ARM_FEATURE_SME
    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    std::cout << "SME 版本测试 (stride=1)..." << std::endl;
    stencil3D_27point_sme(g1, g2, DEPTH, ROWS, COLS, 1);
    avg = computeAverage3D(g2, DEPTH, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    std::cout << "SME 版本测试 (stride=2)..." << std::endl;
    stencil3D_27point_sme(g1, g2, DEPTH, ROWS, COLS, 2);
    avg = computeAverage3D(g2, DEPTH, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;
#endif

    free(g1);
    free(g2);
    return 0;
}
#endif
