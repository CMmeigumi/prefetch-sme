// 2D 9-point Stencil 主函数 - 对比标量和SME实现

#include "stencil_2d_9point.h"
#include <iostream>
#include <cstdlib>

#ifdef RUN_MAIN
int main() {
    std::cout << "2D 9-point Stencil 测试" << std::endl;
    const int ROWS = 1024, COLS = 1024;

    double* g1 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));

    initializeGrid2D(g1, ROWS, COLS);

    std::cout << "标量版本测试 (stride=1)..." << std::endl;
    stencil2D_9point_omp(g1, g2, ROWS, COLS, 1);
    double avg = computeAverage2D(g2, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid2D(g1, ROWS, COLS);
    std::cout << "标量版本测试 (stride=2)..." << std::endl;
    stencil2D_9point_omp(g1, g2, ROWS, COLS, 2);
    avg = computeAverage2D(g2, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

#ifdef __ARM_FEATURE_SME
    initializeGrid2D(g1, ROWS, COLS);
    std::cout << "SME 版本测试 (stride=1)..." << std::endl;
    stencil2D_9point_sme(g1, g2, ROWS, COLS, 1);
    avg = computeAverage2D(g2, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid2D(g1, ROWS, COLS);
    std::cout << "SME 版本测试 (stride=2)..." << std::endl;
    stencil2D_9point_sme(g1, g2, ROWS, COLS, 2);
    avg = computeAverage2D(g2, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;
#endif

    free(g1);
    free(g2);
    return 0;
}
#endif
