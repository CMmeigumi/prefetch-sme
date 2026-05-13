// 2D 5-point Stencil 主函数 - OpenMP 版本

#include "stencil_2d_5point.h"
#include <iostream>
#include <cstdlib>

#ifdef RUN_MAIN
int main() {
    std::cout << "2D 5-point OpenMP 版本测试" << std::endl;
    const int ROWS = 1024, COLS = 1024;

    double* g1 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));

    initializeGrid2D(g1, ROWS, COLS);

    std::cout << "执行 stride=1..." << std::endl;
    stencil2D_5point_omp(g1, g2, ROWS, COLS, 1);

    double avg = computeAverage2D(g2, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid2D(g1, ROWS, COLS);
    std::cout << "执行 stride=2..." << std::endl;
    stencil2D_5point_omp(g1, g2, ROWS, COLS, 2);

    avg = computeAverage2D(g2, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    free(g1);
    free(g2);
    return 0;
}
#endif
