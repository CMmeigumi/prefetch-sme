// 3D 13-point Stencil 主函数 - SME 版本

#include "stencil_3d_13point.h"
#include <iostream>
#include <cstdlib>

#ifdef RUN_MAIN_SME
int main() {
    std::cout << "3D 13-point SME 版本测试" << std::endl;
    const int DEPTH = 128, ROWS = 512, COLS = 512;

    double* g1 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));

    initializeGrid3D(g1, DEPTH, ROWS, COLS);

    std::cout << "执行 stride=1..." << std::endl;
    stencil3D_13point_sme(g1, g2, DEPTH, ROWS, COLS, 1);

    double avg = computeAverage3D(g2, DEPTH, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid3D(g1, DEPTH, ROWS, COLS);
    std::cout << "执行 stride=2..." << std::endl;
    stencil3D_13point_sme(g1, g2, DEPTH, ROWS, COLS, 2);

    avg = computeAverage3D(g2, DEPTH, ROWS, COLS);
    std::cout << "  平均: " << avg << std::endl;

    free(g1);
    free(g2);
    return 0;
}
#endif
