// 1D 3-point Stencil 主函数 - 对比标量和SME实现

#include "stencil_1d_3point.h"
#include <iostream>
#include <cstdlib>

#ifdef RUN_MAIN
int main() {
    std::cout << "1D 3-point Stencil 测试" << std::endl;
    const int SIZE = 1048576;

    double* g1 = (double*)aligned_alloc(64, SIZE * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, SIZE * sizeof(double));

    initializeGrid1D(g1, SIZE);

    std::cout << "标量版本测试 (stride=1)..." << std::endl;
    stencil1D_3point_omp(g1, g2, SIZE, 1);
    double avg = computeAverage1D(g2, SIZE);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid1D(g1, SIZE);
    std::cout << "标量版本测试 (stride=2)..." << std::endl;
    stencil1D_3point_omp(g1, g2, SIZE, 2);
    avg = computeAverage1D(g2, SIZE);
    std::cout << "  平均: " << avg << std::endl;

#ifdef __ARM_FEATURE_SME
    initializeGrid1D(g1, SIZE);
    std::cout << "SME 版本测试 (stride=1)..." << std::endl;
    stencil1D_3point_sme(g1, g2, SIZE, 1);
    avg = computeAverage1D(g2, SIZE);
    std::cout << "  平均: " << avg << std::endl;

    initializeGrid1D(g1, SIZE);
    std::cout << "SME 版本测试 (stride=2)..." << std::endl;
    stencil1D_3point_sme(g1, g2, SIZE, 2);
    avg = computeAverage1D(g2, SIZE);
    std::cout << "  平均: " << avg << std::endl;
#endif

    free(g1);
    free(g2);
    return 0;
}
#endif
