// ARM SME 1D 3-point Stencil 实现
// 形状：线性（左、中、右）
// 应用：一维对流/扩散问题

#include "stencil_1d_3point.h"
#include <iostream>

__arm_new("za")
void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 3.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < size - 1; i += SVL * stride) {
        int i_limit = (size - 2 < i + SVL * stride - 1) ? size - 2 : i + SVL * stride - 1;
        svbool_t pg = svwhilelt_b64(i, i_limit + 1);
        if (!svptest_any(svptrue_b64(), pg)) break;

        svfloat64_t left = svld1_f64(pg, &grid[i - 1]);
        svfloat64_t center = svld1_f64(pg, &grid[i]);
        svfloat64_t right = svld1_f64(pg, &grid[i + 1]);

        svzero_za();

        svmopa_za64_f64_m(0, pg_all, pg, ones, left);
        svmopa_za64_f64_m(0, pg_all, pg, ones, center);
        svmopa_za64_f64_m(0, pg_all, pg, ones, right);

        svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
        svfloat64_t result = svmul_f64_z(pg, sum, weight_vec);
        svst1_f64(pg, &new_grid[i], result);
    }
}

#ifdef RUN_MAIN
int main() {
    std::cout << "1D 3-point SME 版本测试" << std::endl;
    const int SIZE = 1048576;

    double* g1 = (double*)aligned_alloc(64, SIZE * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, SIZE * sizeof(double));

    for (int i = 0; i < SIZE; i++) {
        g1[i] = 1.0 + i;
    }

    std::cout << "执行 stride=1..." << std::endl;
    stencil1D_3point_sme(g1, g2, SIZE, 1);
    
    double sum = 0.0;
    for (int i = 0; i < SIZE; i++) {
        sum += g2[i];
    }
    std::cout << "  平均: " << sum / SIZE << std::endl;

    for (int i = 0; i < SIZE; i++) {
        g1[i] = 1.0 + i;
    }
    std::cout << "执行 stride=2..." << std::endl;
    stencil1D_3point_sme(g1, g2, SIZE, 2);
    
    sum = 0.0;
    for (int i = 0; i < SIZE; i++) {
        sum += g2[i];
    }
    std::cout << "  平均: " << sum / SIZE << std::endl;

    free(g1);
    free(g2);
    return 0;
}
#endif
