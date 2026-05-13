// 1D 3-point Stencil SME 实现

#include "stencil_1d_3point.h"

#ifdef __ARM_FEATURE_SME
__arm_new("za")
void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 3.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < size - 1; i += stride) {
        for (int j = 0; j < SVL * stride; j += SVL) {
            int idx = i + j;
            if (idx >= size - 1) break;

            svbool_t pg = svwhilelt_b64(idx, size - 1);
            if (!svptest_any(svptrue_b64(), pg)) break;

            svfloat64_t center = svld1_f64(pg, &grid[idx]);
            svfloat64_t left = svld1_f64(pg, &grid[idx - 1]);
            svfloat64_t right = svld1_f64(pg, &grid[idx + 1]);

            svfloat64_t sum = svadd_x(pg, center, left);
            sum = svadd_x(pg, sum, right);

            svfloat64_t result = svmul_x(pg, sum, weight_vec);
            svst1_f64(pg, &new_grid[idx], result);
        }
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
    for (int iter = 0; iter < 100; iter++) {
        stencil1D_3point_sme(g1, g2, SIZE, 1);
    }

    for (int i = 0; i < SIZE; i++) {
        g1[i] = 1.0 + i;
    }
    std::cout << "执行 stride=2..." << std::endl;
    for (int iter = 0; iter < 100; iter++) {
        stencil1D_3point_sme(g1, g2, SIZE, 2);
    }

    free(g1);
    free(g2);
    return 0;
}
#endif
