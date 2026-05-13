// 1D 3-point Stencil OpenMP 实现

#include "stencil_1d_3point.h"
#include <omp.h>

void stencil1D_3point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride) {
    double weight = 1.0 / 3.0;

    #pragma omp parallel for schedule(static)
    for (int i = 1; i < size - 1; i += stride) {
        double sum = grid[i-1] + grid[i] + grid[i+1];
        new_grid[i] = sum * weight;
    }
}

#ifdef __ARM_FEATURE_SME
void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride)
    __arm_new("za")
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
#endif
