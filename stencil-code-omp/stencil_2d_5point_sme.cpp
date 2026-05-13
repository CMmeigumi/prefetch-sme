// 2D 5-point Stencil SME 实现

#include "stencil_2d_5point.h"

#ifdef __ARM_FEATURE_SME
__arm_new("za")
void stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 5.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += SVL * stride) {
            int j_limit = (cols - 1 < j + SVL * stride - 1) ? cols - 1 : j + SVL * stride - 1;
            svbool_t pg = svwhilelt_b64(j, j_limit + 1);
            if (!svptest_any(svptrue_b64(), pg)) break;

            int base_idx = i * cols + j;

            svfloat64_t center = svld1_f64(pg, &grid[base_idx]);
            svfloat64_t top = svld1_f64(pg, &grid[base_idx - cols]);
            svfloat64_t bottom = svld1_f64(pg, &grid[base_idx + cols]);
            svfloat64_t left = svld1_f64(pg, &grid[base_idx - 1]);
            svfloat64_t right = svld1_f64(pg, &grid[base_idx + 1]);

            svfloat64_t sum = svadd_x(pg, center, top);
            sum = svadd_x(pg, sum, bottom);
            sum = svadd_x(pg, sum, left);
            sum = svadd_x(pg, sum, right);

            svfloat64_t result = svmul_x(pg, sum, weight_vec);
            svst1_f64(pg, &new_grid[base_idx], result);
        }
    }
}
#endif
