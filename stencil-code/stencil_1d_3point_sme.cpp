// ARM SME 1D 3-point Stencil 实现
// 形状：线性（左、中、右）
// 应用：一维对流/扩散问题

#include "stencil_1d_3point.h"

__arm_new("za")
void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 3.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < size - 1; i += SVL) {
        svbool_t pg = svwhilelt_b64(i, size - 1);

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
