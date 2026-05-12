// ARM SME 2D 5-point Stencil 实现
// 形状：十字形（上下左右加中心）
// 应用：2D泊松方程、简单扩散

#include "stencil_2d_5point.h"

__arm_new("za")
void stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 5.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += SVL) {
            svbool_t pg = svwhilelt_b64(j, cols - 1);
            int base_idx = i * cols + j;

            svfloat64_t center = svld1_f64(pg, &grid[base_idx]);
            svfloat64_t up = svld1_f64(pg, &grid[(i-1) * cols + j]);
            svfloat64_t down = svld1_f64(pg, &grid[(i+1) * cols + j]);
            svfloat64_t left = svld1_f64(pg, &grid[i * cols + (j-1)]);
            svfloat64_t right = svld1_f64(pg, &grid[i * cols + (j+1)]);

            svzero_za();

            svmopa_za64_f64_m(0, pg_all, pg, ones, center);
            svmopa_za64_f64_m(0, pg_all, pg, ones, up);
            svmopa_za64_f64_m(0, pg_all, pg, ones, down);
            svmopa_za64_f64_m(0, pg_all, pg, ones, left);
            svmopa_za64_f64_m(0, pg_all, pg, ones, right);

            svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
            svfloat64_t result = svmul_f64_z(pg, sum, weight_vec);
            svst1_f64(pg, &new_grid[base_idx], result);
        }
    }
}
