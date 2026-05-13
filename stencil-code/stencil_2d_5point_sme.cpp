// ARM SME 2D 5-point Stencil 实现
// 形状：十字形（上下左右加中心）
// 应用：2D泊松方程、简单扩散

#include "stencil_2d_5point.h"
#include <iostream>

__arm_new("za")
void stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 5.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += SVL * stride) {
            int j_limit = (cols - 1 < j + SVL * stride - 1) ? cols - 1 : j + SVL * stride - 1;
            svbool_t pg = svwhilelt_b64(j, j_limit + 1);
            if (!svptest_any(svptrue_b64(), pg)) break;

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

#ifdef RUN_MAIN
int main() {
    std::cout << "2D 5-point SME 版本测试" << std::endl;
    const int ROWS = 16, COLS = 16;
    size_t grid_size = ROWS * COLS;

    double* g1 = (double*)aligned_alloc(64, grid_size * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, grid_size * sizeof(double));

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            g1[i * COLS + j] = 1.0 + i * COLS + j;
        }
    }

    std::cout << "执行 stride=1..." << std::endl;
    stencil2D_5point_sme(g1, g2, ROWS, COLS, 1);
    
    double sum = 0.0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            sum += g2[i * COLS + j];
        }
    }
    std::cout << "  平均: " << sum / grid_size << std::endl;

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            g1[i * COLS + j] = 1.0 + i * COLS + j;
        }
    }
    std::cout << "执行 stride=2..." << std::endl;
    stencil2D_5point_sme(g1, g2, ROWS, COLS, 2);
    
    sum = 0.0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            sum += g2[i * COLS + j];
        }
    }
    std::cout << "  平均: " << sum / grid_size << std::endl;

    free(g1);
    free(g2);
    return 0;
}
#endif
