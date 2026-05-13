// 2D 9-point Stencil SME 实现
// 使用 ARM SME ZA MOPA 外积累加

#include "stencil_2d_9point.h"
#include <iostream>
#include <cstdlib>

__arm_new("za")
void stencil2D_9point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                          int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t weight_vec = svdup_f64(1.0 / 9.0);
    svbool_t pg_all = svptrue_b64();

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += SVL * stride) {
            int j_limit = (cols - 1 < j + SVL * stride - 1) ? cols - 1 : j + SVL * stride - 1;
            svbool_t pg = svwhilelt_b64(j, j_limit + 1);
            if (!svptest_any(svptrue_b64(), pg)) break;

            int base_idx = i * cols + j;

            svfloat64_t m_i1_j1 = svld1_f64(pg, &grid[(i-1) * cols + (j-1)]);
            svfloat64_t m_i1_j0 = svld1_f64(pg, &grid[(i-1) * cols + j]);
            svfloat64_t m_i1_jp1 = svld1_f64(pg, &grid[(i-1) * cols + (j+1)]);

            svfloat64_t m_i0_j1 = svld1_f64(pg, &grid[i * cols + (j-1)]);
            svfloat64_t center = svld1_f64(pg, &grid[i * cols + j]);
            svfloat64_t m_i0_jp1 = svld1_f64(pg, &grid[i * cols + (j+1)]);

            svfloat64_t m_ip1_j1 = svld1_f64(pg, &grid[(i+1) * cols + (j-1)]);
            svfloat64_t m_ip1_j0 = svld1_f64(pg, &grid[(i+1) * cols + j]);
            svfloat64_t m_ip1_jp1 = svld1_f64(pg, &grid[(i+1) * cols + (j+1)]);

            svfloat64_t sum = svadd_x(pg, m_i1_j1, m_i1_j0);
            sum = svadd_x(pg, sum, m_i1_jp1);
            sum = svadd_x(pg, sum, m_i0_j1);
            sum = svadd_x(pg, sum, center);
            sum = svadd_x(pg, sum, m_i0_jp1);
            sum = svadd_x(pg, sum, m_ip1_j1);
            sum = svadd_x(pg, sum, m_ip1_j0);
            sum = svadd_x(pg, sum, m_ip1_jp1);

            svfloat64_t result = svmul_x(pg, sum, weight_vec);
            svst1_f64(pg, &new_grid[base_idx], result);
        }
    }
}

#ifdef RUN_MAIN
int main() {
    std::cout << "2D 9-point SME 版本测试" << std::endl;
    const int ROWS = 1024, COLS = 1024;

    double* g1 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* g2 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            g1[i * COLS + j] = 1.0 + i * COLS + j;
        }
    }

    std::cout << "执行 stride=1..." << std::endl;
    for (int iter = 0; iter < 100; iter++) {
        stencil2D_9point_sme(g1, g2, ROWS, COLS, 1);
    }

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            g1[i * COLS + j] = 1.0 + i * COLS + j;
        }
    }
    std::cout << "执行 stride=2..." << std::endl;
    for (int iter = 0; iter < 100; iter++) {
        stencil2D_9point_sme(g1, g2, ROWS, COLS, 2);
    }

    free(g1);
    free(g2);
    return 0;
}
#endif
