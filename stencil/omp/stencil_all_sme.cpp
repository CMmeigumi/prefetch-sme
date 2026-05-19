#include <arm_sme.h>
#include <arm_sve.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cstdlib>

__arm_new("za")
void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 3.0);
    svbool_t pg_all = svptrue_b64();

    #pragma omp parallel for
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

__arm_new("za")
void stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                          int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    svfloat64_t weight_vec = svdup_f64(1.0 / 5.0);
    svbool_t pg_all = svptrue_b64();

    #pragma omp parallel for
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

__arm_new("za")
void stencil2D_9point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                          int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t weight_vec = svdup_f64(1.0 / 9.0);
    svbool_t pg_all = svptrue_b64();

    #pragma omp parallel for
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

__arm_new("za")
void stencil3D_13point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                           int depth, int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t weight_vec = svdup_f64(1.0 / 13.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    #pragma omp parallel for
    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += SVL * stride) {
                int j_limit = (cols - 1 < j + SVL * stride - 1) ? cols - 1 : j + SVL * stride - 1;
                svbool_t pg = svwhilelt_b64(j, j_limit + 1);
                if (!svptest_any(svptrue_b64(), pg)) break;

                int base_idx = k * plane_size + i * cols + j;

                svfloat64_t center = svld1_f64(pg, &grid[base_idx]);

                svfloat64_t k1_i0_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + j]);
                svfloat64_t kp1_i0_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + j]);
                svfloat64_t k0_i1_j0 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + j]);
                svfloat64_t k0_ip1_j0 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + j]);
                svfloat64_t k0_i0_j1 = svld1_f64(pg, &grid[k * plane_size + i * cols + (j-1)]);
                svfloat64_t k0_i0_jp1 = svld1_f64(pg, &grid[k * plane_size + i * cols + (j+1)]);

                svfloat64_t k1_i1_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + j]);
                svfloat64_t k1_ip1_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + j]);
                svfloat64_t kp1_i1_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + j]);
                svfloat64_t kp1_ip1_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + j]);

                svfloat64_t k1_i0_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + (j-1)]);
                svfloat64_t kp1_i0_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + (j+1)]);

                svzero_za();

                svmopa_za64_f64_m(0, pg_all, pg, ones, center);

                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i0_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i0_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i0_jp1);

                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i0_jp1);

                svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
                svfloat64_t result = svmul_f64_z(pg, sum, weight_vec);
                svst1_f64(pg, &new_grid[base_idx], result);
            }
        }
    }
}

__arm_new("za")
void stencil3D_25point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                           int depth, int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t weight_vec = svdup_f64(1.0 / 25.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    #pragma omp parallel for
    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += SVL * stride) {
                int j_limit = (cols - 1 < j + SVL * stride - 1) ? cols - 1 : j + SVL * stride - 1;
                svbool_t pg = svwhilelt_b64(j, j_limit + 1);
                if (!svptest_any(svptrue_b64(), pg)) break;

                int base_idx = k * plane_size + i * cols + j;

                svfloat64_t center = svld1_f64(pg, &grid[base_idx]);

                svfloat64_t k1_i0_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + j]);
                svfloat64_t kp1_i0_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + j]);
                svfloat64_t k0_i1_j0 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + j]);
                svfloat64_t k0_ip1_j0 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + j]);
                svfloat64_t k0_i0_j1 = svld1_f64(pg, &grid[k * plane_size + i * cols + (j-1)]);
                svfloat64_t k0_i0_jp1 = svld1_f64(pg, &grid[k * plane_size + i * cols + (j+1)]);

                svfloat64_t k1_i1_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + j]);
                svfloat64_t k1_ip1_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + j]);
                svfloat64_t kp1_i1_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + j]);
                svfloat64_t kp1_ip1_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + j]);
                svfloat64_t k1_i0_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + (j-1)]);
                svfloat64_t k1_i0_jp1 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + (j+1)]);
                svfloat64_t kp1_i0_j1 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + (j-1)]);
                svfloat64_t kp1_i0_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + (j+1)]);
                svfloat64_t k0_i1_j1 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + (j-1)]);
                svfloat64_t k0_i1_jp1 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + (j+1)]);
                svfloat64_t k0_ip1_j1 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + (j-1)]);
                svfloat64_t k0_ip1_jp1 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + (j+1)]);

                svfloat64_t k1_i1_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + (j-1)]);
                svfloat64_t k1_i1_jp1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + (j+1)]);
                svfloat64_t k1_ip1_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + (j-1)]);
                svfloat64_t k1_ip1_jp1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + (j+1)]);
                svfloat64_t kp1_i1_j1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + (j-1)]);
                svfloat64_t kp1_i1_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + (j+1)]);
                svfloat64_t kp1_ip1_j1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + (j-1)]);
                svfloat64_t kp1_ip1_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + (j+1)]);

                svzero_za();

                svmopa_za64_f64_m(0, pg_all, pg, ones, center);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i0_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i0_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i0_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i0_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i0_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_i1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_ip1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k0_ip1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_i1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_ip1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, k1_ip1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_i1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_ip1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, kp1_ip1_jp1);

                svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
                svfloat64_t result = svmul_f64_z(pg, sum, weight_vec);
                svst1_f64(pg, &new_grid[base_idx], result);
            }
        }
    }
}

__arm_new("za")
void stencil3D_27point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                           int depth, int rows, int cols, int stride)
    __arm_streaming {

    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t weight_vec = svdup_f64(1.0 / 27.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

    #pragma omp parallel for
    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += SVL * stride) {
                int j_limit = (cols - 1 < j + SVL * stride - 1) ? cols - 1 : j + SVL * stride - 1;
                svbool_t pg = svwhilelt_b64(j, j_limit + 1);
                if (!svptest_any(svptrue_b64(), pg)) break;

                int base_idx = k * plane_size + i * cols + j;

                svfloat64_t m_k1_i1_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + (j-1)]);
                svfloat64_t m_k1_i1_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + j]);
                svfloat64_t m_k1_i1_jp1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i-1) * cols + (j+1)]);

                svfloat64_t m_k1_i0_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + (j-1)]);
                svfloat64_t m_k1_i0_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + j]);
                svfloat64_t m_k1_i0_jp1 = svld1_f64(pg, &grid[(k-1) * plane_size + i * cols + (j+1)]);

                svfloat64_t m_k1_ip1_j1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + (j-1)]);
                svfloat64_t m_k1_ip1_j0 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + j]);
                svfloat64_t m_k1_ip1_jp1 = svld1_f64(pg, &grid[(k-1) * plane_size + (i+1) * cols + (j+1)]);

                svfloat64_t m_k0_i1_j1 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + (j-1)]);
                svfloat64_t m_k0_i1_j0 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + j]);
                svfloat64_t m_k0_i1_jp1 = svld1_f64(pg, &grid[k * plane_size + (i-1) * cols + (j+1)]);

                svfloat64_t m_k0_i0_j1 = svld1_f64(pg, &grid[k * plane_size + i * cols + (j-1)]);
                svfloat64_t center = svld1_f64(pg, &grid[k * plane_size + i * cols + j]);
                svfloat64_t m_k0_i0_jp1 = svld1_f64(pg, &grid[k * plane_size + i * cols + (j+1)]);

                svfloat64_t m_k0_ip1_j1 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + (j-1)]);
                svfloat64_t m_k0_ip1_j0 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + j]);
                svfloat64_t m_k0_ip1_jp1 = svld1_f64(pg, &grid[k * plane_size + (i+1) * cols + (j+1)]);

                svfloat64_t m_kp1_i1_j1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + (j-1)]);
                svfloat64_t m_kp1_i1_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + j]);
                svfloat64_t m_kp1_i1_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i-1) * cols + (j+1)]);

                svfloat64_t m_kp1_i0_j1 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + (j-1)]);
                svfloat64_t m_kp1_i0_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + j]);
                svfloat64_t m_kp1_i0_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + i * cols + (j+1)]);

                svfloat64_t m_kp1_ip1_j1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + (j-1)]);
                svfloat64_t m_kp1_ip1_j0 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + j]);
                svfloat64_t m_kp1_ip1_jp1 = svld1_f64(pg, &grid[(k+1) * plane_size + (i+1) * cols + (j+1)]);

                svzero_za();

                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_i1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_i1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_i0_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_i0_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_ip1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k1_ip1_jp1);

                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_i1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_i1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, center);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_i0_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_ip1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_k0_ip1_jp1);

                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_i1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_i1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_i1_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_i0_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_i0_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_i0_jp1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_ip1_j1);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_ip1_j0);
                svmopa_za64_f64_m(0, pg_all, pg, ones, m_kp1_ip1_jp1);

                svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);

                svfloat64_t result = svmul_f64_z(pg, sum, weight_vec);
                svst1_f64(pg, &new_grid[base_idx], result);
            }
        }
    }
}

int main() {
    std::cout << "=== All Stencil SME Tests (OpenMP) ===" << std::endl;

    double total_time = 0.0;

    std::cout << std::endl << "--- 1D 3-point ---" << std::endl;
    {
        const int SIZE = 1048576;
        double* g1 = (double*)aligned_alloc(64, SIZE * sizeof(double));
        double* g2 = (double*)aligned_alloc(64, SIZE * sizeof(double));
        for (int i = 0; i < SIZE; i++) g1[i] = 1.0 + i;

        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil1D_3point_sme(g1, g2, SIZE, 1);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        for (int i = 0; i < SIZE; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil1D_3point_sme(g1, g2, SIZE, 2);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        free(g1); free(g2);
    }

    std::cout << std::endl << "--- 2D 5-point ---" << std::endl;
    {
        const int ROWS = 1024, COLS = 1024;
        double* g1 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
        double* g2 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                g1[i * COLS + j] = 1.0 + i * COLS + j;

        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_5point_sme(g1, g2, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                g1[i * COLS + j] = 1.0 + i * COLS + j;
        std::cout << "stride=2..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_5point_sme(g1, g2, ROWS, COLS, 2);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        free(g1); free(g2);
    }

    std::cout << std::endl << "--- 2D 9-point ---" << std::endl;
    {
        const int ROWS = 1024, COLS = 1024;
        double* g1 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
        double* g2 = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                g1[i * COLS + j] = 1.0 + i * COLS + j;

        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_9point_sme(g1, g2, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        for (int i = 0; i < ROWS; i++)
            for (int j = 0; j < COLS; j++)
                g1[i * COLS + j] = 1.0 + i * COLS + j;
        std::cout << "stride=2..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_9point_sme(g1, g2, ROWS, COLS, 2);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        free(g1); free(g2);
    }

    std::cout << std::endl << "--- 3D 13-point ---" << std::endl;
    {
        const int DEPTH = 128, ROWS = 512, COLS = 512;
        double* g1 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
        double* g2 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
        for (int k = 0; k < DEPTH; k++)
            for (int i = 0; i < ROWS; i++)
                for (int j = 0; j < COLS; j++)
                    g1[k * ROWS * COLS + i * COLS + j] = 1.0 + (k * ROWS + i) * COLS + j;

        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_13point_sme(g1, g2, DEPTH, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        for (int k = 0; k < DEPTH; k++)
            for (int i = 0; i < ROWS; i++)
                for (int j = 0; j < COLS; j++)
                    g1[k * ROWS * COLS + i * COLS + j] = 1.0 + (k * ROWS + i) * COLS + j;
        std::cout << "stride=2..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_13point_sme(g1, g2, DEPTH, ROWS, COLS, 2);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        free(g1); free(g2);
    }

    std::cout << std::endl << "--- 3D 25-point ---" << std::endl;
    {
        const int DEPTH = 128, ROWS = 512, COLS = 512;
        double* g1 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
        double* g2 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
        for (int k = 0; k < DEPTH; k++)
            for (int i = 0; i < ROWS; i++)
                for (int j = 0; j < COLS; j++)
                    g1[k * ROWS * COLS + i * COLS + j] = 1.0 + (k * ROWS + i) * COLS + j;

        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_25point_sme(g1, g2, DEPTH, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        for (int k = 0; k < DEPTH; k++)
            for (int i = 0; i < ROWS; i++)
                for (int j = 0; j < COLS; j++)
                    g1[k * ROWS * COLS + i * COLS + j] = 1.0 + (k * ROWS + i) * COLS + j;
        std::cout << "stride=2..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_25point_sme(g1, g2, DEPTH, ROWS, COLS, 2);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        free(g1); free(g2);
    }

    std::cout << std::endl << "--- 3D 27-point ---" << std::endl;
    {
        const int DEPTH = 128, ROWS = 512, COLS = 512;
        double* g1 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
        double* g2 = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
        for (int k = 0; k < DEPTH; k++)
            for (int i = 0; i < ROWS; i++)
                for (int j = 0; j < COLS; j++)
                    g1[k * ROWS * COLS + i * COLS + j] = 1.0 + (k * ROWS + i) * COLS + j;

        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_27point_sme(g1, g2, DEPTH, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        for (int k = 0; k < DEPTH; k++)
            for (int i = 0; i < ROWS; i++)
                for (int j = 0; j < COLS; j++)
                    g1[k * ROWS * COLS + i * COLS + j] = 1.0 + (k * ROWS + i) * COLS + j;
        std::cout << "stride=2..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_27point_sme(g1, g2, DEPTH, ROWS, COLS, 2);
        end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;

        free(g1); free(g2);
    }

    std::cout << std::endl << "=== All Tests Completed ===" << std::endl;
    std::cout << "Total Time: " << total_time << " s" << std::endl;
    return 0;
}
