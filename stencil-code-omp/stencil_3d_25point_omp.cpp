// 3D 25-point Stencil OpenMP 实现

#include "stencil_3d_25point.h"
#include <omp.h>

void stencil3D_25point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 25.0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += stride) {
                double sum = 0.0;
                int base_idx = k * plane_size + i * cols + j;

                sum += grid[base_idx];

                sum += grid[(k-1) * plane_size + i * cols + j];
                sum += grid[(k+1) * plane_size + i * cols + j];
                sum += grid[k * plane_size + (i-1) * cols + j];
                sum += grid[k * plane_size + (i+1) * cols + j];
                sum += grid[k * plane_size + i * cols + (j-1)];
                sum += grid[k * plane_size + i * cols + (j+1)];

                sum += grid[(k-1) * plane_size + (i-1) * cols + j];
                sum += grid[(k-1) * plane_size + (i+1) * cols + j];
                sum += grid[(k+1) * plane_size + (i-1) * cols + j];
                sum += grid[(k+1) * plane_size + (i+1) * cols + j];
                sum += grid[(k-1) * plane_size + i * cols + (j-1)];
                sum += grid[(k-1) * plane_size + i * cols + (j+1)];
                sum += grid[(k+1) * plane_size + i * cols + (j-1)];
                sum += grid[(k+1) * plane_size + i * cols + (j+1)];
                sum += grid[k * plane_size + (i-1) * cols + (j-1)];
                sum += grid[k * plane_size + (i-1) * cols + (j+1)];
                sum += grid[k * plane_size + (i+1) * cols + (j-1)];
                sum += grid[k * plane_size + (i+1) * cols + (j+1)];

                sum += grid[(k-1) * plane_size + (i-1) * cols + (j-1)];
                sum += grid[(k-1) * plane_size + (i-1) * cols + (j+1)];
                sum += grid[(k-1) * plane_size + (i+1) * cols + (j-1)];
                sum += grid[(k-1) * plane_size + (i+1) * cols + (j+1)];
                sum += grid[(k+1) * plane_size + (i-1) * cols + (j-1)];
                sum += grid[(k+1) * plane_size + (i-1) * cols + (j+1)];
                sum += grid[(k+1) * plane_size + (i+1) * cols + (j-1)];
                sum += grid[(k+1) * plane_size + (i+1) * cols + (j+1)];

                new_grid[base_idx] = sum * weight;
            }
        }
    }
}

#ifdef __ARM_FEATURE_SME
void stencil3D_25point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int depth, int rows, int cols, int stride)
    __arm_new("za")
    __arm_streaming {

    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t weight_vec = svdup_f64(1.0 / 25.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();

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
#endif
