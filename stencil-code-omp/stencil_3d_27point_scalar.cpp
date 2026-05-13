// 3D 27-point Stencil 标量实现

#include "stencil_3d_27point.h"

void stencil3D_27point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 27.0;

    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += stride) {
                double sum = 0.0;
                int base_idx = k * plane_size + i * cols + j;

                for (int dk = -1; dk <= 1; dk++) {
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            sum += grid[(k + dk) * plane_size + (i + di) * cols + (j + dj)];
                        }
                    }
                }

                new_grid[base_idx] = sum * weight;
            }
        }
    }
}
