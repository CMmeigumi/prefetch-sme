// 2D 9-point Stencil 标量实现

#include "stencil_2d_9point.h"

void stencil2D_9point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int rows, int cols, int stride) {
    double weight = 1.0 / 9.0;

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += stride) {
            double sum = 0.0;
            int base_idx = i * cols + j;

            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    sum += grid[(i + di) * cols + (j + dj)];
                }
            }

            new_grid[base_idx] = sum * weight;
        }
    }
}
