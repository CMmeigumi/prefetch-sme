// 2D 5-point Stencil 标量实现

#include "stencil_2d_5point.h"

void stencil2D_5point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int rows, int cols, int stride) {
    double weight = 1.0 / 5.0;

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += stride) {
            double sum = 0.0;
            int base_idx = i * cols + j;

            sum += grid[base_idx];
            sum += grid[base_idx - cols];
            sum += grid[base_idx + cols];
            sum += grid[base_idx - 1];
            sum += grid[base_idx + 1];

            new_grid[base_idx] = sum * weight;
        }
    }
}
