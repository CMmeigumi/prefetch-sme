// 1D 3-point Stencil 标量实现

#include "stencil_1d_3point.h"
#include <omp.h>

void stencil1D_3point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride) {
    double weight = 1.0 / 3.0;

    #pragma omp parallel for
    for (int i = 1; i < size - 1; i += stride) {
        double sum = grid[i-1] + grid[i] + grid[i+1];
        new_grid[i] = sum * weight;
    }
}
