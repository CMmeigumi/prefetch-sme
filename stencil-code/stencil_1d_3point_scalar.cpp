// 1D 3-point Stencil 标量版本实现
// 形状：线性（左、中、右）
// 应用：一维对流/扩散问题

#include "stencil_1d_3point.h"

void stencil1D_3point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int size, int stride) {
    double weight = 1.0 / 3.0;

    for (int i = 1; i < size - 1; i += stride) {
        double sum = grid[i-1] + grid[i] + grid[i+1];
        new_grid[i] = sum * weight;
    }
}

void initializeGrid1D(double* grid, int size) {
    for (int i = 0; i < size; i++) {
        if (i == 0 || i == size - 1) {
            grid[i] = 0.0;
        } else if (i >= size/3 && i <= 2*size/3) {
            grid[i] = 100.0;
        } else {
            grid[i] = 0.0;
        }
    }
}

double computeAverage1D(double* grid, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += grid[i];
    }
    return sum / size;
}
