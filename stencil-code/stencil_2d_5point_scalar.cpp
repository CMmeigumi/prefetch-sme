// 2D 5-point Stencil 标量版本实现
// 形状：十字形（上下左右加中心）
// 应用：2D泊松方程、简单扩散

#include "stencil_2d_5point.h"

void stencil2D_5point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int rows, int cols, int stride) {
    double weight = 1.0 / 5.0;

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += stride) {
            double sum = 0.0;
            int base_idx = i * cols + j;

            sum += grid[base_idx];
            sum += grid[(i-1) * cols + j];
            sum += grid[(i+1) * cols + j];
            sum += grid[i * cols + (j-1)];
            sum += grid[i * cols + (j+1)];

            new_grid[base_idx] = sum * weight;
        }
    }
}

void initializeGrid2D(double* grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                grid[i * cols + j] = 0.0;
            } else if (i >= rows/3 && i <= 2*rows/3 && j >= cols/3 && j <= 2*cols/3) {
                grid[i * cols + j] = 100.0;
            } else {
                grid[i * cols + j] = 0.0;
            }
        }
    }
}

double computeAverage2D(double* grid, int rows, int cols) {
    double sum = 0.0;
    int total = rows * cols;
    for (int i = 0; i < total; i++) {
        sum += grid[i];
    }
    return sum / total;
}
