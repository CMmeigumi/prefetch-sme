// 2D 9-point Stencil 标量版本实现
// 形状：3×3 正方形 (中心及8个邻点)
// 应用：2D波动方程、图像处理

#include "stencil_2d_9point.h"

void stencil2D_9point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
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
