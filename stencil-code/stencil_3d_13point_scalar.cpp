// 3D 13-point Stencil 标量版本实现
// 形状：包含中心、6个面心、6个边心 (共13点)
// 应用：特定有限差分格式、热传导

#include "stencil_3d_13point.h"

void stencil3D_13point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 13.0;

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
                sum += grid[(k+1) * plane_size + i * cols + (j+1)];

                new_grid[base_idx] = sum * weight;
            }
        }
    }
}

void initializeGrid3D(double* grid, int depth, int rows, int cols) {
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (k == 0 || k == depth - 1 || i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                    grid[k * rows * cols + i * cols + j] = 0.0;
                } else if (k >= depth/3 && k <= 2*depth/3 &&
                          i >= rows/3 && i <= 2*rows/3 &&
                          j >= cols/3 && j <= 2*cols/3) {
                    grid[k * rows * cols + i * cols + j] = 100.0;
                } else {
                    grid[k * rows * cols + i * cols + j] = 0.0;
                }
            }
        }
    }
}

double computeAverage3D(double* grid, int depth, int rows, int cols) {
    double sum = 0.0;
    int total = depth * rows * cols;
    for (int i = 0; i < total; i++) {
        sum += grid[i];
    }
    return sum / total;
}
