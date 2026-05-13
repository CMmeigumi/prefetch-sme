// 3D 25-point Stencil 标量版本实现
// 形状：沿坐标轴方向向外延伸，但角点被部分裁剪
// 应用：四阶或八阶精度地震成像（如Iso3DFD）

#include "stencil_3d_25point.h"

void stencil3D_25point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 25.0;

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
