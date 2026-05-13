#ifndef STENCIL_3D_27POINT_H
#define STENCIL_3D_27POINT_H

#include <cmath>

#ifdef __ARM_FEATURE_SME
#include <arm_sme.h>
#include <arm_sve.h>
#endif

void stencil3D_27point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride);

#ifdef __ARM_FEATURE_SME
__arm_new("za") void stencil3D_27point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                           int depth, int rows, int cols, int stride) __arm_streaming;
#endif

inline void initializeGrid3D(double* grid, int depth, int rows, int cols) {
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = k * rows * cols + i * cols + j;
                grid[idx] = 1.0 + (k * rows + i) * cols + j;
            }
        }
    }
}

inline double computeAverage3D(double* grid, int depth, int rows, int cols) {
    double sum = 0.0;
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += grid[k * rows * cols + i * cols + j];
            }
        }
    }
    return sum / (depth * rows * cols);
}

#endif
