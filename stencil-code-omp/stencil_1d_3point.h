#ifndef STENCIL_1D_3POINT_OMP_H
#define STENCIL_1D_3POINT_OMP_H

#include <cmath>

#ifdef __ARM_FEATURE_SME
#include <arm_sme.h>
#include <arm_sve.h>
#endif

void stencil1D_3point_omp(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride);

#ifdef __ARM_FEATURE_SME
__arm_new("za")
void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride);
#endif

inline void initializeGrid1D(double* grid, int size) {
    for (int i = 0; i < size; i++) {
        grid[i] = 1.0 + i;
    }
}

inline double computeAverage1D(double* grid, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += grid[i];
    }
    return sum / size;
}

#endif
