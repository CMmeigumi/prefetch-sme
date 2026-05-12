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
void __arm_new("za") stencil3D_27point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                           int depth, int rows, int cols, int stride) __arm_streaming;
#endif

void initializeGrid3D(double* grid, int depth, int rows, int cols);
double computeAverage3D(double* grid, int depth, int rows, int cols);

#endif
