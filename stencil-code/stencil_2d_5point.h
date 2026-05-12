#ifndef STENCIL_2D_5POINT_H
#define STENCIL_2D_5POINT_H

#include <cmath>

#ifdef __ARM_FEATURE_SME
#include <arm_sme.h>
#include <arm_sve.h>
#endif

void stencil2D_5point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int rows, int cols, int stride);

#ifdef __ARM_FEATURE_SME
void __arm_new("za") stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int rows, int cols, int stride) __arm_streaming;
#endif

void initializeGrid2D(double* grid, int rows, int cols);
double computeAverage2D(double* grid, int rows, int cols);

#endif
