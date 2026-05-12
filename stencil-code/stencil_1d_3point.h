#ifndef STENCIL_1D_3POINT_H
#define STENCIL_1D_3POINT_H

#include <cmath>

#ifdef __ARM_FEATURE_SME
#include <arm_sme.h>
#include <arm_sve.h>
#endif

void stencil1D_3point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int size, int stride);

#ifdef __ARM_FEATURE_SME
__arm_new("za") void stencil1D_3point_sme(double* __restrict__ grid, double* __restrict__ new_grid,
                            int size, int stride) __arm_streaming;
#endif

void initializeGrid1D(double* grid, int size);
double computeAverage1D(double* grid, int size);

#endif
