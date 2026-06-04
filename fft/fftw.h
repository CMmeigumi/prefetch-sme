/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * The authors hereby grant permission to use, copy, modify, distribute,
 * and license this software and its documentation for any purpose, provided
 * that existing copyright notices are retained in all copies and that this
 * notice is included verbatim in all distributions. No written agreement,
 * license, or royalty fee is required.  Author accepts no liability for
 * incidental or consequential damages arising from use of this software.
 */

#ifndef FFTW_H
#define FFTW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* type definitions */
typedef double fftw_real;
typedef struct { fftw_real re, im; } fftw_complex;

typedef enum { FFTW_FORWARD = -1, FFTW_BACKWARD = 1 } fftw_direction;

typedef enum {
     FFTW_ESTIMATE = 0,
     FFTW_MEASURE = 1,
     FFTW_PATIENT = 2,
     FFTW_EXHAUSTIVE = 3,
     FFTW_WISDOM_ONLY = 4
} fftw_planner;

typedef struct fftw_plan_s *fftw_plan;

typedef void (*fftw_iodim)(int, int, int);

/* memory allocation */
void *fftw_alloc_real(size_t n);
void *fftw_alloc_complex(size_t n);
void fftw_free(void *p);

/* planning */
fftw_plan fftw_plan_dft(int rank, const int *n, 
                        fftw_complex *in, fftw_complex *out, 
                        int sign, unsigned flags);
fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,
                           int sign, unsigned flags);
fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex *in, fftw_complex *out,
                           int sign, unsigned flags);
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2,
                           fftw_complex *in, fftw_complex *out,
                           int sign, unsigned flags);
fftw_plan fftw_plan_dft_r2c(int rank, const int *n,
                            double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out,
                               unsigned flags);
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out,
                               unsigned flags);
fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2,
                               double *in, fftw_complex *out,
                               unsigned flags);
fftw_plan fftw_plan_dft_c2r(int rank, const int *n,
                            fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex *in, double *out,
                               unsigned flags);
fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex *in, double *out,
                               unsigned flags);
fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2,
                               fftw_complex *in, double *out,
                               unsigned flags);
fftw_plan fftw_plan_r2r(int rank, const int *n,
                        double *in, double *out,
                        const fftw_iodim *iodims, 
                        const fftw_iodim *iodim_end,
                        int sign, unsigned flags);

/* execution */
void fftw_execute(const fftw_plan plan);
void fftw_execute_dft(const fftw_plan plan, fftw_complex *in, fftw_complex *out);
void fftw_execute_dft_r2c(const fftw_plan plan, double *in, fftw_complex *out);
void fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex *in, double *out);
void fftw_execute_r2r(const fftw_plan plan, double *in, double *out);

/* plan destruction */
void fftw_destroy_plan(fftw_plan plan);
void fftw_forget_wisdom(void);
void fftw_cleanup(void);

/* wisdom export/import */
void fftw_export_wisdom(void (*write_char)(char, void *), void *data);
int fftw_import_system_wisdom(void);
int fftw_import_wisdom(void (*read_char)(void *), void *data);

/* misc */
void fftw_set_timelimit(double seconds);

#ifdef __cplusplus
}
#endif

#endif /* FFTW_H */
