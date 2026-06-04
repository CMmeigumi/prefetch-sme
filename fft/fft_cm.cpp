#include "fftw.h"

void test_c2c()
{
    int rank = 2;
    int *n;
    n = (int*)fftw_malloc(sizeof(int) * rank);
    n[0] = 2;
    n[1] = 3;
    double init[6][2] = {{120,0},{8,8},{0,0},{0,16},{0,16},{-8,8}};
    FFTW_COMPLEX *in;
    in = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * n[0] * n[1]);
    for (int i = 0; i < n[0] * n[1]; i++) {
        in[i].re = init[i][0];
        in[i].im = init[i][1];
    }
    FFTW_COMPLEX *out;
    out = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * n[0] * n[1]);
    fftw_plan plan;
    plan = fftw_plan_dft(rank, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute_dft(forward_plan, in, out);

    fftw_destoryt_plan(plan);
    fftw_free(n);
    fftw_free(in);
    fftw_free(out);
}