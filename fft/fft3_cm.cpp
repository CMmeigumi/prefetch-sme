// FFTW Complex Matrix FFT Test Case
// Uses standard fftw3.h interface

#include "fftw3.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int test_c2c() {
    int n[2];
    n[0] = 2;
    n[1] = 3;
    double init[6][2] = {{120,0},{8,8},{0,0},{0,16},{0,16},{-8,8}};

    printf("Testing C2C FFT with size %d x %d\n", n[0], n[1]);

    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
    for (int i = 0; i < n[0] * n[1]; i++) {
        in[i][0] = init[i][0];
        in[i][1] = init[i][1];
    }

    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);

    // Use fftw_plan_dft
    fftw_plan plan = fftw_plan_dft(2, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Use fftw_execute_dft
    fftw_execute_dft(plan, in, out);

    printf("Input complex values:\n");
    for (int i = 0; i < n[0] * n[1]; i++) {
        printf("  [%d]: %.1f + %.1fi\n", i, in[i][0], in[i][1]);
    }

    printf("Output complex values:\n");
    for (int i = 0; i < n[0] * n[1]; i++) {
        printf("  [%d]: %.6f + %.6fi\n", i, out[i][0], out[i][1]);
    }

    // Verify by inverse FFT
    fftw_complex* inverse = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n[0] * n[1]);
    fftw_plan inverse_plan = fftw_plan_dft(2, n, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute_dft(inverse_plan, out, inverse);

    printf("Inverse FFT (normalized) - should match input:\n");
    int N = n[0] * n[1];
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double re_err = fabs(inverse[i][0] / N - init[i][0]);
        double im_err = fabs(inverse[i][1] / N - init[i][1]);
        double error = sqrt(re_err * re_err + im_err * im_err);
        if (error > max_error) max_error = error;
        printf("  [%d]: %.6f + %.6fi\n", i, inverse[i][0] / N, inverse[i][1] / N);
    }

    printf("Max reconstruction error: %.2e\n", max_error);
    bool success = max_error < 1e-10;

    fftw_destroy_plan(plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);

    printf("C2C FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW Complex Matrix FFT Test (fftw3.h) ===\n\n");
    return test_c2c();
}