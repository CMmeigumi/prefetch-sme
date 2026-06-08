// FFTW 2D FFT Test Case
// Uses standard fftw3.h interface

#include "fftw3.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_2d_test_signal(double* signal, int N0, int N1) {
    for (int i = 0; i < N0; i++) {
        for (int j = 0; j < N1; j++) {
            double x = (double)i / N0;
            double y = (double)j / N1;
            signal[i * N1 + j] = sin(2 * PI * 3 * x) * cos(2 * PI * 5 * y);
        }
    }
}

bool verify_2d_inverse_fft(fftw_complex* in, fftw_complex* out, int N0, int N1) {
    int N = N0 * N1;
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(out[i][0] / N - in[i][0]);
        if (error > max_error) max_error = error;
    }
    printf("Max 2D inverse FFT error: %.2e\n", max_error);
    return max_error < 1e-10;
}

int test_fft_2d() {
    const int N0 = 64;
    const int N1 = 128;
    const int N = N0 * N1;
    printf("Testing 2D FFT with size %d x %d\n", N0, N1);

    // Use fftw_malloc for aligned memory
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* inverse = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create signal and put into complex input
    double* signal = (double*)malloc(sizeof(double) * N);
    create_2d_test_signal(signal, N0, N1);

    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Use fftw_plan_dft_2d
    fftw_plan forward_plan = fftw_plan_dft_2d(N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(N0, N1, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Use fftw_execute
    fftw_execute(forward_plan);
    fftw_execute(inverse_plan);

    bool success = verify_2d_inverse_fft(in, inverse, N0, N1);

    printf("Sample frequency domain values (magnitude):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i * N1 + j;
            double magnitude = sqrt(out[idx][0] * out[idx][0] + out[idx][1] * out[idx][1]);
            printf("  [%d,%d]: %.6f\n", i, j, magnitude);
        }
    }

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);
    free(signal);

    printf("2D FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW 2D FFT Test (fftw3.h) ===\n\n");
    return test_fft_2d();
}