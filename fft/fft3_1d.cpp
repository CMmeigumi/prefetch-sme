// FFTW 1D FFT Test Case
// Uses standard fftw3.h interface

#include "fftw3.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_test_signal(double* signal, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        signal[i] = sin(2 * PI * 3 * t) + 0.5 * sin(2 * PI * 7 * t);
    }
}

bool verify_inverse_fft(fftw_complex* in, fftw_complex* out, int N) {
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(out[i][0] / N - in[i][0]);
        if (error > max_error) max_error = error;
    }
    printf("Max inverse FFT error: %.2e\n", max_error);
    return max_error < 1e-10;
}

int test_fft_1d() {
    const int N = 1024;
    printf("Testing 1D FFT with N = %d\n", N);

    // Use fftw_malloc for aligned memory allocation
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* inverse = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create signal and put into complex input
    double* signal = (double*)malloc(sizeof(double) * N);
    create_test_signal(signal, N);

    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Use fftw_plan_dft_1d
    fftw_plan forward_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_1d(N, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Use fftw_execute
    fftw_execute(forward_plan);
    fftw_execute(inverse_plan);

    bool success = verify_inverse_fft(in, inverse, N);

    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5 && i < N/2; i++) {
        double magnitude = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        printf("  Bin %d: %.6f\n", i, magnitude);
    }

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);
    free(signal);

    printf("1D FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW 1D FFT Test (fftw3.h) ===\n\n");
    return test_fft_1d();
}