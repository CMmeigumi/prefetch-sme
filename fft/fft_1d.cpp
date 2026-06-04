// FFTW 1D FFT Test Case
// Uses Quantum Espresso fftw.h interface

#include "fftw.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_test_signal(FFTW_REAL* signal, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        signal[i] = sin(2 * PI * 3 * t) + 0.5 * sin(2 * PI * 7 * t);
    }
}

bool verify_inverse_fft(FFTW_COMPLEX* in, FFTW_COMPLEX* out, int N) {
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(c_re(out[i]) / N - c_re(in[i]));
        if (error > max_error) max_error = error;
    }
    printf("Max inverse FFT error: %.2e\n", max_error);
    return max_error < 1e-10;
}

int test_fft_1d() {
    const int N = 1024;
    printf("Testing 1D FFT with N = %d\n", N);

    // Use fftw_malloc for aligned memory allocation
    FFTW_COMPLEX* in = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* out = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* inverse = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create signal and put into complex input
    FFTW_REAL* signal = (FFTW_REAL*)malloc(sizeof(FFTW_REAL) * N);
    create_test_signal(signal, N);

    for (int i = 0; i < N; i++) {
        c_re(in[i]) = signal[i];
        c_im(in[i]) = 0.0;
    }

    // Use fftw_create_plan
    fftw_plan forward_plan = fftw_create_plan(N, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_create_plan(N, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Use fftw() function
    fftw(forward_plan, 1, in, 1, 0, out, 1, 0);
    fftw(inverse_plan, 1, out, 1, 0, inverse, 1, 0);

    bool success = verify_inverse_fft(in, inverse, N);

    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5 && i < N/2; i++) {
        double magnitude = sqrt(c_re(out[i]) * c_re(out[i]) + c_im(out[i]) * c_im(out[i]));
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
    printf("=== FFTW 1D FFT Test ===\n\n");
    return test_fft_1d();
}
