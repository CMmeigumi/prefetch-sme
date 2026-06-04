// FFTW 2D FFT Test Case
// Uses Quantum Espresso fftw.h interface

#include "fftw.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_2d_test_signal(FFTW_REAL* signal, int N0, int N1) {
    for (int i = 0; i < N0; i++) {
        for (int j = 0; j < N1; j++) {
            double x = (double)i / N0;
            double y = (double)j / N1;
            signal[i * N1 + j] = sin(2 * PI * 3 * x) * cos(2 * PI * 5 * y);
        }
    }
}

bool verify_2d_inverse_fft(FFTW_COMPLEX* in, FFTW_COMPLEX* out, int N0, int N1) {
    int N = N0 * N1;
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(c_re(out[i]) / N - c_re(in[i]));
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
    FFTW_COMPLEX* in = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* out = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* inverse = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create signal and put into complex input
    FFTW_REAL* signal = (FFTW_REAL*)malloc(sizeof(FFTW_REAL) * N);
    create_2d_test_signal(signal, N0, N1);

    for (int i = 0; i < N; i++) {
        c_re(in[i]) = signal[i];
        c_im(in[i]) = 0.0;
    }

    // Use fftwnd_create_plan
    int n[2];
    n[0] = N0;
    n[1] = N1;
    fftwnd_plan forward_plan = fftwnd_create_plan(2, n, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwnd_plan inverse_plan = fftwnd_create_plan(2, n, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Use fftwnd() function
    fftwnd(forward_plan, 1, in, 1, 0, out, 1, 0);
    fftwnd(inverse_plan, 1, out, 1, 0, inverse, 1, 0);

    bool success = verify_2d_inverse_fft(in, inverse, N0, N1);

    printf("Sample frequency domain values (magnitude):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i * N1 + j;
            double magnitude = sqrt(c_re(out[idx]) * c_re(out[idx]) + c_im(out[idx]) * c_im(out[idx]));
            printf("  [%d,%d]: %.6f\n", i, j, magnitude);
        }
    }

    fftwnd_destroy_plan(forward_plan);
    fftwnd_destroy_plan(inverse_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);
    free(signal);

    printf("2D FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW 2D FFT Test ===\n\n");
    return test_fft_2d();
}
