// FFTW Real Signal FFT Test Case
// Uses Quantum Espresso fftw.h interface
// Note: This version uses complex FFT to simulate real signal processing

#include "fftw.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_test_signal(FFTW_REAL* signal, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        signal[i] = sin(2 * PI * 3 * t) + 0.5 * sin(2 * PI * 7 * t) + 0.25;
    }
}

bool verify_real_inverse_fft(FFTW_REAL* original, FFTW_REAL* reconstructed, int N) {
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(reconstructed[i] - original[i]);
        if (error > max_error) max_error = error;
    }
    printf("Max real inverse FFT error: %.2e\n", max_error);
    return max_error < 1e-10;
}

int test_fft_real_1d() {
    const int N = 1024;
    printf("Testing 1D Real Signal FFT with N = %d\n", N);

    // Use fftw_malloc for aligned memory
    FFTW_COMPLEX* in = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* out = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* inverse = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create real signal
    FFTW_REAL* signal = (FFTW_REAL*)malloc(sizeof(FFTW_REAL) * N);
    create_test_signal(signal, N);

    // Put real signal into complex input (imaginary part = 0)
    for (int i = 0; i < N; i++) {
        c_re(in[i]) = signal[i];
        c_im(in[i]) = 0.0;
    }

    // Use fftw_create_plan
    fftw_plan forward_plan = fftw_create_plan(N, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_create_plan(N, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Forward FFT
    fftw(forward_plan, 1, in, 1, 0, out, 1, 0);

    // Inverse FFT
    fftw(inverse_plan, 1, out, 1, 0, inverse, 1, 0);

    // Extract real part and normalize
    FFTW_REAL* reconstructed = (FFTW_REAL*)malloc(sizeof(FFTW_REAL) * N);
    for (int i = 0; i < N; i++) {
        reconstructed[i] = c_re(inverse[i]) / N;
    }

    bool success = verify_real_inverse_fft(signal, reconstructed, N);

    // Print first 5 frequency bins
    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5; i++) {
        double magnitude = sqrt(c_re(out[i]) * c_re(out[i]) + c_im(out[i]) * c_im(out[i]));
        printf("  Bin %d: %.6f\n", i, magnitude);
    }

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);
    free(signal);
    free(reconstructed);

    printf("Real Signal FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int test_fft_real_2d() {
    const int N0 = 64;
    const int N1 = 128;
    const int N = N0 * N1;
    printf("Testing 2D Real Signal FFT with size %d x %d\n", N0, N1);

    // Use fftw_malloc for aligned memory
    FFTW_COMPLEX* in = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* out = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);
    FFTW_COMPLEX* inverse = (FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create 2D real signal
    FFTW_REAL* signal = (FFTW_REAL*)malloc(sizeof(FFTW_REAL) * N);
    for (int i = 0; i < N0; i++) {
        for (int j = 0; j < N1; j++) {
            double x = (double)i / N0;
            double y = (double)j / N1;
            signal[i * N1 + j] = sin(2 * PI * 3 * x) * cos(2 * PI * 5 * y);
        }
    }

    // Put real signal into complex input
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

    // Forward FFT
    fftwnd(forward_plan, 1, in, 1, 0, out, 1, 0);

    // Inverse FFT
    fftwnd(inverse_plan, 1, out, 1, 0, inverse, 1, 0);

    // Extract real part and normalize
    FFTW_REAL* reconstructed = (FFTW_REAL*)malloc(sizeof(FFTW_REAL) * N);
    for (int i = 0; i < N; i++) {
        reconstructed[i] = c_re(inverse[i]) / N;
    }

    // Verify
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(reconstructed[i] - signal[i]);
        if (error > max_error) max_error = error;
    }
    printf("Max 2D real inverse FFT error: %.2e\n", max_error);
    bool success = max_error < 1e-10;

    fftwnd_destroy_plan(forward_plan);
    fftwnd_destroy_plan(inverse_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);
    free(signal);
    free(reconstructed);

    printf("2D Real Signal FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW Real Signal FFT Test ===\n\n");

    int result = 0;
    result += test_fft_real_1d();
    result += test_fft_real_2d();

    printf("=== All Real Signal FFT tests %s ===\n\n", (result == 0) ? "PASSED" : "FAILED");
    return result;
}
