// FFTW Real Signal FFT Test Case
// Uses standard fftw3.h interface
// Note: This version uses complex FFT to simulate real signal processing

#include "fftw3.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_test_signal(double* signal, int N) {
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        signal[i] = sin(2 * PI * 3 * t) + 0.5 * sin(2 * PI * 7 * t) + 0.25;
    }
}

bool verify_real_inverse_fft(double* original, double* reconstructed, int N) {
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
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* inverse = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create real signal
    double* signal = (double*)malloc(sizeof(double) * N);
    create_test_signal(signal, N);

    // Put real signal into complex input (imaginary part = 0)
    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Use fftw_plan_dft_1d
    fftw_plan forward_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_1d(N, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Forward FFT
    fftw_execute(forward_plan);

    // Inverse FFT
    fftw_execute(inverse_plan);

    // Extract real part and normalize
    double* reconstructed = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        reconstructed[i] = inverse[i][0] / N;
    }

    bool success = verify_real_inverse_fft(signal, reconstructed, N);

    // Print first 5 frequency bins
    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5; i++) {
        double magnitude = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
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
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* inverse = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);

    if (!in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create 2D real signal
    double* signal = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N0; i++) {
        for (int j = 0; j < N1; j++) {
            double x = (double)i / N0;
            double y = (double)j / N1;
            signal[i * N1 + j] = sin(2 * PI * 3 * x) + 0.5 * cos(2 * PI * 5 * y);
        }
    }

    // Put real signal into complex input
    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Use fftw_plan_dft_2d
    fftw_plan forward_plan = fftw_plan_dft_2d(N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(N0, N1, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Forward FFT
    fftw_execute(forward_plan);

    // Inverse FFT
    fftw_execute(inverse_plan);

    // Extract real part and normalize
    double* reconstructed = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        reconstructed[i] = inverse[i][0] / N;
    }

    // Verify
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(reconstructed[i] - signal[i]);
        if (error > max_error) max_error = error;
    }
    printf("Max 2D real inverse FFT error: %.2e\n", max_error);
    bool success = max_error < 1e-10;

    printf("Sample frequency bins (magnitude):\n");
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
    free(reconstructed);

    printf("2D Real Signal FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW Real Signal FFT Test (fftw3.h) ===\n\n");
    int result = 0;
    result |= test_fft_real_1d();
    result |= test_fft_real_2d();
    return result;
}