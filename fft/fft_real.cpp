// FFTW Real FFT Test Case
// Tests forward (real-to-complex) and inverse (complex-to-real) FFT operations
// Real FFT is more memory efficient for real-valued signals

#include <fftw3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_test_signal(double* signal, int N) {
    // Create a test signal: sum of two sine waves
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
    printf("Testing 1D Real FFT with N = %d\n", N);

    // For real FFT, output size is N/2+1 complex numbers
    const int N_complex = N / 2 + 1;

    // Allocate arrays
    double* signal = (double*)fftw_alloc_real(N);
    double* reconstructed = (double*)fftw_alloc_real(N);
    fftw_complex* out = (fftw_complex*)fftw_alloc_complex(N_complex);

    if (!signal || !reconstructed || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create test signal
    create_test_signal(signal, N);

    // Create FFTW plans
    // Real-to-complex FFT: input is real, output is complex
    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(N, signal, out, FFTW_ESTIMATE);
    // Complex-to-real FFT: input is complex, output is real
    fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(N, out, reconstructed, FFTW_ESTIMATE);

    // Execute forward FFT (real to complex)
    fftw_execute(forward_plan);

    // Print some frequency domain values
    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5; i++) {
        double magnitude = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        printf("  Bin %d: %.6f\n", i, magnitude);
    }

    // Execute inverse FFT (complex to real)
    fftw_execute(inverse_plan);

    // Normalize the inverse result (FFTW does not normalize)
    for (int i = 0; i < N; i++) {
        reconstructed[i] /= N;
    }

    // Verify result
    bool success = verify_real_inverse_fft(signal, reconstructed, N);

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(signal);
    fftw_free(reconstructed);
    fftw_free(out);

    printf("Real FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int test_fft_real_2d() {
    const int N0 = 64;
    const int N1 = 128;
    printf("Testing 2D Real FFT with size %d x %d\n", N0, N1);

    // For 2D real FFT, output size is N0 x (N1/2+1) complex numbers
    const int N1_complex = N1 / 2 + 1;

    // Allocate arrays
    double* signal = (double*)fftw_alloc_real(N0 * N1);
    double* reconstructed = (double*)fftw_alloc_real(N0 * N1);
    fftw_complex* out = (fftw_complex*)fftw_alloc_complex(N0 * N1_complex);

    if (!signal || !reconstructed || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create test signal
    for (int i = 0; i < N0; i++) {
        for (int j = 0; j < N1; j++) {
            double x = (double)i / N0;
            double y = (double)j / N1;
            signal[i * N1 + j] = sin(2 * PI * 3 * x) * cos(2 * PI * 5 * y);
        }
    }

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(N0, N1, signal, out, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_c2r_2d(N0, N1, out, reconstructed, FFTW_ESTIMATE);

    // Execute forward FFT
    fftw_execute(forward_plan);

    // Execute inverse FFT
    fftw_execute(inverse_plan);

    // Normalize
    int N = N0 * N1;
    for (int i = 0; i < N; i++) {
        reconstructed[i] /= N;
    }

    // Verify
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(reconstructed[i] - signal[i]);
        if (error > max_error) max_error = error;
    }
    printf("Max 2D real inverse FFT error: %.2e\n", max_error);
    bool success = max_error < 1e-10;

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(signal);
    fftw_free(reconstructed);
    fftw_free(out);

    printf("2D Real FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW Real FFT Test ===\n\n");

    int result = 0;
    result += test_fft_real_1d();
    result += test_fft_real_2d();

    printf("=== All Real FFT tests %s ===\n\n", (result == 0) ? "PASSED" : "FAILED");
    return result;
}
