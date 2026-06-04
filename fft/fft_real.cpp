// FFTW Real FFT Test Case
// Uses fftw_malloc and fftw_plan_dft / fftw_execute_dft

#include <fftw.h>
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
    printf("Testing 1D Real FFT with N = %d\n", N);

    // Use fftw_malloc for aligned memory
    double* signal = (double*)fftw_alloc_real(N);
    double* reconstructed = (double*)fftw_alloc_real(N);
    fftw_complex* in = (fftw_complex*)fftw_alloc_complex(N);
    fftw_complex* out = (fftw_complex*)fftw_alloc_complex(N);

    if (!signal || !reconstructed || !in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    create_test_signal(signal, N);

    // Convert real to complex (zero imaginary part)
    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Use fftw_plan_dft for forward FFT
    fftw_plan forward_plan = fftw_plan_dft(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    // Use fftw_plan_dft for inverse FFT
    fftw_plan inverse_plan = fftw_plan_dft(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute_dft(forward_plan, in, out);

    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5; i++) {
        double magnitude = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        printf("  Bin %d: %.6f\n", i, magnitude);
    }

    fftw_execute_dft(inverse_plan, out, in);

    // Extract real part and normalize
    for (int i = 0; i < N; i++) {
        reconstructed[i] = in[i][0] / N;
    }

    bool success = verify_real_inverse_fft(signal, reconstructed, N);

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(signal);
    fftw_free(reconstructed);
    fftw_free(in);
    fftw_free(out);

    printf("Real FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int test_fft_real_2d() {
    const int N0 = 64;
    const int N1 = 128;
    const int N = N0 * N1;
    printf("Testing 2D Real FFT with size %d x %d\n", N0, N1);

    // Use fftw_malloc for aligned memory
    double* signal = (double*)fftw_alloc_real(N);
    double* reconstructed = (double*)fftw_alloc_real(N);
    fftw_complex* in = (fftw_complex*)fftw_alloc_complex(N);
    fftw_complex* out = (fftw_complex*)fftw_alloc_complex(N);

    if (!signal || !reconstructed || !in || !out) {
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

    // Convert real to complex
    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Use fftw_plan_dft for forward FFT
    fftw_plan forward_plan = fftw_plan_dft(2, (int[]){N0, N1}, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    // Use fftw_plan_dft for inverse FFT
    fftw_plan inverse_plan = fftw_plan_dft(2, (int[]){N0, N1}, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute_dft(forward_plan, in, out);
    fftw_execute_dft(inverse_plan, out, in);

    // Extract real part and normalize
    for (int i = 0; i < N; i++) {
        reconstructed[i] = in[i][0] / N;
    }

    // Verify
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(reconstructed[i] - signal[i]);
        if (error > max_error) max_error = error;
    }
    printf("Max 2D real inverse FFT error: %.2e\n", max_error);
    bool success = max_error < 1e-10;

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(signal);
    fftw_free(reconstructed);
    fftw_free(in);
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
