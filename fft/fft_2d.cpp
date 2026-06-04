// FFTW 2D FFT Test Case
// Tests forward and inverse 2D FFT operations

#include <fftw3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_2d_test_signal(double* signal, int N0, int N1) {
    // Create a 2D test signal: product of two sine waves
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
        double error = fabs(out[i][0] - in[i][0]);
        if (error > max_error) max_error = error;
    }
    printf("Max 2D inverse FFT error: %.2e\n", max_error);
    return max_error < 1e-10;
}

int test_fft_2d() {
    const int N0 = 64;  // rows
    const int N1 = 128;  // columns
    printf("Testing 2D FFT with size %d x %d\n", N0, N1);

    // Allocate arrays
    double* signal = (double*)fftw_alloc_real(N0 * N1);
    fftw_complex* in = (fftw_complex*)fftw_alloc_complex(N0 * N1);
    fftw_complex* out = (fftw_complex*)fftw_alloc_complex(N0 * N1);
    fftw_complex* inverse = (fftw_complex*)fftw_alloc_complex(N0 * N1);

    if (!signal || !in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create test signal
    create_2d_test_signal(signal, N0, N1);

    // Copy to complex input
    for (int i = 0; i < N0 * N1; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_2d(N0, N1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_2d(N0, N1, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute forward FFT
    fftw_execute(forward_plan);

    // Execute inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse result
    int N = N0 * N1;
    for (int i = 0; i < N; i++) {
        inverse[i][0] /= N;
        inverse[i][1] /= N;
    }

    // Verify result
    bool success = verify_2d_inverse_fft(in, inverse, N0, N1);

    // Print some frequency domain values
    printf("Sample frequency domain values (magnitude):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int idx = i * N1 + j;
            double magnitude = sqrt(out[idx][0] * out[idx][0] + out[idx][1] * out[idx][1]);
            printf("  [%d,%d]: %.6f\n", i, j, magnitude);
        }
    }

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(signal);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);

    printf("2D FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW 2D FFT Test ===\n\n");
    return test_fft_2d();
}
