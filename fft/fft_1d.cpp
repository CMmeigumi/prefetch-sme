// FFTW 1D FFT Test Case
// Tests forward and inverse 1D FFT operations

#include <fftw3.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const double PI = 3.14159265358979323846;

void create_test_signal(double* signal, int N) {
    // Create a simple test signal: sum of two sine waves
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        signal[i] = sin(2 * PI * 3 * t) + 0.5 * sin(2 * PI * 7 * t);
    }
}

bool verify_inverse_fft(fftw_complex* in, fftw_complex* out, int N) {
    // After inverse FFT, the result should match the original (scaled by N)
    // Allow small numerical error
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(out[i][0] - in[i][0]);
        if (error > max_error) max_error = error;
    }
    printf("Max inverse FFT error: %.2e\n", max_error);
    return max_error < 1e-10;
}

int test_fft_1d() {
    const int N = 1024;
    printf("Testing 1D FFT with N = %d\n", N);

    // Allocate arrays
    double* signal = (double*)fftw_alloc_real(N);
    fftw_complex* in = (fftw_complex*)fftw_alloc_complex(N);
    fftw_complex* out = (fftw_complex*)fftw_alloc_complex(N);
    fftw_complex* inverse = (fftw_complex*)fftw_alloc_complex(N);

    if (!signal || !in || !out || !inverse) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Create test signal
    create_test_signal(signal, N);

    // Copy to complex input
    for (int i = 0; i < N; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan inverse_plan = fftw_plan_dft_1d(N, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Execute forward FFT
    fftw_execute(forward_plan);

    // Execute inverse FFT
    fftw_execute(inverse_plan);

    // Normalize the inverse result
    for (int i = 0; i < N; i++) {
        inverse[i][0] /= N;
        inverse[i][1] /= N;
    }

    // Verify result
    bool success = verify_inverse_fft(in, inverse, N);

    // Print some frequency domain values
    printf("First 5 frequency bins (magnitude):\n");
    for (int i = 0; i < 5 && i < N/2; i++) {
        double magnitude = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        printf("  Bin %d: %.6f\n", i, magnitude);
    }

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(signal);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);

    printf("1D FFT test %s\n\n", success ? "PASSED" : "FAILED");
    return success ? 0 : 1;
}

int main() {
    printf("=== FFTW 1D FFT Test ===\n\n");
    return test_fft_1d();
}
