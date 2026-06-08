/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "kutacc_matmul.h"
#include "bgemm_kernel_pf.h"

using bfloat16_t = __bf16;

int main(int argc, char **argv)
{
    // Default parameters - based on bgemm_kernel from kutacc-br_dist_infer
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;
    int repeat_count = 1000;

    // Parse command line
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [OPTIONS]\n\n", argv[0]);
            printf("Options:\n");
            printf("  -h, --help            Show this help message\n");
            printf("  -r, --repeat N        Set repeat count (default: 1000)\n");
            printf("\nExamples:\n");
            printf("  %s              # Run with default (1000 repeats)\n", argv[0]);
            printf("  %s -r 100       # Run with 100 repeats\n", argv[0]);
            printf("\nMatrix dimensions: M=%d, N=%d, K=%d\n", m, n, k);
            printf("Kernel: bgemm_kernel_pf (4VL_1VL, 1VL_1VL with prefetch)\n");
            return 0;
        } else if ((strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--repeat") == 0) && i + 1 < argc) {
            repeat_count = atoi(argv[++i]);
        } else {
            repeat_count = atoi(argv[i]);
        }
    }

    if (repeat_count <= 0) {
        repeat_count = 100000;
    }

    printf("BGEMM Kernel Test (With Prefetch, from kutacc-br_dist_infer)\n");
    printf("M=%d, N=%d, K=%d, Repeat=%d\n", m, n, k, repeat_count);

    // Allocate matrices (all bf16)
    // A: bf16, shape (m, k)
    // B: bf16, shape (k, n)
    // C: bf16, shape (m, n)
    bfloat16_t *A = (bfloat16_t *)malloc(m * k * sizeof(bfloat16_t));
    bfloat16_t *B = (bfloat16_t *)malloc(k * n * sizeof(bfloat16_t));
    bfloat16_t *C = (bfloat16_t *)malloc(m * n * sizeof(bfloat16_t));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize with random data
    for (int i = 0; i < m * k; i++) {
        float val = (float)(rand() % 65536 - 32768) / 256.0f;
        A[i] = (__bf16)val;
    }
    for (int i = 0; i < k * n; i++) {
        float val = (float)(rand() % 65536 - 32768) / 256.0f;
        B[i] = (__bf16)val;
    }
    for (int i = 0; i < m * n; i++) {
        C[i] = (__bf16)0.0f;
    }

    // Call kernel repeatedly
    for (int repeat = 0; repeat < repeat_count; repeat++) {
        // Reset output
        for (int i = 0; i < m * n; i++) {
            C[i] = (__bf16)0.0f;
        }

        // Call the BGEMM kernel (alpha=1, beta=0)
        kutacc::bgemm_kernel_pf(m, n, n, k, A, B, C);
    }
    printf("\nDone. Total kernel calls: %d\n", repeat_count);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}