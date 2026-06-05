/*
 * Test program for mla_bf16_32x64 kernel (no prefetch version)
 *
 * Designed for ARM SVE/SME hardware with cross-compilation toolchain.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "kutacc.h"
#include "attention/flash_mla/common.h"
#include "attention/flash_mla/flash_fwd_mla_kernel/kernel_traits.h"
#include "attention/flash_mla/flash_fwd_mla_kernel/kernels/mla_bf16_32x64.h"

struct TestKernelTraits {
    static constexpr int kBlockM = 32;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 576;
    static constexpr int kHeadDimV = 512;
    using Element = bfloat16_t;
    using ElementAccum = float;
    using index_t = int64_t;
};

int main(int argc, char **argv)
{
    // Test parameters
    const int batch_size = 1;
    const int seqlen_q = 128;
    const int seqlen_k = 128;
    const int num_heads = 1;
    const int repeat_count = 100000;  // High count for perf sampling

    // Initialize params
    kutacc::FlashMLAFwdParams params = {};
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.d = 576;
    params.d_v = 512;
    params.h = num_heads;
    params.ngroups = 1;
    params.is_causal = true;
    params.scale_softmax = 0.08838834764831845f;
    params.scale_softmax_log2 = 0.12748366579437256f;

    // Allocate buffers
    const size_t q_size = batch_size * seqlen_q * 576 * sizeof(bfloat16_t);
    const size_t k_size = batch_size * seqlen_k * 576 * sizeof(bfloat16_t);
    const size_t o_size = batch_size * seqlen_q * 512 * sizeof(bfloat16_t);
    const size_t lse_size = batch_size * num_heads * seqlen_q * sizeof(float);
    const size_t block_table_size = batch_size * 2 * sizeof(int);

    params.q_ptr = malloc(q_size);
    params.k_ptr = malloc(k_size);
    params.v_ptr = params.k_ptr;
    params.o_ptr = malloc(o_size);
    params.softmax_lse_ptr = malloc(lse_size);

    // Initialize Q/K with random data
    if (params.q_ptr) {
        bfloat16_t *q = (bfloat16_t *)params.q_ptr;
        for (size_t i = 0; i < q_size / sizeof(bfloat16_t); i++) {
            q[i] = (bfloat16_t)(rand() / (float)RAND_MAX * 2.0f - 1.0f);
        }
    }
    if (params.k_ptr) {
        bfloat16_t *k = (bfloat16_t *)params.k_ptr;
        for (size_t i = 0; i < k_size / sizeof(bfloat16_t); i++) {
            k[i] = (bfloat16_t)(rand() / (float)RAND_MAX * 2.0f - 1.0f);
        }
    }
    if (params.o_ptr) memset(params.o_ptr, 0, o_size);
    if (params.softmax_lse_ptr) memset(params.softmax_lse_ptr, 0, lse_size);

    // Set strides
    params.q_batch_stride = seqlen_q * 576;
    params.k_batch_stride = seqlen_k * 576;
    params.v_batch_stride = seqlen_k * 512;
    params.o_batch_stride = seqlen_q * 512;
    params.q_row_stride = 576;
    params.k_row_stride = 576;
    params.v_row_stride = 512;
    params.o_row_stride = 512;
    params.q_head_stride = 0;
    params.k_head_stride = 0;
    params.v_head_stride = 0;
    params.o_head_stride = 0;

    // Block table
    params.block_table = (int *)malloc(block_table_size);
    for (int i = 0; i < batch_size * 2; i++) {
        params.block_table[i] = 0;
    }
    params.block_table_batch_stride = 2;
    params.page_block_size = 64;

    // Tile scheduler metadata
    int tile_scheduler_metadata[8] = {0, 0, 0, seqlen_k, 0, 0, 0, 0};
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata;
    params.num_thread_parts = 1;

    // Num splits
    params.num_splits_ptr = (int *)malloc((batch_size + 1) * sizeof(int));
    params.num_splits_ptr[0] = 0;
    params.num_splits_ptr[1] = 1;

    // cu_seqlens_k
    params.cu_seqlens_k = (int *)malloc((batch_size + 1) * sizeof(int));
    params.cu_seqlens_k[0] = 0;
    params.cu_seqlens_k[1] = seqlen_k;

    // Allocate tiling buffer
    const size_t tiling_buffer_size = 256 * 1024;
    char *tiling_buffer = (char *)malloc(tiling_buffer_size);

    volatile int execution_count = 0;

    if (tiling_buffer) {
        printf("FlashMLA Kernel Test\n");
        printf("Batch=%d, SeqQ=%d, SeqK=%d, Heads=%d, Repeat=%d\n",
               batch_size, seqlen_q, seqlen_k, num_heads, repeat_count);

        const int num_m_block = (seqlen_q + 31) / 32;
        const int num_n_block = (seqlen_k + 63) / 64;

        // Main loop
        for (int repeat = 0; repeat < repeat_count; repeat++) {
            for (int bidb = 0; bidb < batch_size; bidb++) {
                for (int bidh = 0; bidh < num_heads; bidh++) {
                    for (int m_block = 0; m_block < num_m_block; m_block++) {
                        kutacc::compute_attn_1rowblock_bf16_32x64<TestKernelTraits, true>(
                            params,
                            bidb, bidh, m_block,
                            0, seqlen_k,
                            0, num_n_block,
                            true,
                            tiling_buffer
                        );
                        execution_count++;
                    }
                }
            }
        }

        printf("Kernel executed %d times\n", execution_count);
        free(tiling_buffer);
    } else {
        printf("Failed to allocate tiling buffer\n");
    }

    return 0;
}
