/*
 * Test program for mla_bf16_32x64 kernel
 *
 * This code is designed for ARM SVE/SME hardware and requires a proper
 * cross-compilation toolchain for ARM architecture.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "kutacc.h"
#include "attention/flash_mla/common.h"
#include "attention/flash_mla/flash_fwd_mla_kernel/kernel_traits.h"
#include "attention/flash_mla/flash_fwd_mla_kernel/kernels/mla_bf16_32x64.h"

using Element = bfloat16_t;
using ElementAccum = float;
using index_t = int64_t;

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
    printf("FlashMLA BF16 32x64 Kernel Test\n");
    printf("================================\n\n");

    // Create a minimal params structure
    kutacc::FlashMLAFwdParams params = {};
    params.b = 1;
    params.seqlen_q = 128;
    params.d = 576;
    params.d_v = 512;
    params.h = 1;
    params.ngroups = 1;
    params.is_causal = true;
    params.scale_softmax = 0.08838834764831845f;
    params.scale_softmax_log2 = 0.12748366579437256f;

    // Allocate buffers
    const int seqlen_q = params.seqlen_q;
    const int seqlen_k = 128;
    const int batch_size = params.b;

    const size_t q_size = batch_size * seqlen_q * 576 * sizeof(bfloat16_t);
    const size_t k_size = batch_size * seqlen_k * 576 * sizeof(bfloat16_t);
    const size_t o_size = batch_size * seqlen_q * 512 * sizeof(bfloat16_t);
    const size_t lse_size = batch_size * params.h * seqlen_q * sizeof(float);
    const size_t block_table_size = batch_size * 2 * sizeof(int);

    params.q_ptr = malloc(q_size);
    params.k_ptr = malloc(k_size);
    params.v_ptr = params.k_ptr;  // Shared KV
    params.o_ptr = malloc(o_size);
    params.softmax_lse_ptr = malloc(lse_size);

    // Initialize with random data
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

    // Block table for paged attention
    params.block_table = (int *)malloc(block_table_size);
    for (int i = 0; i < batch_size * 2; i++) {
        params.block_table[i] = 0;  // Each block points to block 0
    }
    params.block_table_batch_stride = 2;
    params.page_block_size = 64;  // kBlockN

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

    printf("Test parameters:\n");
    printf("  Batch size: %d\n", params.b);
    printf("  Seq len Q: %d\n", params.seqlen_q);
    printf("  Seq len K: %d\n", seqlen_k);
    printf("  Head dim: %d\n", params.d);
    printf("  Head dim V: %d\n", params.d_v);
    printf("  Num heads: %d\n", params.h);
    printf("  Is causal: %s\n", params.is_causal ? "true" : "false");
    printf("\n");

    // Allocate tiling buffer
    const size_t tiling_buffer_size = 256 * 1024;
    char *tiling_buffer = (char *)malloc(tiling_buffer_size);

    if (tiling_buffer) {
        printf("Calling compute_attn_1rowblock_bf16_32x64...\n\n");

        // Calculate block indices
        const int num_m_block = (params.seqlen_q + 31) / 32;  // kBlockM = 32
        const int num_n_block = (seqlen_k + 63) / 64;          // kBlockN = 64

        // Iterate over blocks
        for (int bidb = 0; bidb < params.b; bidb++) {
            for (int bidh = 0; bidh < params.h; bidh++) {
                for (int m_block = 0; m_block < num_m_block; m_block++) {
                    // Call the kernel for causal case
                    kutacc::compute_attn_1rowblock_bf16_32x64<TestKernelTraits, true>(
                        params,
                        bidb,              // bidb: batch index
                        bidh,              // bidh: head index
                        m_block,           // m_block: M block index
                        0,                 // n_split_idx
                        seqlen_k,          // seqlen_k
                        0,                 // n_block_min
                        num_n_block,       // n_block_max
                        true,              // NoSplit
                        tiling_buffer      // tiling buffer pointer
                    );
                }
            }
        }

        printf("Kernel execution completed.\n");
        free(tiling_buffer);
    } else {
        printf("Failed to allocate tiling buffer\n");
    }

    // Cleanup
    if (params.q_ptr) free(params.q_ptr);
    if (params.k_ptr) free(params.k_ptr);
    if (params.o_ptr) free(params.o_ptr);
    if (params.softmax_lse_ptr) free(params.softmax_lse_ptr);
    if (params.block_table) free(params.block_table);
    if (params.num_splits_ptr) free(params.num_splits_ptr);
    if (params.cu_seqlens_k) free(params.cu_seqlens_k);

    printf("\nTest completed.\n");
    return 0;
}
