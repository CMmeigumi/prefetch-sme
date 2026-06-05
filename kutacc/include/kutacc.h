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
#ifndef KUTACC_H
#define KUTACC_H

#include <vector>
#include <cstdint>
#include <functional>
#include <arm_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief exposed KuTACC symbol table*/
#define kutacc_export           __attribute__((visibility("default")))

/** @brief status in KuTACC library*/
#define KUTACC_OK 0
#define KUTACC_ERROR (-1)

/** @brief KuTACC version info*/
typedef struct kutacc_version {
    const char *product_name;
    const char *product_version;
    const char *component_name;
    const char *component_version;
    const char *component_appendinfo;
} kutacc_version_t;

/*
 * @brief get the kutacc version info
 * @param [out] version     the kutacc version info
 *
 * @return KUTACC_OK for get version info success, other for failed.
 */
kutacc_export int kutacc_get_version(kutacc_version_t *version);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace kutacc {
/*! \enum DType
 *  \brief datatype of tensor.
 */
typedef enum TensorDataType {
    kInt64 = 0,
    kBF16 = 1,
    kNumTypes
} DType;

/** @brief the parameters of FlashMLA forward function */
struct FlashMLAFwdParams {
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int *cu_seqlens_k;

    void *q_ptr;
    void *k_ptr;
    void *v_ptr;
    void *o_ptr;
    void *softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;

    int *block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int *tile_scheduler_metadata_ptr;
    int num_thread_parts;
    int *num_splits_ptr;

    void *softmax_lseaccum_ptr;
    void *oaccum_ptr;

    void *tiling_buffer_ptr = nullptr;
};

constexpr int64_t FUSEDMOE_TILEBUF = 64;
using MatrixTilingBlock = std::tuple<int64_t, int64_t, int64_t>;

}   // namespace kutacc
#endif

#endif
