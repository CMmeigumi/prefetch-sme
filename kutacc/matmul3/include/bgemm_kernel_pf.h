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
#pragma once

#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>

namespace kutacc {

__attribute__((always_inline)) inline void bgemm_4VL_1VL(const bfloat16_t *a0, const bfloat16_t *b, int k, int k_num,
    int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_16_m = svwhilelt_b16(0, 32);
    svbool_t pg_16_n = svwhilelt_b16(0, n_num);
    const bfloat16_t *a1 = a0 + k * 16, *a2 = a1 + k * 16, *a3 = a2 + k * 16;
    for (int ki = 0; ki < (k_num / 2); ki++) {
        svbfloat16_t m_data0 = svld1(pg_16_m, a0 + ki * 32);
        svbfloat16_t m_data1 = svld1(pg_16_m, a1 + ki * 32);
        svbfloat16_t m_data2 = svld1(pg_16_m, a2 + ki * 32);
        svbfloat16_t m_data3 = svld1(pg_16_m, a3 + ki * 32);
        svbfloat16_t n_data = svld1(pg_16_n, b + ki * n_num);
        svmopa_za32_bf16_m(0, pg_16_m, pg_16_n, m_data0, n_data);
        svmopa_za32_bf16_m(1, pg_16_m, pg_16_n, m_data1, n_data);
        svmopa_za32_bf16_m(2, pg_16_m, pg_16_n, m_data2, n_data);
        svmopa_za32_bf16_m(3, pg_16_m, pg_16_n, m_data3, n_data);
    }
}

__attribute__((always_inline)) inline void bgemm_1VL_1VL(const bfloat16_t *a, const bfloat16_t *b, int k, int k_num,
    int m_num, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_16_m = svwhilelt_b16(0, m_num);
    svbool_t pg_16_n = svwhilelt_b16(0, n_num);
    for (int ki = 0; ki < (k_num / 2); ki++) {
        svbfloat16_t m_data = svld1(pg_16_m, a + ki * m_num);
        svbfloat16_t n_data = svld1(pg_16_n, b + ki * n_num);
        svmopa_za32_bf16_m(0, pg_16_m, pg_16_n, m_data, n_data);
    }
}

__attribute__((always_inline)) inline void bgemm41_save(int ldc, bfloat16_t *c, int n_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 2);
    svbool_t pg32_all = svptrue_b32();
    svfloat32_t zero_f32 = svdup_f32(0);
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int i = 0; i < 16; i++) {
        svfloat32_t out_v0_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 0, i);
        svfloat32_t out_v1_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 1, i);
        svfloat32_t out_v2_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 2, i);
        svfloat32_t out_v3_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 3, i);

        svst1(pg_c_16, c + (i + 00) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero_bf16));
        svst1(pg_c_16, c + (i + 16) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v1_f), zero_bf16));
        svst1(pg_c_16, c + (i + 32) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v2_f), zero_bf16));
        svst1(pg_c_16, c + (i + 48) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v3_f), zero_bf16));
    }
}

__attribute__((always_inline)) inline void bgemm11_save(int ldc, bfloat16_t *c, int m_num, int n_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 2);
    svbool_t pg32_all = svptrue_b32();
    svfloat32_t zero_f32 = svdup_f32(0);
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int i = 0; i < m_num / 2; i++) {
        svfloat32_t out_v0_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 0, i);
        svst1(pg_c_16, c + i * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero_bf16));
    }
}

__attribute__((always_inline)) inline void bgemm_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset,
    int k_num, int ldc, bfloat16_t *a, bfloat16_t *b, bfloat16_t *c) __arm_inout("za") __arm_streaming
{
    for (int mi = bm; mi < em;) {
        if (mi + 64 <= em) {
            int ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_4VL_1VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 2, k, k_num, n_num);
                bgemm41_save(ldc, c + mi * ldc + ni, n_num);
            }
            mi += 64;
        } else {
            int m_num = std::min(32, (em - mi) * 2), ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_1VL_1VL(a + mi * k + k_offset * m_num / 2, b + ni * k + k_offset * n_num / 2, k, k_num, m_num,
                    n_num);
                bgemm11_save(ldc, c + mi * ldc + ni, m_num, n_num);
            }
            mi += 16;
        }
    }
}

__attribute__((always_inline)) inline void bgemm_4VL_1VL_pf(const bfloat16_t *a0, const bfloat16_t *b, int k, int k_num,
    int n_num, int prefetch_dis) __arm_inout("za") __arm_streaming
{
    svbool_t pg_16_m = svwhilelt_b16(0, 32);
    svbool_t pg_16_n = svwhilelt_b16(0, n_num);
    const bfloat16_t *a1 = a0 + k * 16, *a2 = a1 + k * 16, *a3 = a2 + k * 16;
    for (int ki = 0; ki < (k_num / 2); ki++) {
        if (ki + prefetch_dis < (k_num / 2)) {
            __builtin_prefetch(a0 + (ki + prefetch_dis) * 32, 0, 0);
            __builtin_prefetch(a1 + (ki + prefetch_dis) * 32, 0, 0);
            __builtin_prefetch(a2 + (ki + prefetch_dis) * 32, 0, 0);
            __builtin_prefetch(a3 + (ki + prefetch_dis) * 32, 0, 0);
            __builtin_prefetch(b + (ki + prefetch_dis) * n_num, 0, 0);
        }
        svbfloat16_t m_data0 = svld1(pg_16_m, a0 + ki * 32);
        svbfloat16_t m_data1 = svld1(pg_16_m, a1 + ki * 32);
        svbfloat16_t m_data2 = svld1(pg_16_m, a2 + ki * 32);
        svbfloat16_t m_data3 = svld1(pg_16_m, a3 + ki * 32);
        svbfloat16_t n_data = svld1(pg_16_n, b + ki * n_num);
        svmopa_za32_bf16_m(0, pg_16_m, pg_16_n, m_data0, n_data);
        svmopa_za32_bf16_m(1, pg_16_m, pg_16_n, m_data1, n_data);
        svmopa_za32_bf16_m(2, pg_16_m, pg_16_n, m_data2, n_data);
        svmopa_za32_bf16_m(3, pg_16_m, pg_16_n, m_data3, n_data);
    }
}

__attribute__((always_inline)) inline void bgemm_1VL_1VL_pf(const bfloat16_t *a, const bfloat16_t *b, int k, int k_num,
    int m_num, int n_num, int prefetch_dis) __arm_inout("za") __arm_streaming
{
    svbool_t pg_16_m = svwhilelt_b16(0, m_num);
    svbool_t pg_16_n = svwhilelt_b16(0, n_num);
    for (int ki = 0; ki < (k_num / 2); ki++) {
        if (ki + prefetch_dis < (k_num / 2)) {
            __builtin_prefetch(a + (ki + prefetch_dis) * m_num, 0, 0);
            __builtin_prefetch(b + (ki + prefetch_dis) * n_num, 0, 0);
        }
        svbfloat16_t m_data = svld1(pg_16_m, a + ki * m_num);
        svbfloat16_t n_data = svld1(pg_16_n, b + ki * n_num);
        svmopa_za32_bf16_m(0, pg_16_m, pg_16_n, m_data, n_data);
    }
}

__attribute__((always_inline)) inline void bgemm_macro_kernel_pf(int bm, int em, int bn, int en, int k, int k_offset,
    int k_num, int ldc, bfloat16_t *a, bfloat16_t *b, bfloat16_t *c) __arm_inout("za") __arm_streaming
{
    const int prefetch_dis = 4;
    for (int mi = bm; mi < em;) {
        if (mi + 64 <= em) {
            int ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_4VL_1VL_pf(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 2, k, k_num, n_num,
                    prefetch_dis);
                bgemm41_save(ldc, c + mi * ldc + ni, n_num);
            }
            mi += 64;
        } else {
            int m_num = std::min(32, (em - mi) * 2), ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_1VL_1VL_pf(a + mi * k + k_offset * m_num / 2, b + ni * k + k_offset * n_num / 2, k, k_num, m_num,
                    n_num, prefetch_dis);
                bgemm11_save(ldc, c + mi * ldc + ni, m_num, n_num);
            }
            mi += 16;
        }
    }
}

__attribute__((always_inline)) inline void bgemm41_save_beta(int ldc, bfloat16_t *c, int n_num, float beta)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 2);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int i = 0; i < 16; i++) {
        svfloat32_t out_v0_f = svread_hor_za32_f32_m(svfloat32_t(), pg32_all, 0, i);
        svfloat32_t out_v1_f = svread_hor_za32_f32_m(svfloat32_t(), pg32_all, 1, i);
        svfloat32_t out_v2_f = svread_hor_za32_f32_m(svfloat32_t(), pg32_all, 2, i);
        svfloat32_t out_v3_f = svread_hor_za32_f32_m(svfloat32_t(), pg32_all, 3, i);

        svbfloat16_t c_v0 = svld1(pg_c_16, c + (i + 00) * ldc);
        svbfloat16_t c_v1 = svld1(pg_c_16, c + (i + 16) * ldc);
        svbfloat16_t c_v2 = svld1(pg_c_16, c + (i + 32) * ldc);
        svbfloat16_t c_v3 = svld1(pg_c_16, c + (i + 48) * ldc);

        auto c_v00 = svmul_x(pg32_all, svreinterpret_f32(svzip1(zero_bf16, c_v0)), beta);
        auto c_v01 = svmul_x(pg32_all, svreinterpret_f32(svzip2(zero_bf16, c_v0)), beta);
        c_v00 = svadd_x(pg32_all, c_v00, out_v0_f);
        c_v01 = svadd_x(pg32_all, c_v01, out_v1_f);
        svst1(pg_c_16, c + (i + 00) * ldc, svuzp1(svcvt_bf16_x(pg32_all, c_v00), svcvt_bf16_x(pg32_all, c_v01)));

        auto c_v10 = svmul_x(pg32_all, svreinterpret_f32(svzip1(zero_bf16, c_v1)), beta);
        auto c_v11 = svmul_x(pg32_all, svreinterpret_f32(svzip2(zero_bf16, c_v1)), beta);
        c_v10 = svadd_x(pg32_all, c_v10, out_v2_f);
        c_v11 = svadd_x(pg32_all, c_v11, out_v3_f);
        svst1(pg_c_16, c + (i + 16) * ldc, svuzp1(svcvt_bf16_x(pg32_all, c_v10), svcvt_bf16_x(pg32_all, c_v11)));

        auto c_v20 = svmul_x(pg32_all, svreinterpret_f32(svzip1(zero_bf16, c_v2)), beta);
        auto c_v21 = svmul_x(pg32_all, svreinterpret_f32(svzip2(zero_bf16, c_v2)), beta);
        c_v20 = svadd_x(pg32_all, c_v20, out_v2_f);
        c_v21 = svadd_x(pg32_all, c_v21, out_v3_f);
        svst1(pg_c_16, c + (i + 32) * ldc, svuzp1(svcvt_bf16_x(pg32_all, c_v20), svcvt_bf16_x(pg32_all, c_v21)));

        auto c_v30 = svmul_x(pg32_all, svreinterpret_f32(svzip1(zero_bf16, c_v3)), beta);
        auto c_v31 = svmul_x(pg32_all, svreinterpret_f32(svzip2(zero_bf16, c_v3)), beta);
        c_v30 = svadd_x(pg32_all, c_v30, out_v3_f);
        c_v31 = svadd_x(pg32_all, c_v31, out_v3_f);
        svst1(pg_c_16, c + (i + 48) * ldc, svuzp1(svcvt_bf16_x(pg32_all, c_v30), svcvt_bf16_x(pg32_all, c_v31)));
    }
}

__attribute__((always_inline)) inline void bgemm11_save_beta(int ldc, bfloat16_t *c, int m_num, int n_num, float beta)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 2);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int i = 0; i < m_num / 2; i++) {
        svfloat32_t out_v0_f = svread_hor_za32_f32_m(svfloat32_t(), pg32_all, 0, i);
        svbfloat16_t c_v0 = svld1(pg_c_16, c + i * ldc);
        auto c_v00 = svmul_x(pg32_all, svreinterpret_f32(svzip1(zero_bf16, c_v0)), beta);
        c_v00 = svadd_x(pg32_all, c_v00, out_v0_f);
        svst1(pg_c_16, c + i * ldc, svuzp1(svcvt_bf16_x(pg32_all, c_v00), zero_bf16));
    }
}

__attribute__((always_inline)) inline void bgemm_macro_kernel_pf_beta(int bm, int em, int bn, int en, int k,
    int k_offset, int k_num, int ldc, bfloat16_t *a, bfloat16_t *b, bfloat16_t *c, float beta)
    __arm_inout("za") __arm_streaming
{
    const int prefetch_dis = 4;
    for (int mi = bm; mi < em;) {
        if (mi + 64 <= em) {
            int ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_4VL_1VL_pf(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 2, k, k_num, n_num,
                    prefetch_dis);
                bgemm41_save_beta(ldc, c + mi * ldc + ni, n_num, beta);
            }
            mi += 64;
        } else {
            int m_num = std::min(32, (em - mi) * 2), ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_1VL_1VL_pf(a + mi * k + k_offset * m_num / 2, b + ni * k + k_offset * n_num / 2, k, k_num, m_num,
                    n_num, prefetch_dis);
                bgemm11_save_beta(ldc, c + mi * ldc + ni, m_num, n_num, beta);
            }
            mi += 16;
        }
    }
}

__arm_new("za") inline void bgemm_kernel_pf(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b, bfloat16_t *c,
    int macro_kernel_m = 256, int macro_kernel_n = 512, int macro_kernel_k = 1024) __arm_streaming
{
    if (macro_kernel_k < k)
        macro_kernel_k = k;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                bgemm_macro_kernel_pf(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c);
            }
        }
    }
}

__arm_new("za") inline void bgemm_kernel_pf_beta(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b,
    bfloat16_t *c, float beta, int macro_kernel_m = 256, int macro_kernel_n = 512, int macro_kernel_k = 1024)
    __arm_streaming
{
    if (macro_kernel_k < k)
        macro_kernel_k = k;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                bgemm_macro_kernel_pf_beta(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c, beta);
            }
        }
    }
}

} // namespace kutacc