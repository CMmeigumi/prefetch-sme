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

template <bool RowQuant>
static void bgemm_woqs8_kernel_16(int64_t m, int64_t n, int64_t k, const int8_t *a,
    [[maybe_unused]] int64_t lda, const __bf16 *b, [[maybe_unused]] int64_t ldb, __bf16 *c, int64_t ldc, __bf16 _alpha,
    __bf16 _beta, const float *scale) __arm_streaming
{
    svbool_t pg8 = svptrue_b8();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    float alpha = _alpha;
    [[maybe_unused]] float beta = _beta;

    for (int64_t ni = 0; ni < n; ni += 16) {
        for (int64_t mi = 0; mi < m; mi += 16) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            if constexpr (RowQuant) {
                svfloat32_t rscale0 = svld1(pg32, scale + mi);
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 4) {
                svint8_t a_values = svld1(pg8, a + mi * k + 16 * ki);

                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);

                svfloat32_t a00_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a01_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a10_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));
                svfloat32_t a11_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));

                if (RowQuant) {
                    a00_f32 = svmul_x(pg32, a00_f32, rscale00);
                    a01_f32 = svmul_x(pg32, a01_f32, rscale00);
                    a10_f32 = svmul_x(pg32, a10_f32, rscale01);
                    a11_f32 = svmul_x(pg32, a11_f32, rscale01);
                }
                if (!RowQuant) {
                    svfloat32_t cscale0 = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    svfloat32_t cscale1 = svzip1(svdup_f32(scale[ki + 2]), svdup_f32(scale[ki + 3]));
                    a00_f32 = svmul_x(pg32, a00_f32, cscale0);
                    a01_f32 = svmul_x(pg32, a01_f32, cscale0);
                    a10_f32 = svmul_x(pg32, a10_f32, cscale1);
                    a11_f32 = svmul_x(pg32, a11_f32, cscale1);
                }

                svbfloat16_t a00_bf16 = svcvt_bf16_x(pg32, a00_f32);
                svbfloat16_t a01_bf16 = svcvt_bf16_x(pg32, a01_f32);
                svbfloat16_t a10_bf16 = svcvt_bf16_x(pg32, a10_f32);
                svbfloat16_t a11_bf16 = svcvt_bf16_x(pg32, a11_f32);
                svbfloat16_t a0 = svuzp1(a00_bf16, a01_bf16);
                svbfloat16_t a1 = svuzp1(a10_bf16, a11_bf16);
                svbfloat16_t b0 = svld1(pg16, b + ni * k + 8 * ki);
                svbfloat16_t b1 = svld1(pg16, b + (ni + 16) * k + 8 * ki);

                svmopa_za32_bf16_m(0, pg16, pg16, b0, a0);
                svmopa_za32_bf16_m(1, pg16, pg16, b0, a1);
                svmopa_za32_bf16_m(2, pg16, pg16, b1, a0);
                svmopa_za32_bf16_m(3, pg16, pg16, b1, a1);
            }
            for (int64_t i = 0; i < 8; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svfloat32_t o10 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 2, i);
                svfloat32_t o11 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 3, i);
                o00 = svmul_x(pg32, o00, alpha);
                o01 = svmul_x(pg32, o01, alpha);
                o10 = svmul_x(pg32, o10, alpha);
                o11 = svmul_x(pg32, o11, alpha);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svbfloat16_t o1 = svuzp1(svcvt_bf16_x(pg32, o10), svcvt_bf16_x(pg32, o11));
                svst1(pg16, c + (ni + i) * ldc + mi, o0);
                svst1(pg16, c + (ni + i + 8) * ldc + mi, o1);
            }
        }
    }
}

template <bool RowQuant>
__arm_new("za") static void bgemm_woqs8_enable_matrix(int64_t m, int64_t n, int64_t k, const int8_t *a,
    [[maybe_unused]] int64_t lda, const __bf16 *b, [[maybe_unused]] int64_t ldb, __bf16 *c, int64_t ldc, __bf16 _alpha,
    __bf16 _beta, const float *scale) __arm_streaming
{
    int block_size = std::min(std::min((int)m, (int)n), 32);
    if (block_size == 16) {
        bgemm_woqs8_kernel_16<RowQuant>(m, n, k, a, lda, b, ldb, c, ldc, _alpha, _beta, scale);
        return;
    }

    svbool_t pg8 = svptrue_b8();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    float alpha = _alpha;
    [[maybe_unused]] float beta = _beta;

    for (int64_t ni = 0; ni < n; ni += 32) {
        for (int64_t mi = 0; mi < m; mi += 32) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            svfloat32_t rscale10;
            svfloat32_t rscale11;
            if constexpr (RowQuant) {
                svfloat32_t rscale0 = svld1(pg32, scale + mi);
                svfloat32_t rscale1 = svld1(pg32, scale + mi + 16);
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
                rscale10 = svzip1(rscale1, rscale1);
                rscale11 = svzip2(rscale1, rscale1);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 2) {
                svint8_t a_values = svld1(pg8, a + mi * k + 32 * ki);
                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);
                svfloat32_t a00_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a01_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a10_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));
                svfloat32_t a11_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));
                if constexpr (RowQuant) {
                    a00_f32 = svmul_x(pg32, a00_f32, rscale00);
                    a01_f32 = svmul_x(pg32, a01_f32, rscale01);
                    a10_f32 = svmul_x(pg32, a10_f32, rscale10);
                    a11_f32 = svmul_x(pg32, a11_f32, rscale11);
                }
                if constexpr (!RowQuant) {
                    svfloat32_t cscale = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    a00_f32 = svmul_x(pg32, a00_f32, cscale);
                    a01_f32 = svmul_x(pg32, a01_f32, cscale);
                    a10_f32 = svmul_x(pg32, a10_f32, cscale);
                    a11_f32 = svmul_x(pg32, a11_f32, cscale);
                }
                svbfloat16_t a00_bf16 = svcvt_bf16_x(pg32, a00_f32);
                svbfloat16_t a01_bf16 = svcvt_bf16_x(pg32, a01_f32);
                svbfloat16_t a10_bf16 = svcvt_bf16_x(pg32, a10_f32);
                svbfloat16_t a11_bf16 = svcvt_bf16_x(pg32, a11_f32);
                svbfloat16_t a0 = svuzp1(a00_bf16, a01_bf16);
                svbfloat16_t a1 = svuzp1(a10_bf16, a11_bf16);
                svbfloat16_t b0 = svld1(pg16, b + ni * k + 16 * ki);
                svbfloat16_t b1 = svld1(pg16, b + (ni + 16) * k + 16 * ki);

                svmopa_za32_bf16_m(0, pg16, pg16, b0, a0);
                svmopa_za32_bf16_m(1, pg16, pg16, b0, a1);
                svmopa_za32_bf16_m(2, pg16, pg16, b1, a0);
                svmopa_za32_bf16_m(3, pg16, pg16, b1, a1);
            }
            for (int64_t i = 0; i < 16; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svfloat32_t o10 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 2, i);
                svfloat32_t o11 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 3, i);
                o00 = svmul_x(pg32, o00, alpha);
                o01 = svmul_x(pg32, o01, alpha);
                o10 = svmul_x(pg32, o10, alpha);
                o11 = svmul_x(pg32, o11, alpha);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svbfloat16_t o1 = svuzp1(svcvt_bf16_x(pg32, o10), svcvt_bf16_x(pg32, o11));
                svst1(pg16, c + (ni + i) * ldc + mi, o0);
                svst1(pg16, c + (ni + i + 16) * ldc + mi, o1);
            }
        }
    }
}

} // namespace kutacc
