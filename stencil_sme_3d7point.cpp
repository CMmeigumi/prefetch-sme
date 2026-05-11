// ARM SME 3D 7点 Stencil 模板
// 使用 SVE 向量化 + SME 流模式 + MOPA 外积累加

#include <iostream>
#include <cmath>
#include <chrono>
#include <arm_sme.h>
#include <arm_sve.h>

using namespace std;

// 策略：6项全部 MOPA 累加到 ZA（同行值相同），读出4个 slice 后用基向量提取对角线
__arm_new("za")
void stencil3D_7point_sme(double* __restrict__ grid, double* __restrict__ new_grid, int depth, int rows, int cols) 
    __arm_streaming {
    
    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t one_vec = svdup_f64(1.0);
    svfloat64_t sixth_vec = svdup_f64(1.0 / 6.0);
    svbool_t pg_all = svptrue_b64();
    
    svint64_t idx = svindex_s64(0, 1);
    svfloat64_t zero = svdup_f64(0);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t e0 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 0), one, zero);
    svfloat64_t e1 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 1), one, zero);
    svfloat64_t e2 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 2), one, zero);
    svfloat64_t e3 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 3), one, zero);
    
    for (int k = 1; k < depth - 1; k++) {
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j += SVL) {
                svbool_t pg = svwhilelt_b64(j, cols - 1);
                
                svfloat64_t front = svld1_f64(pg, &grid[(k - 1) * plane_size + i * cols + j]);
                svfloat64_t back = svld1_f64(pg, &grid[(k + 1) * plane_size + i * cols + j]);
                svfloat64_t up = svld1_f64(pg, &grid[k * plane_size + (i - 1) * cols + j]);
                svfloat64_t down = svld1_f64(pg, &grid[k * plane_size + (i + 1) * cols + j]);
                svfloat64_t left = svld1_f64(pg, &grid[k * plane_size + i * cols + j - 1]);
                svfloat64_t right = svld1_f64(pg, &grid[k * plane_size + i * cols + j + 1]);
                
                svzero_za();
                
                svmopa_za64_f64_m(0, pg_all, pg, front, one_vec);
                svmopa_za64_f64_m(0, pg_all, pg, back, one_vec);
                svmopa_za64_f64_m(0, pg_all, pg, up, one_vec);
                svmopa_za64_f64_m(0, pg_all, pg, down, one_vec);
                svmopa_za64_f64_m(0, pg_all, pg, left, one_vec);
                svmopa_za64_f64_m(0, pg_all, pg, right, one_vec);
                
                svfloat64_t s0 = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
                svfloat64_t s1 = svread_hor_za64_m(svundef_f64(), pg_all, 0, 1);
                svfloat64_t s2 = svread_hor_za64_m(svundef_f64(), pg_all, 0, 2);
                svfloat64_t s3 = svread_hor_za64_m(svundef_f64(), pg_all, 0, 3);
                
                svfloat64_t sum = svmul_f64_z(pg, s0, e0);
                sum = svmla_f64_z(pg, sum, s1, e1);
                sum = svmla_f64_z(pg, sum, s2, e2);
                sum = svmla_f64_z(pg, sum, s3, e3);
                
                svfloat64_t result = svmul_f64_z(pg, sum, sixth_vec);
                svst1_f64(pg, &new_grid[k * plane_size + i * cols + j], result);
            }
        }
    }
}

void initializeGrid3D(double* grid, int depth, int rows, int cols) {
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (k == 0 || k == depth - 1 || i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                    grid[k * rows * cols + i * cols + j] = 0.0;
                } else if (k >= depth/3 && k <= 2*depth/3 && 
                          i >= rows/3 && i <= 2*rows/3 && 
                          j >= cols/3 && j <= 2*cols/3) {
                    grid[k * rows * cols + i * cols + j] = 100.0;
                } else {
                    grid[k * rows * cols + i * cols + j] = 0.0;
                }
            }
        }
    }
}

int main() {
    cout << "========================================" << endl;
    cout << "   3D 7点 Stencil 模板 (SME MOPA)" << endl;
    cout << "========================================" << endl << endl;

    const int DEPTH = 64;
    const int ROWS = 128;
    const int COLS = 128;
    const int ITERATIONS = 100;
    
    double* grid = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    double* new_grid = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    
    initializeGrid3D(grid, DEPTH, ROWS, COLS);
    
    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil3D_7point_sme(grid, new_grid, DEPTH, ROWS, COLS);
        swap(grid, new_grid);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    free(grid);
    free(new_grid);

    cout << "========================================" << endl;
    cout << "测试完成！" << endl;
    cout << "========================================" << endl;

    return 0;
}