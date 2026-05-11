// ARM SME 2D 5点 Stencil 模板（热传导方程）
// 使用 SVE 向量化 + SME 流模式 + MOPA 外积累加

#include <iostream>
#include <cmath>
#include <chrono>
#include <arm_sme.h>
#include <arm_sve.h>

using namespace std;

// 策略：基向量 e_k 作 zn，邻居作 zm，4个邻居存入 ZA 的 4 个 slice，读出求和
__arm_new("za")
void stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid, int rows, int cols) 
    __arm_streaming {
    
    uint64_t SVL = svcntd();
    svfloat64_t quarter_vec = svdup_f64(0.25);
    svbool_t pg_all = svptrue_b64();
    
    svint64_t idx = svindex_s64(0, 1);
    svfloat64_t zero = svdup_f64(0);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t e0 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 0), one, zero);
    svfloat64_t e1 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 1), one, zero);
    svfloat64_t e2 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 2), one, zero);
    svfloat64_t e3 = svsel_f64(svcmpeq_n_s64(pg_all, idx, 3), one, zero);
    
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j += SVL) {
            svbool_t pg = svwhilelt_b64(j, cols - 1);
            
            svfloat64_t up = svld1_f64(pg, &grid[(i - 1) * cols + j]);
            svfloat64_t down = svld1_f64(pg, &grid[(i + 1) * cols + j]);
            svfloat64_t left = svld1_f64(pg, &grid[i * cols + j - 1]);
            svfloat64_t right = svld1_f64(pg, &grid[i * cols + j + 1]);
            
            svzero_za();
            
            svmopa_za64_f64_m(0, pg_all, pg, e0, up);
            svmopa_za64_f64_m(0, pg_all, pg, e1, down);
            svmopa_za64_f64_m(0, pg_all, pg, e2, left);
            svmopa_za64_f64_m(0, pg_all, pg, e3, right);
            
            svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
            sum = svadd_f64_z(pg, sum, svread_hor_za64_m(svundef_f64(), pg_all, 0, 1));
            sum = svadd_f64_z(pg, sum, svread_hor_za64_m(svundef_f64(), pg_all, 0, 2));
            sum = svadd_f64_z(pg, sum, svread_hor_za64_m(svundef_f64(), pg_all, 0, 3));
            
            svfloat64_t result = svmul_f64_z(pg, sum, quarter_vec);
            svst1_f64(pg, &new_grid[i * cols + j], result);
        }
    }
}

void initializeGrid(double* grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                grid[i * cols + j] = 0.0;
            } else if (i >= rows/3 && i <= 2*rows/3 && j >= cols/3 && j <= 2*cols/3) {
                grid[i * cols + j] = 100.0;
            } else {
                grid[i * cols + j] = 0.0;
            }
        }
    }
}

void printGrid(double* grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", grid[i * cols + j]);
        }
        cout << endl;
    }
    cout << endl;
}

double computeAverage(double* grid, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += grid[i * cols + j];
        }
    }
    return sum / (rows * cols);
}

double computeMaxDiff(double* grid1, double* grid2, int rows, int cols) {
    double maxDiff = 0.0;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double diff = fabs(grid1[i * cols + j] - grid2[i * cols + j]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
    }
    return maxDiff;
}

int main() {
    cout << "========================================" << endl;
    cout << "   2D 5点 Stencil 模板 (SME MOPA)" << endl;
    cout << "========================================" << endl << endl;

    const int ROWS = 128;
    const int COLS = 128;
    const int ITERATIONS = 1000;
    
    double* grid = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* new_grid = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    
    initializeGrid(grid, ROWS, COLS);
    
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    cout << "初始平均温度: " << computeAverage(grid, ROWS, COLS) << endl;
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_5point_sme(grid, new_grid, ROWS, COLS);
        swap(grid, new_grid);
        
        if ((iter + 1) % 100 == 0) {
            double maxDiff = computeMaxDiff(grid, new_grid, ROWS, COLS);
            if (maxDiff < 1e-6) {
                cout << "在第 " << (iter + 1) << " 次迭代后收敛" << endl;
                break;
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均温度: " << computeAverage(grid, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    cout << "【小网格可视化演示 (10x10)】" << endl;
    cout << "----------------------------------------" << endl;
    
    const int SMALL_SIZE = 10;
    double* small_grid = (double*)aligned_alloc(64, SMALL_SIZE * SMALL_SIZE * sizeof(double));
    double* small_new_grid = (double*)aligned_alloc(64, SMALL_SIZE * SMALL_SIZE * sizeof(double));
    
    initializeGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    
    cout << "初始状态:" << endl;
    printGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    
    for (int iter = 0; iter < 5; iter++) {
        stencil2D_5point_sme(small_grid, small_new_grid, SMALL_SIZE, SMALL_SIZE);
        swap(small_grid, small_new_grid);
        cout << "第 " << (iter + 1) << " 次迭代后:" << endl;
        printGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    }

    free(grid);
    free(new_grid);
    free(small_grid);
    free(small_new_grid);

    cout << "========================================" << endl;
    cout << "测试完成！" << endl;
    cout << "========================================" << endl;

    return 0;
}