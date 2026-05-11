// ARM SME (Scalable Matrix Extension) Stencil 算法实现
// 使用 SVE 向量指令和 SME ZA 累加器

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <arm_sme.h>
#include <arm_sve.h>

using namespace std;

// 2D Stencil 计算 - 5点模板（热传导方程）
// 使用 SVE 向量化 + SME 流模式 + MOPA 外积累加
// 策略：全1向量作 zn，4个邻居全部 MOPA 累加到同一 slice，单次读出，消除 ADD
__arm_new("za")
void stencil2D_5point_sme(double* __restrict__ grid, double* __restrict__ new_grid, int rows, int cols) 
    __arm_streaming {
    
    uint64_t SVL = svcntd();
    svfloat64_t quarter_vec = svdup_f64(0.25);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();
    
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j += SVL) {
            svbool_t pg = svwhilelt_b64(j, cols - 1);
            
            svfloat64_t up = svld1_f64(pg, &grid[(i - 1) * cols + j]);
            svfloat64_t down = svld1_f64(pg, &grid[(i + 1) * cols + j]);
            svfloat64_t left = svld1_f64(pg, &grid[i * cols + j - 1]);
            svfloat64_t right = svld1_f64(pg, &grid[i * cols + j + 1]);
            
            svzero_za();
            
            svmopa_za64_f64_m(0, pg_all, pg, ones, up);
            svmopa_za64_f64_m(0, pg_all, pg, ones, down);
            svmopa_za64_f64_m(0, pg_all, pg, ones, left);
            svmopa_za64_f64_m(0, pg_all, pg, ones, right);
            
            svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
            
            svfloat64_t result = svmul_f64_z(pg, sum, quarter_vec);
            svst1_f64(pg, &new_grid[i * cols + j], result);
        }
    }
}

// 2D Stencil 计算 - 9点模板（更精确的拉普拉斯算子）
// 使用 SVE 向量化 + SME 流模式 + MOPA 外积累加
// 策略：预加权后用全1向量作 zn，全部 MOPA 累加到同一 slice，单次读出，消除 MLA
__arm_new("za")
void stencil2D_9point_sme(double* __restrict__ grid, double* __restrict__ new_grid, int rows, int cols) 
    __arm_streaming {
    
    uint64_t SVL = svcntd();
    svfloat64_t four_vec = svdup_f64(4.0);
    svfloat64_t half_vec = svdup_f64(0.5);
    svfloat64_t eighth_vec = svdup_f64(0.125);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();
    
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j += SVL) {
            svbool_t pg = svwhilelt_b64(j, cols - 1);
            
            svfloat64_t center = svld1_f64(pg, &grid[i * cols + j]);
            svfloat64_t up = svld1_f64(pg, &grid[(i - 1) * cols + j]);
            svfloat64_t down = svld1_f64(pg, &grid[(i + 1) * cols + j]);
            svfloat64_t left = svld1_f64(pg, &grid[i * cols + j - 1]);
            svfloat64_t right = svld1_f64(pg, &grid[i * cols + j + 1]);
            svfloat64_t up_left = svld1_f64(pg, &grid[(i - 1) * cols + j - 1]);
            svfloat64_t up_right = svld1_f64(pg, &grid[(i - 1) * cols + j + 1]);
            svfloat64_t down_left = svld1_f64(pg, &grid[(i + 1) * cols + j - 1]);
            svfloat64_t down_right = svld1_f64(pg, &grid[(i + 1) * cols + j + 1]);
            
            svfloat64_t w_center = svmul_f64_z(pg, center, four_vec);
            svfloat64_t w_ul = svmul_f64_z(pg, up_left, half_vec);
            svfloat64_t w_ur = svmul_f64_z(pg, up_right, half_vec);
            svfloat64_t w_dl = svmul_f64_z(pg, down_left, half_vec);
            svfloat64_t w_dr = svmul_f64_z(pg, down_right, half_vec);
            
            svzero_za();
            
            svmopa_za64_f64_m(0, pg_all, pg, ones, w_center);
            svmopa_za64_f64_m(0, pg_all, pg, ones, up);
            svmopa_za64_f64_m(0, pg_all, pg, ones, down);
            svmopa_za64_f64_m(0, pg_all, pg, ones, left);
            svmopa_za64_f64_m(0, pg_all, pg, ones, right);
            svmopa_za64_f64_m(0, pg_all, pg, ones, w_ul);
            svmopa_za64_f64_m(0, pg_all, pg, ones, w_ur);
            svmopa_za64_f64_m(0, pg_all, pg, ones, w_dl);
            svmopa_za64_f64_m(0, pg_all, pg, ones, w_dr);
            
            svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
            
            svfloat64_t result = svmul_f64_z(pg, sum, eighth_vec);
            svst1_f64(pg, &new_grid[i * cols + j], result);
        }
    }
}

// 3D Stencil 计算 - 7点模板
// 使用 SVE 向量化 + SME 流模式 + MOPA 外积累加
// 策略：全1向量作 zn，6个邻居全部 MOPA 累加到同一 slice，单次读出，消除 MLA
__arm_new("za")
void stencil3D_7point_sme(double* __restrict__ grid, double* __restrict__ new_grid, int depth, int rows, int cols) 
    __arm_streaming {
    
    uint64_t SVL = svcntd();
    int plane_size = rows * cols;
    svfloat64_t sixth_vec = svdup_f64(1.0 / 6.0);
    svfloat64_t ones = svdup_f64(1.0);
    svbool_t pg_all = svptrue_b64();
    
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
                
                svmopa_za64_f64_m(0, pg_all, pg, ones, front);
                svmopa_za64_f64_m(0, pg_all, pg, ones, back);
                svmopa_za64_f64_m(0, pg_all, pg, ones, up);
                svmopa_za64_f64_m(0, pg_all, pg, ones, down);
                svmopa_za64_f64_m(0, pg_all, pg, ones, left);
                svmopa_za64_f64_m(0, pg_all, pg, ones, right);
                
                svfloat64_t sum = svread_hor_za64_m(svundef_f64(), pg_all, 0, 0);
                
                svfloat64_t result = svmul_f64_z(pg, sum, sixth_vec);
                svst1_f64(pg, &new_grid[k * plane_size + i * cols + j], result);
            }
        }
    }
}

// 初始化网格（中心高温度，边缘低温度）
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

// 初始化3D网格
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

// 打印网格（用于小尺寸调试）
void printGrid(double* grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", grid[i * cols + j]);
        }
        cout << endl;
    }
    cout << endl;
}

// 计算网格的平均值
double computeAverage(double* grid, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += grid[i * cols + j];
        }
    }
    return sum / (rows * cols);
}

// 计算最大变化量
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
    cout << "      Stencil 算法演示程序" << endl;
    cout << "========================================" << endl << endl;

    // ==================== 2D 5点模板测试 ====================
    cout << "【测试1】2D 5点 Stencil 模板" << endl;
    cout << "----------------------------------------" << endl;
    
    const int ROWS = 512;
    const int COLS = 512;
    const int ITERATIONS = 100;
    
    double* grid_2d = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* new_grid_2d = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    cout << "初始平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    
    // 执行迭代
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_5point_sme(grid_2d, new_grid_2d, ROWS, COLS);
        
        // 交换网格
        swap(grid_2d, new_grid_2d);
        
        // 每100次迭代检查收敛
        if ((iter + 1) % 100 == 0) {
            double maxDiff = computeMaxDiff(grid_2d, new_grid_2d, ROWS, COLS);
            if (maxDiff < 1e-6) {
                cout << "在第 " << (iter + 1) << " 次迭代后收敛" << endl;
                break;
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    // ==================== 2D 9点模板测试 ====================
    cout << "【测试2】2D 9点 Stencil 模板" << endl;
    cout << "----------------------------------------" << endl;
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    cout << "初始平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    
    start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_9point_sme(grid_2d, new_grid_2d, ROWS, COLS);
        swap(grid_2d, new_grid_2d);
    }
    
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    // ==================== 3D 7点模板测试 ====================
    cout << "【测试3】3D 7点 Stencil 模板" << endl;
    cout << "----------------------------------------" << endl;
    
    const int DEPTH = 128;
    const int ITERATIONS_3D = 20;
    
    double* grid_3d = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    double* new_grid_3d = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    
    initializeGrid3D(grid_3d, DEPTH, ROWS, COLS);
    
    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS_3D << endl;
    
    start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS_3D; iter++) {
        stencil3D_7point_sme(grid_3d, new_grid_3d, DEPTH, ROWS, COLS);
        swap(grid_3d, new_grid_3d);
    }
    
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    // ==================== 小网格可视化 ====================
    cout << "【测试4】小网格可视化演示 (10x10)" << endl;
    cout << "----------------------------------------" << endl;
    
    const int SMALL_SIZE = 10;
    double* small_grid = (double*)aligned_alloc(64, SMALL_SIZE * SMALL_SIZE * sizeof(double));
    double* small_new_grid = (double*)aligned_alloc(64, SMALL_SIZE * SMALL_SIZE * sizeof(double));
    
    initializeGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    
    cout << "初始状态:" << endl;
    printGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    
    // 执行5次迭代
    for (int iter = 0; iter < 5; iter++) {
        stencil2D_5point_sme(small_grid, small_new_grid, SMALL_SIZE, SMALL_SIZE);
        swap(small_grid, small_new_grid);
        cout << "第 " << (iter + 1) << " 次迭代后:" << endl;
        printGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    }

    free(grid_2d);
    free(new_grid_2d);
    free(grid_3d);
    free(new_grid_3d);
    free(small_grid);
    free(small_new_grid);

    cout << "========================================" << endl;
    cout << "所有测试完成！" << endl;
    cout << "========================================" << endl;

    return 0;
}