// ARM SME (Scalable Matrix Extension) Stencil 算法实现
// 使用正确的 ACLE 属性和 intrinsic

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <arm_sme.h>
#include <arm_sve.h>

using namespace std;

// ============================================================
// 使用 __arm_streaming 关键字管理流模式
// 编译器会自动在函数入口插入 smstart，出口插入 smstop
// ============================================================

// ============================================================
// 2D 5点 Stencil - 使用 SVE 向量指令（在流模式下）
// 模板: new[i][j] = (up + down + left + right) / 4
// ============================================================

void stencil2D_5point_sve(double* grid, double* new_grid, int rows, int cols) __arm_streaming {
    uint64_t VL = svcntd();

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j += VL) {
            svbool_t pg = svwhilelt_b64(j, cols - 1);
            
            svfloat64_t up = svld1_f64(pg, &grid[(i - 1) * cols + j]);
            svfloat64_t down = svld1_f64(pg, &grid[(i + 1) * cols + j]);
            svfloat64_t left = svld1_f64(pg, &grid[i * cols + j - 1]);
            svfloat64_t right = svld1_f64(pg, &grid[i * cols + j + 1]);
            
            svfloat64_t sum = svadd_f64_z(pg, up, down);
            sum = svadd_f64_z(pg, sum, left);
            sum = svadd_f64_z(pg, sum, right);
            svfloat64_t result = svmul_f64_z(pg, sum, svdup_f64(0.25));
            
            svst1_f64(pg, &new_grid[i * cols + j], result);
        }
    }
}

// ============================================================
// 2D 9点 Stencil - 使用 SVE 向量指令（在流模式下）
// 模板: new[i][j] = (4*center + up + down + left + right + 0.5*(对角线)) / 8
// ============================================================

void stencil2D_9point_sve(double* grid, double* new_grid, int rows, int cols) __arm_streaming {
    uint64_t VL = svcntd();

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j += VL) {
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
            
            svfloat64_t center_weighted = svmul_f64_z(pg, center, svdup_f64(4.0));
            
            svfloat64_t sum_ortho = svadd_f64_z(pg, up, down);
            sum_ortho = svadd_f64_z(pg, sum_ortho, left);
            sum_ortho = svadd_f64_z(pg, sum_ortho, right);
            
            svfloat64_t sum_diag = svadd_f64_z(pg, up_left, up_right);
            sum_diag = svadd_f64_z(pg, sum_diag, down_left);
            sum_diag = svadd_f64_z(pg, sum_diag, down_right);
            sum_diag = svmul_f64_z(pg, sum_diag, svdup_f64(0.5));
            
            svfloat64_t total = svadd_f64_z(pg, center_weighted, sum_ortho);
            total = svadd_f64_z(pg, total, sum_diag);
            svfloat64_t result = svmul_f64_z(pg, total, svdup_f64(0.125));
            
            svst1_f64(pg, &new_grid[i * cols + j], result);
        }
    }
}

// ============================================================
// 3D 7点 Stencil - 使用 SVE 向量指令（在流模式下）
// 模板: new[k][i][j] = (front + back + up + down + left + right) / 6
// ============================================================

void stencil3D_7point_sve(double* grid, double* new_grid, int depth, int rows, int cols) __arm_streaming {
    uint64_t VL = svcntd();
    int plane_size = rows * cols;

    for (int k = 1; k < depth - 1; k++) {
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j += VL) {
                svbool_t pg = svwhilelt_b64(j, cols - 1);
                
                int idx = k * plane_size + i * cols + j;
                
                svfloat64_t front = svld1_f64(pg, &grid[(k - 1) * plane_size + i * cols + j]);
                svfloat64_t back = svld1_f64(pg, &grid[(k + 1) * plane_size + i * cols + j]);
                svfloat64_t up = svld1_f64(pg, &grid[k * plane_size + (i - 1) * cols + j]);
                svfloat64_t down = svld1_f64(pg, &grid[k * plane_size + (i + 1) * cols + j]);
                svfloat64_t left = svld1_f64(pg, &grid[idx - 1]);
                svfloat64_t right = svld1_f64(pg, &grid[idx + 1]);
                
                svfloat64_t sum = svadd_f64_z(pg, front, back);
                sum = svadd_f64_z(pg, sum, up);
                sum = svadd_f64_z(pg, sum, down);
                sum = svadd_f64_z(pg, sum, left);
                sum = svadd_f64_z(pg, sum, right);
                svfloat64_t result = svmul_f64_z(pg, sum, svdup_f64(1.0 / 6.0));
                
                svst1_f64(pg, &new_grid[idx], result);
            }
        }
    }
}

// ============================================================
// 辅助函数
// ============================================================

void initializeGrid(double* grid, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        int row = i / cols;
        int col = i % cols;
        if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
            grid[i] = 0.0;
        } else if (row >= rows / 3 && row <= 2 * rows / 3 && 
                   col >= cols / 3 && col <= 2 * cols / 3) {
            grid[i] = 100.0;
        } else {
            grid[i] = 0.0;
        }
    }
}

void initializeGrid3D(double* grid, int depth, int rows, int cols) {
    int plane_size = rows * cols;
    for (int i = 0; i < depth * plane_size; i++) {
        int k = i / plane_size;
        int rem = i % plane_size;
        int row = rem / cols;
        int col = rem % cols;
        
        if (k == 0 || k == depth - 1 || row == 0 || row == rows - 1 || 
            col == 0 || col == cols - 1) {
            grid[i] = 0.0;
        } else if (k >= depth / 3 && k <= 2 * depth / 3 && 
                   row >= rows / 3 && row <= 2 * rows / 3 && 
                   col >= cols / 3 && col <= 2 * cols / 3) {
            grid[i] = 100.0;
        } else {
            grid[i] = 0.0;
        }
    }
}

double computeAverage(double* grid, int rows, int cols) {
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (grid[i] != 0.0) {
            sum += grid[i];
            count++;
        }
    }
    return count > 0 ? sum / count : 0.0;
}

// ============================================================
// 主函数（非流模式）
// ============================================================

int main() {
    cout << "========================================" << endl;
    cout << "   ARM SME Stencil 算法演示程序" << endl;
    cout << "   (使用正确的 ACLE 属性和 Intrinsic)" << endl;
    cout << "========================================" << endl << endl;
    
    uint64_t VL = svcntd();
    cout << "Streaming Vector Length (双精度元素个数): " << VL << endl;
    cout << endl;
    
    const int ROWS = 128;
    const int COLS = 128;
    const int ITERATIONS = 1000;
    
    double* grid_2d = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* new_grid_2d = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "======================================" << endl;
    cout << "测试1: SVE 向量化 2D 5点 Stencil" << endl;
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    cout << "初始平均值: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    
    auto start = chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_5point_sve(grid_2d, new_grid_2d, ROWS, COLS);
        swap(grid_2d, new_grid_2d);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均值: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "======================================" << endl;
    cout << "测试2: SVE 向量化 2D 9点 Stencil" << endl;
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    
    start = chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_9point_sve(grid_2d, new_grid_2d, ROWS, COLS);
        swap(grid_2d, new_grid_2d);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均值: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;
    
    const int DEPTH = 64;
    const int ITERATIONS_3D = 100;
    
    double* grid_3d = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    double* new_grid_3d = (double*)aligned_alloc(64, DEPTH * ROWS * COLS * sizeof(double));
    
    initializeGrid3D(grid_3d, DEPTH, ROWS, COLS);
    
    cout << "======================================" << endl;
    cout << "测试3: SVE 向量化 3D 7点 Stencil" << endl;
    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS_3D << endl;
    
    start = chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERATIONS_3D; iter++) {
        stencil3D_7point_sve(grid_3d, new_grid_3d, DEPTH, ROWS, COLS);
        swap(grid_3d, new_grid_3d);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;
    
    free(grid_2d);
    free(new_grid_2d);
    free(grid_3d);
    free(new_grid_3d);
    
    cout << "========================================" << endl;
    cout << "所有 SME 测试完成！" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
