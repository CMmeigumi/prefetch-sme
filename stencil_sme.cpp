// ARM SME (Scalable Matrix Extension) Stencil 算法实现
// 使用 Clang 正确的 SME 属性写法

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <arm_sme.h>
#include <arm_sve.h>

using namespace std;

// ============================================================
// SME 向量长度常量
// ============================================================

// 获取 Streaming Vector Length (SVL) - 双精度元素个数
// svcntd() 返回 SVL 中的双精度元素数量
// ZA 存储是 SVL x SVL 的二维数组，所以 ZA 行数 = svcntd()

// ============================================================
// SME 优化的 2D 5点 Stencil - 使用 ZA 累加器
// 使用 __arm_locally_streaming 限定符：
// - 编译器自动在函数开头插入 smstart，结尾插入 smstop
// - 函数在非流模式下被调用，进入函数后切换到流模式
// ============================================================

void stencil2D_5point_sme_za(double* grid, double* new_grid, int rows, int cols) __arm_locally_streaming {
    // 获取 SVL (Streaming Vector Length) 中的双精度元素个数
    // ZA 存储维度为 SVL x SVL，所以 ZA 行数 = svcntd()
    const uint64_t SVL = svcntd();
    
    // 使用 ZA 累加器进行矩阵级计算
    for (int i = 1; i < rows - 1; i += SVL) {
        int block_rows = min((uint64_t)(rows - 1 - i), SVL);
        
        // 将数据加载到 ZA 累加器中
        for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
            int grid_row = i + za_row;
            
            // 加载当前行及其上下行到 ZA 的不同 slice
            svfloat64_t up = svld1_f64(svptrue_b64(), &grid[(grid_row - 1) * cols + 1]);
            svfloat64_t center = svld1_f64(svptrue_b64(), &grid[grid_row * cols + 1]);
            svfloat64_t down = svld1_f64(svptrue_b64(), &grid[(grid_row + 1) * cols + 1]);
            
            // 使用 SME intrinsic 将向量存储到 ZA 累加器
            // ZA 布局: slice 0 = up, slice 1 = center, slice 2 = down
            smst1_f64(0, za_row, up);
            smst1_f64(1, za_row, center);
            smst1_f64(2, za_row, down);
        }
        
        // 使用 SME 矩阵操作进行 Stencil 计算
        for (int j = 1; j < cols - 1; j += SVL) {
            // 加载左右列数据到 ZA
            for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
                int grid_row = i + za_row;
                int grid_col = j;

                svfloat64_t left = svld1_f64(svptrue_b64(), &grid[grid_row * cols + grid_col - 1]);
                svfloat64_t right = svld1_f64(svptrue_b64(), &grid[grid_row * cols + grid_col + 1]);

                smst1_f64(3, za_row, left);   // slice 3 = left
                smst1_f64(4, za_row, right);  // slice 4 = right
            }

            // 使用 SME MOPA 指令从 ZA 读取并计算
            for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
                // 从 ZA 累加器读取数据 (MOPA 核心操作)
                svfloat64_t up_val = smld1_f64(0, za_row);
                svfloat64_t down_val = smld1_f64(2, za_row);
                svfloat64_t left_val = smld1_f64(3, za_row);
                svfloat64_t right_val = smld1_f64(4, za_row);
                
                // 计算: result = (up + down + left + right) / 4
                svfloat64_t sum = svadd_f64_x(svptrue_b64(), up_val, down_val);
                sum = svadd_f64_x(svptrue_b64(), sum, left_val);
                sum = svadd_f64_x(svptrue_b64(), sum, right_val);
                
                // 使用 SME 缩放操作
                svfloat64_t result = svmul_f64_x(svptrue_b64(), sum, svdup_f64(0.25));
                
                // 存储结果
                svst1_f64(svptrue_b64(), &new_grid[(i + za_row) * cols + j], result);
            }
        }
    }
}

// ============================================================
// SME 优化的 2D 9点 Stencil
// ============================================================

void stencil2D_9point_sme_za(double* grid, double* new_grid, int rows, int cols) __arm_locally_streaming {
    // 9点模板: result = (4*center + up + down + left + right + 0.5*(对角线)) / 8

    const uint64_t SVL = svcntd();

    for (int i = 1; i < rows - 1; i += SVL) {
        int block_rows = min((uint64_t)(rows - 1 - i), SVL);
        
        // 预加载三行数据到 ZA
        for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
            int grid_row = i + za_row;
            
            svfloat64_t up = svld1_f64(svptrue_b64(), &grid[(grid_row - 1) * cols + 1]);
            svfloat64_t center = svld1_f64(svptrue_b64(), &grid[grid_row * cols + 1]);
            svfloat64_t down = svld1_f64(svptrue_b64(), &grid[(grid_row + 1) * cols + 1]);
            
            smst1_f64(0, za_row, up);      // slice 0 = up
            smst1_f64(1, za_row, center);  // slice 1 = center
            smst1_f64(2, za_row, down);    // slice 2 = down
        }
        
        for (int j = 1; j < cols - 1; j += SVL) {
            // 加载对角元素到 ZA
            for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
                int grid_row = i + za_row;
                int grid_col = j;
                
                svfloat64_t up_left = svld1_f64(svptrue_b64(), &grid[(grid_row - 1) * cols + grid_col - 1]);
                svfloat64_t up_right = svld1_f64(svptrue_b64(), &grid[(grid_row - 1) * cols + grid_col + 1]);
                svfloat64_t down_left = svld1_f64(svptrue_b64(), &grid[(grid_row + 1) * cols + grid_col - 1]);
                svfloat64_t down_right = svld1_f64(svptrue_b64(), &grid[(grid_row + 1) * cols + grid_col + 1]);
                
                smst1_f64(3, za_row, up_left);
                smst1_f64(4, za_row, up_right);
                smst1_f64(5, za_row, down_left);
                smst1_f64(6, za_row, down_right);
            }
            
            // 执行 9点模板计算
            for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
                // 从 ZA 读取数据
                svfloat64_t up = smld1_f64(0, za_row);
                svfloat64_t center = smld1_f64(1, za_row);
                svfloat64_t down = smld1_f64(2, za_row);
                svfloat64_t up_left = smld1_f64(3, za_row);
                svfloat64_t up_right = smld1_f64(4, za_row);
                svfloat64_t down_left = smld1_f64(5, za_row);
                svfloat64_t down_right = smld1_f64(6, za_row);
                
                // 计算正交邻居和
                svfloat64_t left = svld1_f64(svptrue_b64(), &grid[(i + za_row) * cols + j - 1]);
                svfloat64_t right = svld1_f64(svptrue_b64(), &grid[(i + za_row) * cols + j + 1]);
                
                svfloat64_t sum_ortho = svadd_f64_x(svptrue_b64(), up, down);
                sum_ortho = svadd_f64_x(svptrue_b64(), sum_ortho, left);
                sum_ortho = svadd_f64_x(svptrue_b64(), sum_ortho, right);
                
                // 计算对角邻居和（加权 0.5）
                svfloat64_t sum_diag = svadd_f64_x(svptrue_b64(), up_left, up_right);
                sum_diag = svadd_f64_x(svptrue_b64(), sum_diag, down_left);
                sum_diag = svadd_f64_x(svptrue_b64(), sum_diag, down_right);
                sum_diag = svmul_f64_x(svptrue_b64(), sum_diag, svdup_f64(0.5));
                
                // 计算中心加权 (4 * center)
                svfloat64_t center_weighted = svmul_f64_x(svptrue_b64(), center, svdup_f64(4.0));
                
                // 总和并除以8
                svfloat64_t total = svadd_f64_x(svptrue_b64(), center_weighted, sum_ortho);
                total = svadd_f64_x(svptrue_b64(), total, sum_diag);
                svfloat64_t result = svmul_f64_x(svptrue_b64(), total, svdup_f64(0.125));
                
                // 存储结果
                svst1_f64(svptrue_b64(), &new_grid[(i + za_row) * cols + j], result);
            }
        }
    }
}

// ============================================================
// SME 优化的 3D 7点 Stencil
// ============================================================

void stencil3D_7point_sme_za(double* grid, double* new_grid, int depth, int rows, int cols) __arm_locally_streaming {
    const uint64_t SVL = svcntd();
    int plane_size = rows * cols;

    for (int k = 1; k < depth - 1; k++) {
        for (int i = 1; i < rows - 1; i += SVL) {
            int block_rows = min((uint64_t)(rows - 1 - i), SVL);
            
            // 加载三层数据到 ZA
            for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
                int grid_row = i + za_row;
                
                svfloat64_t front = svld1_f64(svptrue_b64(), &grid[(k - 1) * plane_size + grid_row * cols + 1]);
                svfloat64_t center = svld1_f64(svptrue_b64(), &grid[k * plane_size + grid_row * cols + 1]);
                svfloat64_t back = svld1_f64(svptrue_b64(), &grid[(k + 1) * plane_size + grid_row * cols + 1]);
                
                smst1_f64(0, za_row, front);
                smst1_f64(1, za_row, center);
                smst1_f64(2, za_row, back);
            }
            
            for (int j = 1; j < cols - 1; j += SVL) {
                for (uint64_t za_row = 0; za_row < block_rows; za_row++) {
                    int grid_row = i + za_row;
                    int idx = k * plane_size + grid_row * cols + j;
                    
                    // 从 ZA 读取前后层
                    svfloat64_t front = smld1_f64(0, za_row);
                    svfloat64_t back = smld1_f64(2, za_row);
                    
                    // 加载上下左右
                    svfloat64_t up = svld1_f64(svptrue_b64(), &grid[k * plane_size + (grid_row - 1) * cols + j]);
                    svfloat64_t down = svld1_f64(svptrue_b64(), &grid[k * plane_size + (grid_row + 1) * cols + j]);
                    svfloat64_t left = svld1_f64(svptrue_b64(), &grid[idx - 1]);
                    svfloat64_t right = svld1_f64(svptrue_b64(), &grid[idx + 1]);
                    
                    // 7点求和
                    svfloat64_t sum = svadd_f64_x(svptrue_b64(), front, back);
                    sum = svadd_f64_x(svptrue_b64(), sum, up);
                    sum = svadd_f64_x(svptrue_b64(), sum, down);
                    sum = svadd_f64_x(svptrue_b64(), sum, left);
                    sum = svadd_f64_x(svptrue_b64(), sum, right);
                    
                    // 除以6
                    svfloat64_t result = svmul_f64_x(svptrue_b64(), sum, svdup_f64(1.0/6.0));
                    
                    svst1_f64(svptrue_b64(), &new_grid[idx], result);
                }
            }
        }
    }
}

// ============================================================
// 初始化和辅助函数
// ============================================================

void initializeGrid(double* grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                grid[idx] = 0.0;
            } else if (i >= rows/3 && i <= 2*rows/3 && j >= cols/3 && j <= 2*cols/3) {
                grid[idx] = 100.0;
            } else {
                grid[idx] = 0.0;
            }
        }
    }
}

void initializeGrid3D(double* grid, int depth, int rows, int cols) {
    int plane_size = rows * cols;
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = k * plane_size + i * cols + j;
                if (k == 0 || k == depth - 1 || i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                    grid[idx] = 0.0;
                } else if (k >= depth/3 && k <= 2*depth/3 && 
                          i >= rows/3 && i <= 2*rows/3 && 
                          j >= cols/3 && j <= 2*cols/3) {
                    grid[idx] = 100.0;
                } else {
                    grid[idx] = 0.0;
                }
            }
        }
    }
}

double computeAverage(double* grid, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        sum += grid[i];
    }
    return sum / (rows * cols);
}

// ============================================================
// 主函数（非流模式）
// ============================================================

int main() {
    cout << "========================================" << endl;
    cout << "   ARM SME Stencil 算法演示程序" << endl;
    cout << "   (使用 Clang 正确的 SME 属性)" << endl;
    cout << "========================================" << endl << endl;
    
    // 获取 SME 配置信息
    uint64_t vl = svcntd();
    cout << "SME Streaming Vector Length (双精度元素个数): " << vl << endl;
    cout << "ZA 存储维度: " << vl << " x " << vl << endl;
    cout << endl;
    
    // ==================== 2D 5点模板测试 (locally_streaming) ====================
    cout << "【测试1】SME ZA 2D 5点 Stencil (__arm_locally_streaming)" << endl;
    cout << "----------------------------------------" << endl;
    
    const int ROWS = 128;
    const int COLS = 128;
    const int ITERATIONS = 1000;
    
    double* grid_2d = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    double* new_grid_2d = (double*)aligned_alloc(64, ROWS * COLS * sizeof(double));
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    cout << "初始平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_5point_sme_za(grid_2d, new_grid_2d, ROWS, COLS);
        swap(grid_2d, new_grid_2d);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;
    
    // ==================== 2D 9点模板测试 ====================
    cout << "【测试2】SME ZA 2D 9点 Stencil (__arm_locally_streaming)" << endl;
    cout << "----------------------------------------" << endl;
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    
    start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_9point_sme_za(grid_2d, new_grid_2d, ROWS, COLS);
        swap(grid_2d, new_grid_2d);
    }
    
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;
    
    // ==================== 3D 7点模板测试 ====================
    cout << "【测试3】SME ZA 3D 7点 Stencil (__arm_locally_streaming)" << endl;
    cout << "----------------------------------------" << endl;
    
    const int DEPTH = 64;
    const int ITERATIONS_3D = 100;
    int total_size_3d = DEPTH * ROWS * COLS;
    
    double* grid_3d = (double*)aligned_alloc(64, total_size_3d * sizeof(double));
    double* new_grid_3d = (double*)aligned_alloc(64, total_size_3d * sizeof(double));
    
    initializeGrid3D(grid_3d, DEPTH, ROWS, COLS);
    
    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS_3D << endl;
    
    start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS_3D; iter++) {
        stencil3D_7point_sme_za(grid_3d, new_grid_3d, DEPTH, ROWS, COLS);
        swap(grid_3d, new_grid_3d);
    }
    
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;
    
    // 释放内存
    free(grid_2d);
    free(new_grid_2d);
    free(grid_3d);
    free(new_grid_3d);
    
    cout << "========================================" << endl;
    cout << "所有 SME 测试完成！" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
