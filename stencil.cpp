#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

// 2D Stencil 计算 - 5点模板（热传导方程）
// 计算: new_grid[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]) / 4
void stencil2D_5point(vector<vector<double>>& grid, vector<vector<double>>& new_grid, int rows, int cols) {
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1]);
        }
    }
}

// 2D Stencil 计算 - 9点模板（更精确的拉普拉斯算子）
void stencil2D_9point(vector<vector<double>>& grid, vector<vector<double>>& new_grid, int rows, int cols) {
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            new_grid[i][j] = (4.0 * grid[i][j] + 
                             grid[i-1][j] + grid[i+1][j] + 
                             grid[i][j-1] + grid[i][j+1] +
                             0.5 * (grid[i-1][j-1] + grid[i-1][j+1] + 
                                    grid[i+1][j-1] + grid[i+1][j+1])) / 8.0;
        }
    }
}

// 3D Stencil 计算 - 7点模板
void stencil3D_7point(vector<vector<vector<double>>>& grid, 
                      vector<vector<vector<double>>>& new_grid,
                      int depth, int rows, int cols) {
    for (int k = 1; k < depth - 1; k++) {
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                new_grid[k][i][j] = (grid[k-1][i][j] + grid[k+1][i][j] +
                                     grid[k][i-1][j] + grid[k][i+1][j] +
                                     grid[k][i][j-1] + grid[k][i][j+1]) / 6.0;
            }
        }
    }
}

// 初始化网格（中心高温度，边缘低温度）
void initializeGrid(vector<vector<double>>& grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 边界条件：边缘为0，中心为100
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                grid[i][j] = 0.0;
            } else if (i >= rows/3 && i <= 2*rows/3 && j >= cols/3 && j <= 2*cols/3) {
                grid[i][j] = 100.0;
            } else {
                grid[i][j] = 0.0;
            }
        }
    }
}

// 初始化3D网格
void initializeGrid3D(vector<vector<vector<double>>>& grid, int depth, int rows, int cols) {
    for (int k = 0; k < depth; k++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (k == 0 || k == depth - 1 || i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                    grid[k][i][j] = 0.0;
                } else if (k >= depth/3 && k <= 2*depth/3 && 
                          i >= rows/3 && i <= 2*rows/3 && 
                          j >= cols/3 && j <= 2*cols/3) {
                    grid[k][i][j] = 100.0;
                } else {
                    grid[k][i][j] = 0.0;
                }
            }
        }
    }
}

// 打印网格（用于小尺寸调试）
void printGrid(const vector<vector<double>>& grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", grid[i][j]);
        }
        cout << endl;
    }
    cout << endl;
}

// 计算网格的平均值
double computeAverage(const vector<vector<double>>& grid, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += grid[i][j];
        }
    }
    return sum / (rows * cols);
}

// 计算最大变化量
double computeMaxDiff(const vector<vector<double>>& grid1, 
                      const vector<vector<double>>& grid2, 
                      int rows, int cols) {
    double maxDiff = 0.0;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double diff = fabs(grid1[i][j] - grid2[i][j]);
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
    
    const int ROWS = 128;
    const int COLS = 128;
    const int ITERATIONS = 1000;
    
    vector<vector<double>> grid_2d(ROWS, vector<double>(COLS));
    vector<vector<double>> new_grid_2d(ROWS, vector<double>(COLS));
    
    initializeGrid(grid_2d, ROWS, COLS);
    
    cout << "网格大小: " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS << endl;
    cout << "初始平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    
    // 执行迭代
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        stencil2D_5point(grid_2d, new_grid_2d, ROWS, COLS);
        
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
        stencil2D_9point(grid_2d, new_grid_2d, ROWS, COLS);
        swap(grid_2d, new_grid_2d);
    }
    
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "最终平均温度: " << computeAverage(grid_2d, ROWS, COLS) << endl;
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    // ==================== 3D 7点模板测试 ====================
    cout << "【测试3】3D 7点 Stencil 模板" << endl;
    cout << "----------------------------------------" << endl;
    
    const int DEPTH = 64;
    const int ITERATIONS_3D = 100;
    
    vector<vector<vector<double>>> grid_3d(DEPTH, 
                                           vector<vector<double>>(ROWS, 
                                                                  vector<double>(COLS)));
    vector<vector<vector<double>>> new_grid_3d(DEPTH, 
                                               vector<vector<double>>(ROWS, 
                                                                      vector<double>(COLS)));
    
    initializeGrid3D(grid_3d, DEPTH, ROWS, COLS);
    
    cout << "网格大小: " << DEPTH << " x " << ROWS << " x " << COLS << endl;
    cout << "迭代次数: " << ITERATIONS_3D << endl;
    
    start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS_3D; iter++) {
        stencil3D_7point(grid_3d, new_grid_3d, DEPTH, ROWS, COLS);
        swap(grid_3d, new_grid_3d);
    }
    
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    cout << "执行时间: " << duration.count() << " ms" << endl << endl;

    // ==================== 小网格可视化 ====================
    cout << "【测试4】小网格可视化演示 (10x10)" << endl;
    cout << "----------------------------------------" << endl;
    
    const int SMALL_SIZE = 10;
    vector<vector<double>> small_grid(SMALL_SIZE, vector<double>(SMALL_SIZE));
    vector<vector<double>> small_new_grid(SMALL_SIZE, vector<double>(SMALL_SIZE));
    
    initializeGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    
    cout << "初始状态:" << endl;
    printGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    
    // 执行5次迭代
    for (int iter = 0; iter < 5; iter++) {
        stencil2D_5point(small_grid, small_new_grid, SMALL_SIZE, SMALL_SIZE);
        swap(small_grid, small_new_grid);
        cout << "第 " << (iter + 1) << " 次迭代后:" << endl;
        printGrid(small_grid, SMALL_SIZE, SMALL_SIZE);
    }

    cout << "========================================" << endl;
    cout << "所有测试完成！" << endl;
    cout << "========================================" << endl;

    return 0;
}
