#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

void stencil1D_3point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int size, int stride);
void stencil2D_5point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int rows, int cols, int stride);
void stencil2D_9point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int rows, int cols, int stride);
void stencil3D_13point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride);
void stencil3D_25point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride);
void stencil3D_27point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride);

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --1d3p-s1    Run 1D 3-point stencil with stride=1" << std::endl;
    std::cout << "  --1d3p-s2    Run 1D 3-point stencil with stride=2" << std::endl;
    std::cout << "  --2d5p-s1    Run 2D 5-point stencil with stride=1" << std::endl;
    std::cout << "  --2d5p-s2    Run 2D 5-point stencil with stride=2" << std::endl;
    std::cout << "  --2d9p-s1    Run 2D 9-point stencil with stride=1" << std::endl;
    std::cout << "  --2d9p-s2    Run 2D 9-point stencil with stride=2" << std::endl;
    std::cout << "  --3d13p-s1    Run 3D 13-point stencil with stride=1" << std::endl;
    std::cout << "  --3d13p-s2    Run 3D 13-point stencil with stride=2" << std::endl;
    std::cout << "  --3d25p-s1    Run 3D 25-point stencil with stride=1" << std::endl;
    std::cout << "  --3d25p-s2    Run 3D 25-point stencil with stride=2" << std::endl;
    std::cout << "  --3d27p-s1    Run 3D 27-point stencil with stride=1" << std::endl;
    std::cout << "  --3d27p-s2    Run 3D 27-point stencil with stride=2" << std::endl;
    std::cout << "  --all        Run all stencils with all strides" << std::endl;
    std::cout << "  -h, --help   Show this help message" << std::endl;
}

double test_stencil_1d_3point(bool run_stride1, bool run_stride2) {
    std::cout << std::endl << "--- 1D 3-point ---" << std::endl;
    const int SIZE = 1048576;
    double* g1 = (double*)malloc(SIZE * sizeof(double));
    double* g2 = (double*)malloc(SIZE * sizeof(double));
    double total_time = 0.0, elapsed;

    if (run_stride1) {
        for (int i = 0; i < SIZE; i++) g1[i] = 1.0 + i;
        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; iter++) stencil1D_3point_scalar(g1, g2, SIZE, 1);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    if (run_stride2) {
        for (int i = 0; i < SIZE; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; iter++) stencil1D_3point_scalar(g1, g2, SIZE, 2);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    free(g1); free(g2);
    return total_time;
}

double test_stencil_2d_5point(bool run_stride1, bool run_stride2) {
    std::cout << std::endl << "--- 2D 5-point ---" << std::endl;
    const int ROWS = 1024, COLS = 1024;
    double* g1 = (double*)malloc(ROWS * COLS * sizeof(double));
    double* g2 = (double*)malloc(ROWS * COLS * sizeof(double));
    double total_time = 0.0, elapsed;

    if (run_stride1) {
        for (int i = 0; i < ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_5point_scalar(g1, g2, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    if (run_stride2) {
        for (int i = 0; i < ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_5point_scalar(g1, g2, ROWS, COLS, 2);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    free(g1); free(g2);
    return total_time;
}

double test_stencil_2d_9point(bool run_stride1, bool run_stride2) {
    std::cout << std::endl << "--- 2D 9-point ---" << std::endl;
    const int ROWS = 1024, COLS = 1024;
    double* g1 = (double*)malloc(ROWS * COLS * sizeof(double));
    double* g2 = (double*)malloc(ROWS * COLS * sizeof(double));
    double total_time = 0.0, elapsed;

    if (run_stride1) {
        for (int i = 0; i < ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_9point_scalar(g1, g2, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    if (run_stride2) {
        for (int i = 0; i < ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil2D_9point_scalar(g1, g2, ROWS, COLS, 2);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    free(g1); free(g2);
    return total_time;
}

double test_stencil_3d_13point(bool run_stride1, bool run_stride2) {
    std::cout << std::endl << "--- 3D 13-point ---" << std::endl;
    const int DEPTH = 128, ROWS = 512, COLS = 512;
    double* g1 = (double*)malloc(DEPTH * ROWS * COLS * sizeof(double));
    double* g2 = (double*)malloc(DEPTH * ROWS * COLS * sizeof(double));
    double total_time = 0.0, elapsed;

    if (run_stride1) {
        for (int i = 0; i < DEPTH * ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_13point_scalar(g1, g2, DEPTH, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    if (run_stride2) {
        for (int i = 0; i < DEPTH * ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_13point_scalar(g1, g2, DEPTH, ROWS, COLS, 2);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    free(g1); free(g2);
    return total_time;
}

double test_stencil_3d_25point(bool run_stride1, bool run_stride2) {
    std::cout << std::endl << "--- 3D 25-point ---" << std::endl;
    const int DEPTH = 128, ROWS = 512, COLS = 512;
    double* g1 = (double*)malloc(DEPTH * ROWS * COLS * sizeof(double));
    double* g2 = (double*)malloc(DEPTH * ROWS * COLS * sizeof(double));
    double total_time = 0.0, elapsed;

    if (run_stride1) {
        for (int i = 0; i < DEPTH * ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_25point_scalar(g1, g2, DEPTH, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    if (run_stride2) {
        for (int i = 0; i < DEPTH * ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_25point_scalar(g1, g2, DEPTH, ROWS, COLS, 2);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    free(g1); free(g2);
    return total_time;
}

double test_stencil_3d_27point(bool run_stride1, bool run_stride2) {
    std::cout << std::endl << "--- 3D 27-point ---" << std::endl;
    const int DEPTH = 128, ROWS = 512, COLS = 512;
    double* g1 = (double*)malloc(DEPTH * ROWS * COLS * sizeof(double));
    double* g2 = (double*)malloc(DEPTH * ROWS * COLS * sizeof(double));
    double total_time = 0.0, elapsed;

    if (run_stride1) {
        for (int i = 0; i < DEPTH * ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=1..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_27point_scalar(g1, g2, DEPTH, ROWS, COLS, 1);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    if (run_stride2) {
        for (int i = 0; i < DEPTH * ROWS * COLS; i++) g1[i] = 1.0 + i;
        std::cout << "stride=2..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; iter++) stencil3D_27point_scalar(g1, g2, DEPTH, ROWS, COLS, 2);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Time: " << elapsed << " s" << std::endl;
        total_time += elapsed;
    }

    free(g1); free(g2);
    return total_time;
}

int main(int argc, char* argv[]) {
    bool run_1d_3point_s1 = false;
    bool run_1d_3point_s2 = false;
    bool run_2d_5point_s1 = false;
    bool run_2d_5point_s2 = false;
    bool run_2d_9point_s1 = false;
    bool run_2d_9point_s2 = false;
    bool run_3d_13point_s1 = false;
    bool run_3d_13point_s2 = false;
    bool run_3d_25point_s1 = false;
    bool run_3d_25point_s2 = false;
    bool run_3d_27point_s1 = false;
    bool run_3d_27point_s2 = false;

    if (argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--1d3p-s1") == 0) {
            run_1d_3point_s1 = true;
        } else if (strcmp(argv[i], "--1d3p-s2") == 0) {
            run_1d_3point_s2 = true;
        } else if (strcmp(argv[i], "--2d5p-s1") == 0) {
            run_2d_5point_s1 = true;
        } else if (strcmp(argv[i], "--2d5p-s2") == 0) {
            run_2d_5point_s2 = true;
        } else if (strcmp(argv[i], "--2d9p-s1") == 0) {
            run_2d_9point_s1 = true;
        } else if (strcmp(argv[i], "--2d9p-s2") == 0) {
            run_2d_9point_s2 = true;
        } else if (strcmp(argv[i], "--3d13p-s1") == 0) {
            run_3d_13point_s1 = true;
        } else if (strcmp(argv[i], "--3d13p-s2") == 0) {
            run_3d_13point_s2 = true;
        } else if (strcmp(argv[i], "--3d25p-s1") == 0) {
            run_3d_25point_s1 = true;
        } else if (strcmp(argv[i], "--3d25p-s2") == 0) {
            run_3d_25point_s2 = true;
        } else if (strcmp(argv[i], "--3d27p-s1") == 0) {
            run_3d_27point_s1 = true;
        } else if (strcmp(argv[i], "--3d27p-s2") == 0) {
            run_3d_27point_s2 = true;
        } else if (strcmp(argv[i], "--all") == 0) {
            run_1d_3point_s1 = true;
            run_1d_3point_s2 = true;
            run_2d_5point_s1 = true;
            run_2d_5point_s2 = true;
            run_2d_9point_s1 = true;
            run_2d_9point_s2 = true;
            run_3d_13point_s1 = true;
            run_3d_13point_s2 = true;
            run_3d_25point_s1 = true;
            run_3d_25point_s2 = true;
            run_3d_27point_s1 = true;
            run_3d_27point_s2 = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "=== Stencil Scalar Tests ===" << std::endl;

    double total_time = 0.0;

    if (run_1d_3point_s1) total_time += test_stencil_1d_3point(true, false);
    if (run_1d_3point_s2) total_time += test_stencil_1d_3point(false, true);
    if (run_2d_5point_s1) total_time += test_stencil_2d_5point(true, false);
    if (run_2d_5point_s2) total_time += test_stencil_2d_5point(false, true);
    if (run_2d_9point_s1) total_time += test_stencil_2d_9point(true, false);
    if (run_2d_9point_s2) total_time += test_stencil_2d_9point(false, true);
    if (run_3d_13point_s1) total_time += test_stencil_3d_13point(true, false);
    if (run_3d_13point_s2) total_time += test_stencil_3d_13point(false, true);
    if (run_3d_25point_s1) total_time += test_stencil_3d_25point(true, false);
    if (run_3d_25point_s2) total_time += test_stencil_3d_25point(false, true);
    if (run_3d_27point_s1) total_time += test_stencil_3d_27point(true, false);
    if (run_3d_27point_s2) total_time += test_stencil_3d_27point(false, true);

    std::cout << std::endl << "=== Total Time: " << total_time << " s ===" << std::endl;

    return 0;
}

// 1D 3-point Stencil 标量版本实现
void stencil1D_3point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int size, int stride) {
    double weight = 1.0 / 3.0;

    for (int i = 1; i < size - 1; i += stride) {
        double sum = grid[i-1] + grid[i] + grid[i+1];
        new_grid[i] = sum * weight;
    }
}

// 2D 5-point Stencil 标量版本实现
void stencil2D_5point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int rows, int cols, int stride) {
    double weight = 1.0 / 5.0;

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += stride) {
            double sum = 0.0;
            int base_idx = i * cols + j;

            sum += grid[base_idx];
            sum += grid[(i-1) * cols + j];
            sum += grid[(i+1) * cols + j];
            sum += grid[i * cols + (j-1)];
            sum += grid[i * cols + (j+1)];

            new_grid[base_idx] = sum * weight;
        }
    }
}

// 2D 9-point Stencil 标量版本实现
void stencil2D_9point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                             int rows, int cols, int stride) {
    double weight = 1.0 / 9.0;

    for (int i = 1; i < rows - 1; i += stride) {
        for (int j = 1; j < cols - 1; j += stride) {
            double sum = 0.0;
            int base_idx = i * cols + j;

            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    sum += grid[(i + di) * cols + (j + dj)];
                }
            }

            new_grid[base_idx] = sum * weight;
        }
    }
}

// 3D 13-point Stencil 标量版本实现
void stencil3D_13point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 13.0;

    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += stride) {
                double sum = 0.0;
                int base_idx = k * plane_size + i * cols + j;

                sum += grid[base_idx];

                sum += grid[(k-1) * plane_size + i * cols + j];
                sum += grid[(k+1) * plane_size + i * cols + j];
                sum += grid[k * plane_size + (i-1) * cols + j];
                sum += grid[k * plane_size + (i+1) * cols + j];
                sum += grid[k * plane_size + i * cols + (j-1)];
                sum += grid[k * plane_size + i * cols + (j+1)];

                sum += grid[(k-1) * plane_size + (i-1) * cols + j];
                sum += grid[(k-1) * plane_size + (i+1) * cols + j];
                sum += grid[(k+1) * plane_size + (i-1) * cols + j];
                sum += grid[(k+1) * plane_size + (i+1) * cols + j];

                sum += grid[(k-1) * plane_size + i * cols + (j-1)];
                sum += grid[(k+1) * plane_size + i * cols + (j+1)];

                new_grid[base_idx] = sum * weight;
            }
        }
    }
}

// 3D 25-point Stencil 标量版本实现
void stencil3D_25point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 25.0;

    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += stride) {
                double sum = 0.0;
                int base_idx = k * plane_size + i * cols + j;

                sum += grid[base_idx];

                sum += grid[(k-1) * plane_size + i * cols + j];
                sum += grid[(k+1) * plane_size + i * cols + j];
                sum += grid[k * plane_size + (i-1) * cols + j];
                sum += grid[k * plane_size + (i+1) * cols + j];
                sum += grid[k * plane_size + i * cols + (j-1)];
                sum += grid[k * plane_size + i * cols + (j+1)];

                sum += grid[(k-1) * plane_size + (i-1) * cols + j];
                sum += grid[(k-1) * plane_size + (i+1) * cols + j];
                sum += grid[(k+1) * plane_size + (i-1) * cols + j];
                sum += grid[(k+1) * plane_size + (i+1) * cols + j];
                sum += grid[(k-1) * plane_size + i * cols + (j-1)];
                sum += grid[(k-1) * plane_size + i * cols + (j+1)];
                sum += grid[(k+1) * plane_size + i * cols + (j-1)];
                sum += grid[(k+1) * plane_size + i * cols + (j+1)];
                sum += grid[k * plane_size + (i-1) * cols + (j-1)];
                sum += grid[k * plane_size + (i-1) * cols + (j+1)];
                sum += grid[k * plane_size + (i+1) * cols + (j-1)];
                sum += grid[k * plane_size + (i+1) * cols + (j+1)];

                sum += grid[(k-1) * plane_size + (i-1) * cols + (j-1)];
                sum += grid[(k-1) * plane_size + (i-1) * cols + (j+1)];
                sum += grid[(k-1) * plane_size + (i+1) * cols + (j-1)];
                sum += grid[(k-1) * plane_size + (i+1) * cols + (j+1)];
                sum += grid[(k+1) * plane_size + (i-1) * cols + (j-1)];
                sum += grid[(k+1) * plane_size + (i-1) * cols + (j+1)];
                sum += grid[(k+1) * plane_size + (i+1) * cols + (j-1)];
                sum += grid[(k+1) * plane_size + (i+1) * cols + (j+1)];

                new_grid[base_idx] = sum * weight;
            }
        }
    }
}

// 3D 27-point Stencil 标量版本实现
void stencil3D_27point_scalar(double* __restrict__ grid, double* __restrict__ new_grid,
                              int depth, int rows, int cols, int stride) {
    int plane_size = rows * cols;
    double weight = 1.0 / 27.0;

    for (int k = 1; k < depth - 1; k += stride) {
        for (int i = 1; i < rows - 1; i += stride) {
            for (int j = 1; j < cols - 1; j += stride) {
                double sum = 0.0;
                int base_idx = k * plane_size + i * cols + j;

                for (int dk = -1; dk <= 1; dk++) {
                    for (int di = -1; di <= 1; di++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            sum += grid[(k + dk) * plane_size + (i + di) * cols + (j + dj)];
                        }
                    }
                }

                new_grid[base_idx] = sum * weight;
            }
        }
    }
}
