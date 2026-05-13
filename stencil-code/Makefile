# Makefile for All Stencil Variants
# Supports: 3D-27, 3D-13, 3D-25, 2D-9, 2D-5, 1D-3

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
LDFLAGS = -lm -lgcc_s

STENCILS = 3d_27 3d_13 3d_25 2d_9 2d_5 1d_3

.PHONY: all clean test_all sme_all $(STENCILS)

all: $(STENCILS) sme_all

sme_all: sme_3d_27 sme_3d_13 sme_3d_25 sme_2d_9 sme_2d_5 sme_1d_3

# 3D-27 Point
3d_27: stencil_3d_27point_test
stencil_3d_27point_test: stencil_3d_27point_test.o stencil_3d_27point_scalar.o stencil_3d_27point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sme_3d_27: CXXFLAGS_SME = $(CXXFLAGS) -DRUN_MAIN
sme_3d_27: stencil_3d_27point_sme_main.o
	$(CXX) $(CXXFLAGS) -o sme_3d_27 stencil_3d_27point_sme_main.o $(LDFLAGS)

stencil_3d_27point_sme_main.o: stencil_3d_27point_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -c -o $@ $<

# 3D-13 Point
3d_13: stencil_3d_13point_test
stencil_3d_13point_test: stencil_3d_13point_test.o stencil_3d_13point_scalar.o stencil_3d_13point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sme_3d_13: CXXFLAGS_SME = $(CXXFLAGS) -DRUN_MAIN
sme_3d_13: stencil_3d_13point_sme_main.o
	$(CXX) $(CXXFLAGS) -o sme_3d_13 stencil_3d_13point_sme_main.o $(LDFLAGS)

stencil_3d_13point_sme_main.o: stencil_3d_13point_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -c -o $@ $<

# 3D-25 Point
3d_25: stencil_3d_25point_test
stencil_3d_25point_test: stencil_3d_25point_test.o stencil_3d_25point_scalar.o stencil_3d_25point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sme_3d_25: CXXFLAGS_SME = $(CXXFLAGS) -DRUN_MAIN
sme_3d_25: stencil_3d_25point_sme_main.o
	$(CXX) $(CXXFLAGS) -o sme_3d_25 stencil_3d_25point_sme_main.o $(LDFLAGS)

stencil_3d_25point_sme_main.o: stencil_3d_25point_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -c -o $@ $<

# 2D-9 Point
2d_9: stencil_2d_9point_test
stencil_2d_9point_test: stencil_2d_9point_test.o stencil_2d_9point_scalar.o stencil_2d_9point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sme_2d_9: CXXFLAGS_SME = $(CXXFLAGS) -DRUN_MAIN
sme_2d_9: stencil_2d_9point_sme_main.o
	$(CXX) $(CXXFLAGS) -o sme_2d_9 stencil_2d_9point_sme_main.o $(LDFLAGS)

stencil_2d_9point_sme_main.o: stencil_2d_9point_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -c -o $@ $<

# 2D-5 Point
2d_5: stencil_2d_5point_test
stencil_2d_5point_test: stencil_2d_5point_test.o stencil_2d_5point_scalar.o stencil_2d_5point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sme_2d_5: CXXFLAGS_SME = $(CXXFLAGS) -DRUN_MAIN
sme_2d_5: stencil_2d_5point_sme_main.o
	$(CXX) $(CXXFLAGS) -o sme_2d_5 stencil_2d_5point_sme_main.o $(LDFLAGS)

stencil_2d_5point_sme_main.o: stencil_2d_5point_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -c -o $@ $<

# 1D-3 Point
1d_3: stencil_1d_3point_test
stencil_1d_3point_test: stencil_1d_3point_test.o stencil_1d_3point_scalar.o stencil_1d_3point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sme_1d_3: CXXFLAGS_SME = $(CXXFLAGS) -DRUN_MAIN
sme_1d_3: stencil_1d_3point_sme_main.o
	$(CXX) $(CXXFLAGS) -o sme_1d_3 stencil_1d_3point_sme_main.o $(LDFLAGS)

stencil_1d_3point_sme_main.o: stencil_1d_3point_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -c -o $@ $<

# Object files (for test builds)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Test targets
test_all: $(STENCILS)
	@echo "Running all tests..."
	@echo ""
	@echo "=== 3D-27 Point ==="
	-./stencil_3d_27point_test
	@echo ""
	@echo "=== 3D-13 Point ==="
	-./stencil_3d_13point_test
	@echo ""
	@echo "=== 3D-25 Point ==="
	-./stencil_3d_25point_test
	@echo ""
	@echo "=== 2D-9 Point ==="
	-./stencil_2d_9point_test
	@echo ""
	@echo "=== 2D-5 Point ==="
	-./stencil_2d_5point_test
	@echo ""
	@echo "=== 1D-3 Point ==="
	-./stencil_1d_3point_test

clean:
	rm -f *.o stencil_3d_27point_test stencil_3d_13point_test stencil_3d_25point_test stencil_2d_9point_test stencil_2d_5point_test stencil_1d_3point_test sme_3d_27 sme_3d_13 sme_3d_25 sme_2d_9 sme_2d_5 sme_1d_3
