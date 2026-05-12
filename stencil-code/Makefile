# Makefile for All Stencil Variants
# Supports: 3D-27, 3D-13, 3D-25, 2D-9, 2D-5, 1D-3

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
LDFLAGS = -lm

# All stencil variants
STENCILS = 3d_27 3d_13 3d_25 2d_9 2d_5 1d_3

.PHONY: all clean test_all $(STENCILS)

all: $(STENCILS)

# 3D-27 Point
3d_27: stencil_3d_27point_test
stencil_3d_27point_test: stencil_3d_27point_test.o stencil_3d_27point_scalar.o stencil_3d_27point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 3D-13 Point
3d_13: stencil_3d_13point_test
stencil_3d_13point_test: stencil_3d_13point_test.o stencil_3d_13point_scalar.o stencil_3d_13point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 3D-25 Point
3d_25: stencil_3d_25point_test
stencil_3d_25point_test: stencil_3d_25point_test.o stencil_3d_25point_scalar.o stencil_3d_25point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 2D-9 Point
2d_9: stencil_2d_9point_test
stencil_2d_9point_test: stencil_2d_9point_test.o stencil_2d_9point_scalar.o stencil_2d_9point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 2D-5 Point
2d_5: stencil_2d_5point_test
stencil_2d_5point_test: stencil_2d_5point_test.o stencil_2d_5point_scalar.o stencil_2d_5point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 1D-3 Point
1d_3: stencil_1d_3point_test
stencil_1d_3point_test: stencil_1d_3point_test.o stencil_1d_3point_scalar.o stencil_1d_3point_sme.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Object files
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

test_3d_27: stencil_3d_27point_test
	./stencil_3d_27point_test

test_3d_13: stencil_3d_13point_test
	./stencil_3d_13point_test

test_3d_25: stencil_3d_25point_test
	./stencil_3d_25point_test

test_2d_9: stencil_2d_9point_test
	./stencil_2d_9point_test

test_2d_5: stencil_2d_5point_test
	./stencil_2d_5point_test

test_1d_3: stencil_1d_3point_test
	./stencil_1d_3point_test

clean:
	rm -f stencil_*_test *.o
