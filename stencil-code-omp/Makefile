CXX = clang++
CXXFLAGS = -fopenmp -O3 -mcpu=hip11 -rtlib=compiler-rt
CXXFLAGS_MAIN = $(CXXFLAGS) -DRUN_MAIN
LDFLAGS = -fopenmp -lm -lgcc_s

STENCILS = stencil_3d_27point stencil_3d_13point stencil_3d_25point \
           stencil_2d_9point stencil_2d_5point stencil_1d_3point

all: $(STENCILS)

# 3D 27-point
stencil_3d_27point: stencil_3d_27point_scalar.o stencil_3d_27point_sme.o stencil_3d_27point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 3D 13-point
stencil_3d_13point: stencil_3d_13point_scalar.o stencil_3d_13point_sme.o stencil_3d_13point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 3D 25-point
stencil_3d_25point: stencil_3d_25point_scalar.o stencil_3d_25point_sme.o stencil_3d_25point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 2D 9-point
stencil_2d_9point: stencil_2d_9point_scalar.o stencil_2d_9point_sme.o stencil_2d_9point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 2D 5-point
stencil_2d_5point: stencil_2d_5point_scalar.o stencil_2d_5point_sme.o stencil_2d_5point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# 1D 3-point
stencil_1d_3point: stencil_1d_3point_scalar.o stencil_1d_3point_sme.o stencil_1d_3point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Scalar object files
stencil_3d_27point_scalar.o: stencil_3d_27point_scalar.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_3d_13point_scalar.o: stencil_3d_13point_scalar.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_3d_25point_scalar.o: stencil_3d_25point_scalar.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_2d_9point_scalar.o: stencil_2d_9point_scalar.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_2d_5point_scalar.o: stencil_2d_5point_scalar.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_1d_3point_scalar.o: stencil_1d_3point_scalar.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# SME object files
stencil_3d_27point_sme.o: stencil_3d_27point_sme.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_3d_13point_sme.o: stencil_3d_13point_sme.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_3d_25point_sme.o: stencil_3d_25point_sme.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_2d_9point_sme.o: stencil_2d_9point_sme.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_2d_5point_sme.o: stencil_2d_5point_sme.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_1d_3point_sme.o: stencil_1d_3point_sme.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main object files
stencil_3d_27point_main.o: stencil_3d_27point_main.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS_MAIN) -c $< -o $@

stencil_3d_13point_main.o: stencil_3d_13point_main.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS_MAIN) -c $< -o $@

stencil_3d_25point_main.o: stencil_3d_25point_main.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS_MAIN) -c $< -o $@

stencil_2d_9point_main.o: stencil_2d_9point_main.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS_MAIN) -c $< -o $@

stencil_2d_5point_main.o: stencil_2d_5point_main.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS_MAIN) -c $< -o $@

stencil_1d_3point_main.o: stencil_1d_3point_main.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS_MAIN) -c $< -o $@

clean:
	rm -f *.o $(STENCILS)

.PHONY: all clean
