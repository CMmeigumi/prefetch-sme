CXX = clang++
CXXFLAGS = -fopenmp -O3 -mcpu=hip11 -rtlib=compiler-rt -lgcc_s
CXXFLAGS_MAIN = $(CXXFLAGS) -DRUN_MAIN
LDFLAGS = -fopenmp -lm

STENCILS_OMP = stencil_3d_27point_omp stencil_3d_13point_omp stencil_3d_25point_omp \
               stencil_2d_9point_omp stencil_2d_5point_omp stencil_1d_3point_omp

all: $(STENCILS_OMP)

stencil_3d_27point_omp: stencil_3d_27point_omp.o stencil_3d_27point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

stencil_3d_13point_omp: stencil_3d_13point_omp.o stencil_3d_13point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

stencil_3d_25point_omp: stencil_3d_25point_omp.o stencil_3d_25point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

stencil_2d_9point_omp: stencil_2d_9point_omp.o stencil_2d_9point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

stencil_2d_5point_omp: stencil_2d_5point_omp.o stencil_2d_5point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

stencil_1d_3point_omp: stencil_1d_3point_omp.o stencil_1d_3point_main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

stencil_3d_27point_omp.o: stencil_3d_27point_omp.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_3d_13point_omp.o: stencil_3d_13point_omp.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_3d_25point_omp.o: stencil_3d_25point_omp.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_2d_9point_omp.o: stencil_2d_9point_omp.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_2d_5point_omp.o: stencil_2d_5point_omp.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

stencil_1d_3point_omp.o: stencil_1d_3point_omp.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

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
	rm -f *.o $(STENCILS_OMP)

.PHONY: all clean
