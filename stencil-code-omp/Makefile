CXX = clang++
CXXFLAGS = -fopenmp -O3 -mcpu=hip11
CXXFLAGS_OMP = $(CXXFLAGS)
CXXFLAGS_SME = $(CXXFLAGS)
CXXFLAGS_MAIN_OMP = $(CXXFLAGS_OMP) -DRUN_MAIN
CXXFLAGS_MAIN_SME = $(CXXFLAGS_SME) -DRUN_MAIN_SME
LDFLAGS = -fopenmp -lm

STENCILS_OMP = stencil_3d_27point_omp stencil_3d_13point_omp stencil_3d_25point_omp \
               stencil_2d_9point_omp stencil_2d_5point_omp stencil_1d_3point_omp

STENCILS_SME = stencil_3d_27point_sme stencil_3d_13point_sme stencil_3d_25point_sme \
               stencil_2d_9point_sme stencil_2d_5point_sme stencil_1d_3point_sme

all: $(STENCILS_OMP) $(STENCILS_SME)

# OpenMP versions
stencil_3d_27point_omp: stencil_3d_27point_omp.o stencil_3d_27point_main.o
	$(CXX) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS)

stencil_3d_13point_omp: stencil_3d_13point_omp.o stencil_3d_13point_main.o
	$(CXX) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS)

stencil_3d_25point_omp: stencil_3d_25point_omp.o stencil_3d_25point_main.o
	$(CXX) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS)

stencil_2d_9point_omp: stencil_2d_9point_omp.o stencil_2d_9point_main.o
	$(CXX) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS)

stencil_2d_5point_omp: stencil_2d_5point_omp.o stencil_2d_5point_main.o
	$(CXX) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS)

stencil_1d_3point_omp: stencil_1d_3point_omp.o stencil_1d_3point_main.o
	$(CXX) $(CXXFLAGS_OMP) -o $@ $^ $(LDFLAGS)

# SME versions
stencil_3d_27point_sme: stencil_3d_27point_omp.o stencil_3d_27point_main_sme.o
	$(CXX) $(CXXFLAGS_SME) -o $@ $^ $(LDFLAGS)

stencil_3d_13point_sme: stencil_3d_13point_omp.o stencil_3d_13point_main_sme.o
	$(CXX) $(CXXFLAGS_SME) -o $@ $^ $(LDFLAGS)

stencil_3d_25point_sme: stencil_3d_25point_omp.o stencil_3d_25point_main_sme.o
	$(CXX) $(CXXFLAGS_SME) -o $@ $^ $(LDFLAGS)

stencil_2d_9point_sme: stencil_2d_9point_omp.o stencil_2d_9point_main_sme.o
	$(CXX) $(CXXFLAGS_SME) -o $@ $^ $(LDFLAGS)

stencil_2d_5point_sme: stencil_2d_5point_omp.o stencil_2d_5point_main_sme.o
	$(CXX) $(CXXFLAGS_SME) -o $@ $^ $(LDFLAGS)

stencil_1d_3point_sme: stencil_1d_3point_omp.o stencil_1d_3point_main_sme.o
	$(CXX) $(CXXFLAGS_SME) -o $@ $^ $(LDFLAGS)

# Object files for OMP implementations
stencil_3d_27point_omp.o: stencil_3d_27point_omp.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

stencil_3d_13point_omp.o: stencil_3d_13point_omp.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

stencil_3d_25point_omp.o: stencil_3d_25point_omp.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

stencil_2d_9point_omp.o: stencil_2d_9point_omp.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

stencil_2d_5point_omp.o: stencil_2d_5point_omp.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

stencil_1d_3point_omp.o: stencil_1d_3point_omp.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS_OMP) -c $< -o $@

# Main files for OMP
stencil_3d_27point_main.o: stencil_3d_27point_main.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS_MAIN_OMP) -c $< -o $@

stencil_3d_13point_main.o: stencil_3d_13point_main.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS_MAIN_OMP) -c $< -o $@

stencil_3d_25point_main.o: stencil_3d_25point_main.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS_MAIN_OMP) -c $< -o $@

stencil_2d_9point_main.o: stencil_2d_9point_main.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS_MAIN_OMP) -c $< -o $@

stencil_2d_5point_main.o: stencil_2d_5point_main.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS_MAIN_OMP) -c $< -o $@

stencil_1d_3point_main.o: stencil_1d_3point_main.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS_MAIN_OMP) -c $< -o $@

# Main files for SME
stencil_3d_27point_main_sme.o: stencil_3d_27point_main_sme.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS_MAIN_SME) -c $< -o $@

stencil_3d_13point_main_sme.o: stencil_3d_13point_main_sme.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS_MAIN_SME) -c $< -o $@

stencil_3d_25point_main_sme.o: stencil_3d_25point_main_sme.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS_MAIN_SME) -c $< -o $@

stencil_2d_9point_main_sme.o: stencil_2d_9point_main_sme.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS_MAIN_SME) -c $< -o $@

stencil_2d_5point_main_sme.o: stencil_2d_5point_main_sme.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS_MAIN_SME) -c $< -o $@

stencil_1d_3point_main_sme.o: stencil_1d_3point_main_sme.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS_MAIN_SME) -c $< -o $@

clean:
	rm -f *.o $(STENCILS_OMP) $(STENCILS_SME)

.PHONY: all clean
