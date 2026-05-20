# Makefile for SME and Scalar Stencil Variants (OpenMP)

CXX = clang++
CXXFLAGS = -fopenmp -O3 -mcpu=hip11 -rtlib=compiler-rt
CXXFLAGS_SME = $(CXXFLAGS) -D__ARM_FEATURE_SME -DRUN_MAIN
LDFLAGS = -fopenmp -lm -lgcc_s

.PHONY: all clean

all: stencil_all stencil_scalar

stencil_all: stencil_all_sme

stencil_all_sme: stencil_all_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

stencil_scalar: stencil_all_scalar

stencil_all_scalar: stencil_all_scalar.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f stencil_all_sme stencil_all_scalar
