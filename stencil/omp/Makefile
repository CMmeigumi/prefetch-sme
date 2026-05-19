# Makefile for SME Stencil Variants (OpenMP)

CXX = clang++
CXXFLAGS = -fopenmp -O3 -mcpu=hip11 -rtlib=compiler-rt
CXXFLAGS_SME = $(CXXFLAGS) -D__ARM_FEATURE_SME -DRUN_MAIN
LDFLAGS = -fopenmp -lm -lgcc_s

.PHONY: all clean

all: stencil_all

stencil_all: stencil_all_sme

stencil_all_sme: stencil_all_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

clean:
	rm -f stencil_all_sme
