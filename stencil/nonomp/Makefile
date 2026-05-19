# Makefile for SME Stencil Variants

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
CXXFLAGS_SME = $(CXXFLAGS) -D__ARM_FEATURE_SME
LDFLAGS = -lm -lgcc_s

.PHONY: all clean

all: stencil_all

stencil_all: stencil_all_sme

stencil_all_sme: stencil_all_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

clean:
	rm -f stencil_all_sme
