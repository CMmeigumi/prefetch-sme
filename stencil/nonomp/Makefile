# Makefile for SME Stencil Variants

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
CXXFLAGS_SME = $(CXXFLAGS) -D__ARM_FEATURE_SME
LDFLAGS = -lm -lgcc_s

.PHONY: all clean sme_all stencil_all

all: stencil_all

stencil_all: stencil_all_sme

stencil_all_sme: stencil_all_sme.cpp
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

sme_all: stencil_3d_27point_sme stencil_3d_13point_sme stencil_3d_25point_sme \
         stencil_2d_9point_sme stencil_2d_5point_sme stencil_1d_3point_sme

stencil_3d_27point_sme: stencil_3d_27point_sme.cpp stencil_3d_27point.h
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

stencil_3d_13point_sme: stencil_3d_13point_sme.cpp stencil_3d_13point.h
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

stencil_3d_25point_sme: stencil_3d_25point_sme.cpp stencil_3d_25point.h
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

stencil_2d_9point_sme: stencil_2d_9point_sme.cpp stencil_2d_9point.h
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

stencil_2d_5point_sme: stencil_2d_5point_sme.cpp stencil_2d_5point.h
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

stencil_1d_3point_sme: stencil_1d_3point_sme.cpp stencil_1d_3point.h
	$(CXX) $(CXXFLAGS_SME) -o $@ $< $(LDFLAGS)

clean:
	rm -f stencil_all_sme stencil_3d_27point_sme stencil_3d_13point_sme stencil_3d_25point_sme \
	      stencil_2d_9point_sme stencil_2d_5point_sme stencil_1d_3point_sme
