# Makefile for FFTW Test Cases

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
LDFLAGS = -lm -lgcc_s -lfftw3

.PHONY: all clean

all: fft_1d fft_2d fft_real

fft_1d: fft_1d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft_2d: fft_2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft_real: fft_real.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f fft_1d fft_2d fft_real
