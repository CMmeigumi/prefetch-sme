# Makefile for FFTW Test Cases

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
LDFLAGS = -lm -lgcc_s -lfftw3

.PHONY: all clean

all: fft_1d fft_2d fft_real fft3_1d fft3_2d fft3_real fft3_cm

fft_1d: fft_1d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft_2d: fft_2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft_real: fft_real.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft3_1d: fft3_1d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft3_2d: fft3_2d.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft3_real: fft3_real.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

fft3_cm: fft3_cm.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f fft_1d fft_2d fft_real fft3_1d fft3_2d fft3_real fft3_cm
