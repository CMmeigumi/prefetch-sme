CXX = clang++
CXXFLAGS = -mcpu=hip11 -O3 -std=c++17 -w -fno-inline -rtlib=compiler-rt
LDFLAGS = -mcpu=hip11 -rtlib=compiler-rt
INC = -Iinclude

all: bgemm_test_32

bgemm_test_32: main.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $@ main.cpp $(LDFLAGS)

clean:
	rm -f bgemm_test_32

.PHONY: all clean