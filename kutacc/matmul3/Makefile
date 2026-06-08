CXX = clang++
CXXFLAGS = -mcpu=hip11 -O3 -std=c++17 -w -fno-inline -rtlib=compiler-rt
LDFLAGS = -mcpu=hip11 -rtlib=compiler-rt
INC = -Iinclude

all: bgemm_test_pf

bgemm_test_pf: main_pf.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $@ main_pf.cpp $(LDFLAGS)

clean:
	rm -f bgemm_test_pf

.PHONY: all clean