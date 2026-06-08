CXX = clang++
CXXFLAGS = -mcpu=hip11 -O3 -std=c++17 -w -fno-inline
LDFLAGS = -mcpu=hip11
INC = -Iinclude

all: bgemm_test bgemm_test_pf

bgemm_test: main.cpp include/kutacc_matmul.h bgemm_woqs8_kernel.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ main.cpp $(LDFLAGS)

bgemm_test_pf: main_pf.cpp include/kutacc_matmul.h bgemm_woqs8_kernel_pf.h
	$(CXX) $(CXXFLAGS) $(INC) -o $@ main_pf.cpp $(LDFLAGS)

nopf: bgemm_test

pf: bgemm_test_pf

clean:
	rm -f bgemm_test bgemm_test_pf

.PHONY: all nopf pf clean
