# Makefile for 3D 27-point Stencil

CXX = clang++
CXXFLAGS = -O3 -mcpu=hip11 -std=c++17 -rtlib=compiler-rt
LDFLAGS = -lm

# Target for SME build (requires ARM compiler with SME support)
TARGET_SME = stencil_3d_27point_sme
TARGET_SCALAR = stencil_3d_27point_scalar
TARGET_TEST = stencil_3d_27point_test

# Source files
SOURCES_SME = stencil_3d_27point_sme.cpp
SOURCES_SCALAR = stencil_3d_27point_scalar.cpp
SOURCES_TEST = stencil_3d_27point_test.cpp stencil_3d_27point_sme.cpp stencil_3d_27point_scalar.cpp

# Header files
HEADERS = stencil_3d_27point.h

# Object files
OBJ_SME = stencil_3d_27point_sme.o
OBJ_SCALAR = stencil_3d_27point_scalar.o
OBJS_TEST = stencil_3d_27point_test.o stencil_3d_27point_sme.o stencil_3d_27point_scalar.o

.PHONY: all clean test_scalar test_sme test

all: $(TARGET_SCALAR) $(TARGET_SME) $(TARGET_TEST)

$(TARGET_SCALAR): $(OBJ_SCALAR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_SME): $(OBJ_SME)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_TEST): $(OBJS_TEST)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test_scalar: $(TARGET_SCALAR)
	./$(TARGET_SCALAR)

test_sme: $(TARGET_SME)
	./$(TARGET_SME)

test: $(TARGET_TEST)
	./$(TARGET_TEST)

clean:
	rm -f $(TARGET_SCALAR) $(TARGET_SME) $(TARGET_TEST) *.o
