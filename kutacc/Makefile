# Makefile for FlashMLA BF16 32x64 Kernel (No Prefetch Version)
#
# This Makefile requires ARM cross-compilation toolchain with SVE/SME support
#

# Compiler - ARM Clang with SVE/SME support
ARM_CC = clang++
ARM_CFLAGS = -mcpu=hip11 -O3 -std=c++17 -w -fno-inline

# Include paths
INCDIRS = -I./include -I./src

# Source files
SRC = main.cpp

# Output
TARGET = flash_mla_test

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(ARM_CC) $(ARM_CFLAGS) $(INCDIRS) -o $@ $<

# Clean
clean:
	rm -f $(TARGET)

# Test (requires ARM hardware or QEMU)
test: $(TARGET)
	@echo "Running on ARM hardware (requires cross-compilation or QEMU)"
	./$(TARGET)

.PHONY: all clean test
