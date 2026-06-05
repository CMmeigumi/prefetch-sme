# Makefile for FlashMLA BF16 32x64 Kernel
#
# This Makefile requires ARM cross-compilation toolchain with SVE/SME support
#

# Compiler - ARM Clang with SVE/SME support
ARM_CC = clang++
ARM_CFLAGS = -mcpu=hip11 -O3 -std=c++17

# Include paths
INCDIRS = -I./include -I./src

# Source files
SRC = main.cpp
SRC_NOPF = main_nopf.cpp

# Output
TARGET = flash_mla_test
TARGET_NOPF = flash_mla_test_nopf

# Default target
all: $(TARGET) $(TARGET_NOPF)

$(TARGET): $(SRC)
	$(ARM_CC) $(ARM_CFLAGS) $(INCDIRS) -o $@ $<

$(TARGET_NOPF): $(SRC_NOPF)
	$(ARM_CC) $(ARM_CFLAGS) $(INCDIRS) -o $@ $<

# Build individual targets
with_prefetch: $(TARGET)
	@echo "Built: $(TARGET)"

without_prefetch: $(TARGET_NOPF)
	@echo "Built: $(TARGET_NOPF)"

# Clean
clean:
	rm -f $(TARGET) $(TARGET_NOPF)

# Test (requires ARM hardware or QEMU)
test: $(TARGET)
	@echo "Running on ARM hardware (requires cross-compilation or QEMU)"
	./$(TARGET)

.PHONY: all clean test
