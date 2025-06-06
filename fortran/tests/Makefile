# Streamlined Memory Testing Makefile
# Usage: make test-memory FILE=<fortran_file> [VERSION=<version>]

.PHONY: help test-memory test-python test-all clean install-deps

# Default target
help:
	@echo "Memory Testing Makefile"
	@echo "======================"
	@echo "Targets:"
	@echo "  test-memory FILE=<file> [VERSION=<ver>]  - Test Fortran file for memory leaks"
	@echo "  test-python [VERSION=<ver>]              - Test CAMB Python wrapper"
	@echo "  test-all                                 - Test all patterns with all versions"
	@echo "  install-deps                             - Install required dependencies"
	@echo "  clean                                    - Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make test-memory FILE=memory_leak_test.f90"
	@echo "  make test-memory FILE=camb_memory_leak_test.f90 VERSION=13"
	@echo "  make test-python VERSION=12"
	@echo "  make test-all"

# Test specific Fortran file
test-memory:
ifndef FILE
	@echo "Error: FILE parameter required"
	@echo "Usage: make test-memory FILE=<fortran_file> [VERSION=<version>]"
	@exit 1
endif
	@if [ ! -f "$(FILE)" ]; then echo "Error: File $(FILE) not found"; exit 1; fi
ifdef VERSION
	./test_memory.sh $(FILE) $(VERSION)
else
	./test_memory.sh $(FILE)
endif

# Test CAMB Python wrapper
test-python:
	cd ../.. && python3 fortran/tests/test_python_memory.py

# Test all patterns
test-all: test-memory-all test-python

test-memory-all:
	@echo "Testing all Fortran patterns..."
	@for file in *.f90; do \
		if [ -f "$$file" ]; then \
			echo "Testing $$file..."; \
			./test_memory.sh "$$file" || true; \
			echo ""; \
		fi; \
	done

# Install dependencies
install-deps:
	@echo "Installing dependencies..."
	sudo apt update
	sudo apt install -y valgrind
	@echo "Installing additional gfortran versions..."
	sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y || true
	sudo apt update
	sudo apt install -y gfortran-9 gfortran-10 gfortran-11 gfortran-12 gfortran-13 || true
	@echo "Dependencies installed."

# Clean up
clean:
	@echo "Cleaning up..."
	rm -f *.log valgrind_* memory_leak_test* camb_memory_leak_test*
	rm -f *.o *.mod *.so
	@echo "Cleanup complete."

# Quick tests for development
quick-test:
	@if [ -f "memory_leak_test.f90" ]; then \
		./test_memory.sh memory_leak_test.f90 13; \
	else \
		echo "No test file found"; \
	fi

# Docker test (latest versions)
test-docker:
	@echo "Testing with latest Docker versions..."
	@if command -v docker >/dev/null 2>&1; then \
		for file in *.f90; do \
			if [ -f "$$file" ]; then \
				echo "Testing $$file with GCC 15.1.1..."; \
				docker run --rm -v $(PWD):/test cmbant/docker-gcc-build:gcc15 /bin/bash -c \
				"apt update -qq && apt install -y valgrind -qq >/dev/null 2>&1 && cd /test && \
				 gfortran -O3 -o test_exe $$file && \
				 valgrind --tool=memcheck --leak-check=full --log-file=docker.log ./test_exe >/dev/null 2>&1 && \
				 if grep -q 'definitely lost: [1-9]' docker.log; then echo '❌ LEAK'; else echo '✅ OK'; fi && \
				 rm -f test_exe docker.log"; \
			fi; \
		done; \
	else \
		echo "Docker not available"; \
	fi
