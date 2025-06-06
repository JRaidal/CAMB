# Memory Testing Makefile
.PHONY: help test-memory test-docker test-python test-all clean

help:
	@echo "Memory Testing Makefile"
	@echo "======================"
	@echo "  test-memory FILE=<file> [VERSION=<ver>]  - Test locally (requires setup)"
	@echo "  test-docker FILE=<file> [VERSION=<ver>]  - Test in Docker (recommended)"
	@echo "  test-python                              - Test CAMB Python wrapper"
	@echo "  test-all                                 - Test all patterns"
	@echo "  clean                                    - Clean up"
	@echo ""
	@echo "Examples:"
	@echo "  make test-docker FILE=memory_leak_test.f90"
	@echo "  make test-memory FILE=camb_memory_leak_test.f90 VERSION=13"
	@echo "  make test-all"

# Local testing
test-memory:
ifndef FILE
	@echo "Error: FILE parameter required"
	@exit 1
endif
ifdef VERSION
	./test_memory.sh $(FILE) $(VERSION)
else
	./test_memory.sh $(FILE)
endif

# Docker testing (recommended)
test-docker:
ifndef FILE
	@echo "Error: FILE parameter required"
	@exit 1
endif
ifdef VERSION
	./test_memory_docker.sh $(FILE) $(VERSION)
else
	./test_memory_docker.sh $(FILE)
endif

# Test CAMB Python wrapper
test-python:
	cd ../.. && python3 fortran/tests/test_python_memory.py

# Test all patterns (using Docker)
test-all:
	@echo "Testing all Fortran patterns with Docker..."
	@for file in *.f90; do \
		if [ -f "$$file" ]; then \
			echo "Testing $$file..."; \
			./test_memory_docker.sh "$$file" || true; \
			echo ""; \
		fi; \
	done
	@echo "Testing Python..."
	@$(MAKE) test-python

# Clean up
clean:
	@echo "Cleaning up..."
	rm -f *.log valgrind_* memory_leak_test* camb_memory_leak_test*
	rm -f *.o *.mod *.so
	@echo "Cleanup complete."
