# Streamlined Memory Testing Framework

## Files Created:
- `test_memory.sh` - Universal Fortran memory tester
- `test_python_memory.py` - CAMB Python wrapper tester  
- `Makefile` - Easy-to-use build targets
- `README.md` - Complete usage documentation
- `memory_leak_test.f90` - Simple test case
- `camb_memory_leak_test.f90` - CAMB-style complex test

## Quick Usage:

```bash
# Test any Fortran file
./test_memory.sh your_file.f90

# Test with specific version  
./test_memory.sh your_file.f90 13

# Test CAMB Python wrapper
python3 test_python_memory.py

# Using Makefile
make test-memory FILE=your_file.f90
make test-python
make test-all
```

## Features:
- Tests gfortran 9-15 (including latest via Docker)
- Both debug and optimized builds
- Automatic valgrind analysis  
- Clean, concise output
- Easy to extend for new test files

## Test Results Summary:
✅ **Fortran patterns show NO MEMORY LEAKS** across gfortran 9-15
- Simple allocatable patterns: SAFE
- Complex CAMB-style nested types: SAFE
- Both debug and optimized builds: SAFE
- Assignment operations: SAFE
- Large memory allocations: SAFE

⚠️ **Python wrapper**: Uses CAMB's built-in memory monitoring
- Detects memory usage patterns between iterations
- Small variations may be normal system behavior
- No valgrind needed (inappropriate for Python)
