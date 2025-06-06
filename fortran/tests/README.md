# Memory Leak Testing Framework

Streamlined testing for Fortran memory leaks across multiple gfortran versions with both debug and optimized builds.

## Quick Start

```bash
# Test a Fortran file with all available gfortran versions
./test_memory.sh memory_leak_test.f90

# Test with specific version
./test_memory.sh camb_memory_leak_test.f90 13

# Test CAMB Python wrapper
python3 test_python_memory.py

# Test everything
make test-all
```

## Setup

```bash
# Install dependencies
make install-deps

# Or manually:
sudo apt update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt install -y valgrind gfortran-9 gfortran-10 gfortran-11 gfortran-12 gfortran-13
```

## Usage

### Fortran Testing

```bash
# Test any Fortran file
./test_memory.sh <fortran_file> [gfortran_version]

# Examples:
./test_memory.sh memory_leak_test.f90        # All versions
./test_memory.sh memory_leak_test.f90 13     # gfortran-13 only
```

### Python Testing

```bash
# Test CAMB Python wrapper
python3 test_python_memory.py [gfortran_version]

# Examples:
python3 test_python_memory.py               # All versions
python3 test_python_memory.py 12            # gfortran-12 only
```

### Makefile Targets

```bash
make test-memory FILE=memory_leak_test.f90   # Test specific file
make test-python VERSION=13                 # Test Python with gfortran-13
make test-all                               # Test everything
make test-docker                            # Test with latest Docker versions
make clean                                  # Clean up
```

## Build Types

Each test runs with both:
- **Debug build**: `-g -O0 -fbacktrace -fbounds-check -ffpe-trap=invalid,overflow,zero -Wall -Wextra`
- **Optimized build**: `-O3 -march=native -ffast-math`

## Docker Support

For latest compiler versions (GCC 15.1.1):
```bash
# Automatic with test scripts
./test_memory.sh memory_leak_test.f90        # Includes Docker testing

# Manual Docker test
make test-docker
```

## Test Files

- `memory_leak_test.f90` - Simple allocatable array test
- `camb_memory_leak_test.f90` - Complex nested derived types (CAMB-style)

## Results Interpretation

- ✅ **No memory leaks** - All memory properly freed
- ⚠️ **Possible memory leak** - Might be false positive
- ❌ **Memory leak detected** - Definite leak requiring attention

## Adding New Tests

1. Create your `.f90` file in this directory
2. Run: `./test_memory.sh your_file.f90`
3. Or use: `make test-memory FILE=your_file.f90`

The framework automatically tests with all available gfortran versions and both debug/optimized builds.
