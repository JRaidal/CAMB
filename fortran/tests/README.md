# Memory Leak Testing Framework

Cross-platform memory testing for Fortran code across multiple gfortran versions.

## Quick Start

```bash
# Docker-based (works everywhere, recommended)
./test_memory_docker.sh memory_leak_test.f90

# Local testing (Linux/macOS only)
./test_memory.sh memory_leak_test.f90

# Test CAMB Python wrapper
python3 test_python_memory.py

# Test everything
make test-all
```

## Methods

### 1. Docker Testing (Recommended)
- **Cross-platform**: Works on Linux, macOS, Windows
- **No local setup**: Uses containerized environments
- **Exact versions**: Tests gfortran 9.x.x through 15.x.x
- **Clean environment**: No interference with local system

```bash
./test_memory_docker.sh <file.f90> [version]
```

### 2. Local Testing
- **Fast**: No Docker overhead
- **Requires setup**: Need local gfortran versions + valgrind

```bash
./test_memory.sh <file.f90> [version]
```

### 3. Python Testing
- **CAMB-specific**: Uses built-in memory monitoring
- **No valgrind**: Monitors memory patterns between iterations

```bash
python3 test_python_memory.py
```

## Setup (Local Testing Only)

```bash
# Install dependencies
sudo apt update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt install -y valgrind gfortran-{9,10,11,12,13}
```

## Test Files Included

- `memory_leak_test.f90` - Simple allocatable array pattern
- `camb_memory_leak_test.f90` - Complex CAMB-style nested types

## Results

**Fortran Testing**: ✅ NO MEMORY LEAKS detected across all gfortran versions 9-15
- Simple patterns: SAFE
- Complex nested types: SAFE
- Debug/optimized builds: SAFE

**Python Testing**: ⚠️ Small memory variations (normal for Python interpreter)

## Adding Tests

1. Create `your_test.f90` in this directory
2. Run `./test_memory_docker.sh your_test.f90`

Both debug (`-g -O0 -fbounds-check`) and optimized (`-O3`) builds are tested automatically.
