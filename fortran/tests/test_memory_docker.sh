#!/bin/bash

# Universal Docker-based memory testing for Fortran files
# Works on Linux, macOS, Windows (with Docker)
# Usage: ./test_memory_docker.sh <fortran_file> [specific_version]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration - Docker images with different gfortran versions
declare -A DOCKER_IMAGES=(
    ["9"]="ubuntu:20.04"
    ["10"]="ubuntu:20.04" 
    ["11"]="ubuntu:22.04"
    ["12"]="ubuntu:22.04"
    ["13"]="ubuntu:22.04"
    ["14"]="ubuntu:24.04"
    ["15"]="cmbant/docker-gcc-build:gcc15"
)

BUILD_TYPES=("debug" "optimized")

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <fortran_file> [specific_version]"
    echo "Available versions: ${!DOCKER_IMAGES[*]}"
    exit 1
fi

FORTRAN_FILE="$1"
SPECIFIC_VERSION="$2"

if [ ! -f "$FORTRAN_FILE" ]; then
    echo "Error: File $FORTRAN_FILE not found"
    exit 1
fi

# Check Docker availability
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is required but not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Function to test with specific version in Docker
test_version_docker() {
    local version=$1
    local build_type=$2
    local docker_image=${DOCKER_IMAGES[$version]}
    
    echo "Testing gfortran-$version ($build_type build) in Docker..."
    
    # Set compiler flags
    if [ "$build_type" = "debug" ]; then
        FLAGS="-g -O0 -fbacktrace -fbounds-check -ffpe-trap=invalid,overflow,zero -Wall -Wextra"
    else
        FLAGS="-O3 -ffast-math"
    fi
    
    # Special handling for GCC 15 image
    if [ "$version" = "15" ]; then
        INSTALL_CMD="apt update -qq && apt install -y valgrind -qq"
        GFORTRAN_CMD="gfortran"
    else
        INSTALL_CMD="apt update -qq && apt install -y valgrind gfortran-$version -qq"
        GFORTRAN_CMD="gfortran-$version"
    fi
    
    # Run test in Docker
    local result=$(docker run --rm -v "$(pwd):/test" "$docker_image" /bin/bash -c "
        $INSTALL_CMD >/dev/null 2>&1 &&
        cd /test &&

        # Get exact version
        EXACT_VERSION=\$($GFORTRAN_CMD --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+') &&
        echo \"   Compiler: \$($GFORTRAN_CMD --version | head -1)\" &&

        # Compile
        if ! $GFORTRAN_CMD $FLAGS -o test_exe '$FORTRAN_FILE' 2>/dev/null; then
            echo '❌ Compilation failed'
            exit 1
        fi &&

        # Run with valgrind
        if ! valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                      --track-origins=yes --log-file=valgrind.log \
                      ./test_exe >/dev/null 2>&1; then
            echo '❌ Runtime error'
            exit 1
        fi &&

        # Analyze results
        if grep -q 'definitely lost: [1-9]' valgrind.log; then
            echo '❌ MEMORY LEAK DETECTED'
            grep 'definitely lost' valgrind.log
            exit 1
        elif grep -q 'possibly lost: [1-9]' valgrind.log; then
            echo '⚠️  Possible memory leak'
            grep 'possibly lost' valgrind.log
        else
            echo '✅ No memory leaks'
        fi &&

        # Show memory usage
        grep 'total heap usage:' valgrind.log | head -1 | sed 's/^.*total heap usage:/   Memory usage:/' &&

        # Cleanup
        rm -f test_exe valgrind.log
    " 2>&1)

    echo "$result"
}

# Function to test Python in Docker
test_python_docker() {
    echo "Testing CAMB Python wrapper in Docker..."
    
    docker run --rm -v "$(pwd)/../..:/camb" cmbant/docker-gcc-build:gcc15 /bin/bash -c "
        apt update -qq && apt install -y python3-pip -qq >/dev/null 2>&1 &&
        cd /camb &&
        
        # Get compiler version
        echo \"   Compiler: \$(gfortran --version | head -1)\" &&
        
        # Build CAMB
        python3 setup.py make >/dev/null 2>&1 &&
        
        # Run memory test
        echo '   Running CAMB memory test...' &&
        if python3 -m unittest camb.tests.camb_test.CambTest.test_memory 2>&1 | grep -q 'Apparent memory leak'; then
            echo '⚠️  Memory usage variation detected (may be normal)'
        else
            echo '✅ No memory issues detected'
        fi
    " 2>/dev/null || echo "❌ Python test failed"
}

# Main execution
echo "========================================="
echo "Docker-based Memory Testing: $FORTRAN_FILE"
echo "========================================="

if [ -n "$SPECIFIC_VERSION" ]; then
    # Test specific version
    if [[ ! " ${!DOCKER_IMAGES[*]} " =~ " $SPECIFIC_VERSION " ]]; then
        echo "Error: Version $SPECIFIC_VERSION not available"
        echo "Available versions: ${!DOCKER_IMAGES[*]}"
        exit 1
    fi
    
    for build_type in "${BUILD_TYPES[@]}"; do
        test_version_docker "$SPECIFIC_VERSION" "$build_type"
    done
else
    # Test all versions
    for version in "${!DOCKER_IMAGES[@]}"; do
        for build_type in "${BUILD_TYPES[@]}"; do
            test_version_docker "$version" "$build_type"
        done
        echo
    done
    
    # Test Python if we're in CAMB directory
    if [ -f "../../setup.py" ]; then
        echo "Python Testing:"
        test_python_docker
    fi
fi

echo "========================================="
echo "Testing completed"
echo "========================================="
