#!/bin/bash

# Universal memory leak tester for Fortran files
# Usage: ./test_memory.sh <fortran_file> [gfortran_version]
# Example: ./test_memory.sh memory_leak_test.f90 13

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
GFORTRAN_VERSIONS=(9 10 11 12 13)  # Add more as needed
BUILD_TYPES=("debug" "optimized")

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <fortran_file> [gfortran_version]"
    echo "Example: $0 memory_leak_test.f90 13"
    echo "Available versions: ${GFORTRAN_VERSIONS[*]}"
    exit 1
fi

FORTRAN_FILE="$1"
SPECIFIC_VERSION="$2"

if [ ! -f "$FORTRAN_FILE" ]; then
    echo "Error: File $FORTRAN_FILE not found"
    exit 1
fi

# Function to test with specific version and build type
test_version() {
    local version=$1
    local build_type=$2
    local gfortran_cmd="gfortran-$version"
    
    if [ "$version" = "default" ]; then
        gfortran_cmd="gfortran"
    fi
    
    if ! command -v "$gfortran_cmd" >/dev/null 2>&1; then
        echo "⚠️  $gfortran_cmd not available, skipping..."
        return 0
    fi
    
    echo "Testing $gfortran_cmd ($build_type build)..."
    
    # Set compiler flags based on build type
    if [ "$build_type" = "debug" ]; then
        FLAGS="-g -O0 -fbacktrace -fbounds-check -ffpe-trap=invalid,overflow,zero -Wall -Wextra"
    else
        FLAGS="-O3 -ffast-math"
    fi
    
    # Compile
    local exe_name="${FORTRAN_FILE%.*}_${version}_${build_type}"
    if ! $gfortran_cmd $FLAGS -o "$exe_name" "$FORTRAN_FILE" 2>/dev/null; then
        echo "❌ Compilation failed"
        return 1
    fi
    
    # Run with valgrind
    local log_file="valgrind_${version}_${build_type}.log"
    if ! valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                  --track-origins=yes --log-file="$log_file" \
                  "./$exe_name" >/dev/null 2>&1; then
        echo "❌ Runtime error"
        rm -f "$exe_name"
        return 1
    fi
    
    # Check results
    if grep -q "definitely lost: [1-9]" "$log_file"; then
        echo "❌ MEMORY LEAK DETECTED"
        grep "definitely lost" "$log_file"
        rm -f "$exe_name"
        return 1
    elif grep -q "possibly lost: [1-9]" "$log_file"; then
        echo "⚠️  Possible memory leak"
        grep "possibly lost" "$log_file"
    else
        echo "✅ No memory leaks"
    fi
    
    # Show memory usage
    local heap_usage=$(grep "total heap usage:" "$log_file" | head -1)
    if [ -n "$heap_usage" ]; then
        echo "   $heap_usage"
    fi
    
    rm -f "$exe_name" "$log_file"
    return 0
}

# Function to test with Docker (for newer versions)
test_docker() {
    local docker_image="$1"
    local build_type="$2"
    
    echo "Testing with Docker ($docker_image, $build_type build)..."
    
    if [ "$build_type" = "debug" ]; then
        FLAGS="-g -O0 -fbacktrace -fbounds-check -ffpe-trap=invalid,overflow,zero -Wall -Wextra"
    else
        FLAGS="-O3 -march=native -ffast-math"
    fi
    
    docker run --rm -v "$(pwd):/test" "$docker_image" /bin/bash -c "
        apt update -qq && apt install -y valgrind -qq >/dev/null 2>&1 &&
        cd /test &&
        gfortran $FLAGS -o test_exe '$FORTRAN_FILE' &&
        valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                 --track-origins=yes --log-file=docker_valgrind.log ./test_exe >/dev/null 2>&1 &&
        if grep -q 'definitely lost: [1-9]' docker_valgrind.log; then
            echo '❌ MEMORY LEAK DETECTED'
            grep 'definitely lost' docker_valgrind.log
            exit 1
        elif grep -q 'possibly lost: [1-9]' docker_valgrind.log; then
            echo '⚠️  Possible memory leak'
            grep 'possibly lost' docker_valgrind.log
        else
            echo '✅ No memory leaks'
        fi
        grep 'total heap usage:' docker_valgrind.log | head -1
        rm -f test_exe docker_valgrind.log
    " 2>/dev/null
}

# Main testing logic
echo "========================================="
echo "Memory Leak Testing: $FORTRAN_FILE"
echo "========================================="

# Check if valgrind is available
if ! command -v valgrind >/dev/null 2>&1; then
    echo "Installing valgrind..."
    sudo apt update -qq && sudo apt install -y valgrind -qq
fi

if [ -n "$SPECIFIC_VERSION" ]; then
    # Test specific version only
    for build_type in "${BUILD_TYPES[@]}"; do
        test_version "$SPECIFIC_VERSION" "$build_type"
        echo
    done
else
    # Test all available versions
    for version in "${GFORTRAN_VERSIONS[@]}"; do
        for build_type in "${BUILD_TYPES[@]}"; do
            test_version "$version" "$build_type"
        done
        echo
    done
    
    # Test with Docker for latest versions
    echo "Testing latest versions via Docker:"
    if command -v docker >/dev/null 2>&1; then
        for build_type in "${BUILD_TYPES[@]}"; do
            if test_docker "cmbant/docker-gcc-build:gcc15" "$build_type"; then
                echo "✅ GCC 15.1.1 ($build_type): No memory leaks"
            else
                echo "❌ GCC 15.1.1 ($build_type): Issues detected"
            fi
        done
    else
        echo "⚠️  Docker not available for latest version testing"
    fi
fi

echo "========================================="
echo "Testing completed for $FORTRAN_FILE"
echo "========================================="
