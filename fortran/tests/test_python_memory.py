#!/usr/bin/env python3
"""
Python memory testing for CAMB across different gfortran versions
Usage: python test_python_memory.py [gfortran_version]
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Configuration
GFORTRAN_VERSIONS = [9, 10, 11, 12, 13]
BUILD_TYPES = ["debug", "optimized"]

def run_command(cmd, cwd=None, capture_output=True):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_gfortran_version(version):
    """Check if gfortran version is available"""
    if version == "default":
        cmd = "gfortran --version"
    else:
        cmd = f"gfortran-{version} --version"
    
    success, stdout, _ = run_command(cmd)
    return success

def setup_camb_build(gfortran_version, build_type, temp_dir):
    """Setup CAMB build with specific gfortran version"""

    # Copy CAMB source to temp directory
    camb_root = Path(__file__).parent.parent
    temp_camb = Path(temp_dir) / "camb_test"

    # Copy essential files
    shutil.copytree(camb_root, temp_camb,
                   ignore=shutil.ignore_patterns('*.o', '*.mod', '*.so', '__pycache__',
                                                'Releaselib', '.git'))

    # Set compiler
    if gfortran_version == "default":
        compiler = "gfortran"
    else:
        compiler = f"gfortran-{gfortran_version}"

    # Set build flags
    if build_type == "debug":
        flags = "-g -O0 -fbacktrace -fbounds-check -ffpe-trap=invalid,overflow,zero -Wall"
    else:
        flags = "-O3 -ffast-math"

    # Set environment variables for the build
    env_vars = {
        'F90C': compiler,
        'FFLAGS': flags,
        'SFFLAGS': flags
    }

    return temp_camb, env_vars

def test_camb_python_simple():
    """Simple test of CAMB Python wrapper with current build"""

    print("Testing CAMB Python wrapper (current build)...")

    try:
        # Run memory test with valgrind on current CAMB installation
        print("   Running memory test with valgrind...")
        test_cmd = """
        valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                 --track-origins=yes --log-file=python_valgrind.log \
                 python3 -m unittest camb.tests.camb_test.CambTest.test_memory 2>/dev/null
        """

        success, stdout, stderr = run_command(test_cmd)

        # Check valgrind results
        valgrind_log = Path("python_valgrind.log")
        if valgrind_log.exists():
            with open(valgrind_log) as f:
                log_content = f.read()

            if "definitely lost:" in log_content and not "definitely lost: 0 bytes" in log_content:
                print("❌ MEMORY LEAK DETECTED")
                # Extract leak info
                for line in log_content.split('\n'):
                    if "definitely lost:" in line:
                        print(f"   {line.strip()}")
                return False
            elif "possibly lost:" in log_content and not "possibly lost: 0 bytes" in log_content:
                print("⚠️  Possible memory leak (often normal for Python)")
                for line in log_content.split('\n'):
                    if "possibly lost:" in line:
                        print(f"   {line.strip()}")
                print("   Note: 'Possibly lost' is often due to Python interpreter memory management")
            else:
                print("✅ No memory leaks")

            # Show heap usage
            for line in log_content.split('\n'):
                if "total heap usage:" in line:
                    print(f"   {line.strip()}")
                    break

            # Cleanup
            valgrind_log.unlink()
        else:
            print("⚠️  No valgrind log found")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_docker_python():
    """Test with Docker for latest versions"""
    print("Testing latest versions via Docker:")
    
    # Check if docker is available
    success, _, _ = run_command("docker --version")
    if not success:
        print("⚠️  Docker not available")
        return
    
    for build_type in BUILD_TYPES:
        print(f"Testing GCC 15.1.1 ({build_type} build)...")
        
        if build_type == "debug":
            flags = "-g -O0 -fbacktrace -fbounds-check -ffpe-trap=invalid,overflow,zero -Wall"
        else:
            flags = "-O3 -ffast-math"
        
        docker_cmd = f"""
        docker run --rm -v $(pwd)/../../:/camb cmbant/docker-gcc-build:gcc15 /bin/bash -c "
            apt update -qq && apt install -y valgrind python3-pip -qq >/dev/null 2>&1 &&
            cd /camb &&
            export F90C=gfortran &&
            export FFLAGS='{flags}' &&
            make clean && make camb &&
            python3 setup.py build_ext --inplace &&
            valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                     --track-origins=yes --log-file=docker_python_valgrind.log \
                     python3 -m unittest camb.tests.camb_test.CambTest.test_memory 2>/dev/null &&
            if grep -q 'definitely lost: [1-9]' docker_python_valgrind.log; then
                echo '❌ MEMORY LEAK DETECTED'
                grep 'definitely lost' docker_python_valgrind.log
            elif grep -q 'possibly lost: [1-9]' docker_python_valgrind.log; then
                echo '⚠️  Possible memory leak'
                grep 'possibly lost' docker_python_valgrind.log
            else
                echo '✅ No memory leaks'
            fi
            grep 'total heap usage:' docker_python_valgrind.log | head -1
        "
        """
        
        success, stdout, stderr = run_command(docker_cmd, cwd=Path(__file__).parent)
        if success:
            print(stdout)
        else:
            print(f"❌ Docker test failed: {stderr}")

def main():
    """Main function"""
    
    print("=========================================")
    print("CAMB Python Memory Testing")
    print("=========================================")
    
    # Find CAMB root directory
    current_dir = Path(__file__).parent
    camb_root = current_dir.parent  # Go up from fortran/tests to root

    # Check if we can find setup.py
    if not (camb_root / "setup.py").exists():
        # Try current directory
        if Path("setup.py").exists():
            camb_root = Path(".")
        else:
            print("Error: Cannot find CAMB setup.py")
            print(f"Looked in: {camb_root} and current directory")
            sys.exit(1)

    # Change to CAMB root directory
    os.chdir(camb_root)
    print(f"Working in: {os.getcwd()}")
    
    # Check if valgrind is available
    success, _, _ = run_command("valgrind --version")
    if not success:
        print("Installing valgrind...")
        run_command("sudo apt update -qq && sudo apt install -y valgrind -qq")
    
    # Parse arguments
    specific_version = None
    if len(sys.argv) > 1:
        try:
            specific_version = int(sys.argv[1])
        except ValueError:
            print(f"Invalid version: {sys.argv[1]}")
            sys.exit(1)
    
    # Run simple test with current CAMB build
    test_camb_python_simple()
    
    print("=========================================")
    print("Python testing completed")
    print("=========================================")

if __name__ == "__main__":
    main()
