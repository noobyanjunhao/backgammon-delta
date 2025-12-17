#!/usr/bin/env python3
# Quick script to check if CUDA libraries are accessible
import os
import ctypes.util

print("=== CUDA Library Check ===")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')[:300]}")

# Check for key CUDA libraries
libs = ['libcusparse.so', 'libcublas.so', 'libcudnn.so', 'libcurand.so']
lib_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
lib_paths.extend(['/usr/lib64', '/lib64'])

if os.environ.get('CUDA_HOME'):
    lib_paths.insert(0, os.path.join(os.environ['CUDA_HOME'], 'lib64'))

print("\nSearching for CUDA libraries:")
for lib_name in libs:
    found = False
    for path in lib_paths:
        if path and os.path.exists(path):
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                print(f"  ✓ {lib_name} found at: {lib_path}")
                found = True
                break
    if not found:
        print(f"  ✗ {lib_name} NOT FOUND")

