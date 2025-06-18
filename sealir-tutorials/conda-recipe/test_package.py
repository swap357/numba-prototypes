#!/usr/bin/env python3
"""
Simple validation script for the sealir-tutorials conda package
"""
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import the tutorial modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_importable_modules():
    """Test that key modules can be found (syntax check)"""
    import ast
    
    try:
        with open("ch01_basic_compiler.py", "r") as f:
            ast.parse(f.read())
        print("✓ ch01_basic_compiler.py syntax is valid")
    except Exception as e:
        print(f"✗ ch01_basic_compiler.py syntax error: {e}")
        return False
        
    try:
        with open("utils.py", "r") as f:
            ast.parse(f.read())
        print("✓ utils.py syntax is valid")  
    except Exception as e:
        print(f"✗ utils.py syntax error: {e}")
        return False
        
    return True

def test_files_exist():
    """Test that expected files exist"""
    expected_files = [
        "pyproject.toml",
        "README.md", 
        "Makefile",
        "jupytext.toml"
    ]
    
    all_exist = True
    for file in expected_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
            
    return all_exist

if __name__ == "__main__":
    print("Testing sealir-tutorials package...")
    print("-" * 40)
    
    files_ok = test_files_exist()
    imports_ok = test_importable_modules()
    
    if files_ok and imports_ok:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)