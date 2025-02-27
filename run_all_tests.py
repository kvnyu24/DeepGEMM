#!/usr/bin/env python3
"""
Comprehensive test runner for DeepGEMM

This script runs all the tests including the new robustness tests
for the JIT compilation system and template module.
"""

import os
import sys
import unittest
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def find_test_modules():
    """Find all test modules in the current directory."""
    test_modules = []
    for filename in os.listdir(os.path.dirname(__file__)):
        if filename.startswith('test_') and filename.endswith('.py'):
            test_modules.append(filename[:-3])  # Remove .py extension
    return test_modules 