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


def run_tests(test_modules=None, verbosity=1):
    """
    Run the specified test modules.
    
    Args:
        test_modules: List of test modules to run, or None to run all tests
        verbosity: Verbosity level for test output
    """
    if test_modules is None:
        test_modules = find_test_modules()
        
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            # Import the module
            module = __import__(module_name)
            # Add all tests from the module
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Error importing {module_name}: {e}")
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepGEMM tests")
    parser.add_argument(
        "--tests", 
        nargs="*", 
        help="Specific test modules to run (e.g., test_jit test_template)",
        default=None
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only the quick tests (skip the slower core tests)"
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.tests:
        # Convert test names to module names if needed
        test_modules = [t if t.startswith('test_') else f'test_{t}' for t in args.tests]
    else:
        test_modules = find_test_modules()
        
        # Filter out slow tests if requested
        if args.quick:
            test_modules = [m for m in test_modules if m != 'test_core']
            
    verbosity = 2 if args.verbose else 1
    
    print(f"Running tests: {', '.join(test_modules)}")
    success = run_tests(test_modules, verbosity)
    
    sys.exit(0 if success else 1) 