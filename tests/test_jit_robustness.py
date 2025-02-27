import os
import unittest
import torch
import tempfile
import shutil
from pathlib import Path

import deep_gemm
from deep_gemm.jit import compiler
from deep_gemm.jit.compiler import JITCompilationError, ValidationError


# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Check for NVCC availability
def is_nvcc_available():
    try:
        compiler.get_nvcc_compiler()
        return True
    except ValidationError:
        return False

NVCC_AVAILABLE = is_nvcc_available()


class JITRobustnessTests(unittest.TestCase):
    """Test the robustness of the JIT compilation system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.original_cache_dir = os.environ.get('DG_CACHE_DIR', None)
        os.environ['DG_CACHE_DIR'] = self.temp_dir
        
        # Reset the LRU cache to ensure we get fresh directories
        compiler.get_cache_dir.cache_clear()
        compiler.get_tmp_dir.cache_clear()
        compiler.get_default_user_dir.cache_clear()
        
    def tearDown(self):
        """Clean up after tests."""
        # Restore the original cache directory
        if self.original_cache_dir:
            os.environ['DG_CACHE_DIR'] = self.original_cache_dir
        else:
            os.environ.pop('DG_CACHE_DIR', None)
            
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Clear cache
        compiler.clear_cache()
        
    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        # Empty name
        with self.assertRaises(ValidationError):
            compiler.validate_build_params("", (('a', int),), "int main() {}")
            
        # None name
        with self.assertRaises(ValidationError):
            compiler.validate_build_params(None, (('a', int),), "int main() {}")
            
        # Empty arg_defs
        with self.assertRaises(ValidationError):
            compiler.validate_build_params("test", (), "int main() {}")
            
        # None arg_defs
        with self.assertRaises(ValidationError):
            compiler.validate_build_params("test", None, "int main() {}")
            
        # Invalid arg_def type
        with self.assertRaises(ValidationError):
            compiler.validate_build_params("test", (('a', list),), "int main() {}")
            
        # Empty code
        with self.assertRaises(ValidationError):
            compiler.validate_build_params("test", (('a', int),), "")
            
        # None code
        with self.assertRaises(ValidationError):
            compiler.validate_build_params("test", (('a', int),), None)
            
        # Valid parameters should not raise
        compiler.validate_build_params("test", (('a', int),), "int main() {}")
        
    @unittest.skipIf(not NVCC_AVAILABLE, "NVCC compiler not available")
    def test_error_handling_invalid_code(self):
        """Test that errors in compilation are handled correctly."""
        # Code with a syntax error
        invalid_code = """
        extern "C" void launch(int a, int& __return_code) {
            This is not valid C++ code;
        }
        """
        
        args = (('a', int),)
        
        with self.assertRaises(JITCompilationError):
            compiler.build("test_invalid", args, invalid_code)
            
    def test_cache_clear(self):
        """Test that the cache can be cleared."""
        # We'll test the cache clearing functionality without actually compiling
        # Create a fake cache directory with a kernel-like directory
        cache_dir = compiler.get_cache_dir()
        os.makedirs(os.path.join(cache_dir, 'kernel.test_clear_fake'), exist_ok=True)
        
        # Mock the runtime cache with a fake entry
        compiler.runtime_cache.cache = {'fake_path': 'fake_runtime'}
        
        # Clear the cache
        num_removed = compiler.clear_cache()
        self.assertGreaterEqual(num_removed, 1, "Should return a valid count of removed entries")
        
        # Check that the cache directory is now empty
        directories = [d for d in os.listdir(cache_dir) if d.startswith('kernel.test_clear')]
        self.assertEqual(len(directories), 0, "Cache directory should be empty after clearing")
        
    def test_nvcc_compiler_detection(self):
        """Test that the NVCC compiler is detected or handled gracefully."""
        # If NVCC is not available, this should raise ValidationError
        # If it is available, it should return a valid path and version
        try:
            compiler_path, version = compiler.get_nvcc_compiler()
            self.assertIsNotNone(compiler_path, "Compiler path should not be None")
            self.assertIsNotNone(version, "Compiler version should not be None")
        except ValidationError:
            # Skip this test if NVCC is not available
            self.skipTest("NVCC compiler not available")
            
    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    @unittest.skipIf(not NVCC_AVAILABLE, "NVCC compiler not available")
    def test_runtime_validation(self):
        """Test that runtime validates arguments correctly."""
        # Create a simple kernel
        args = (('tensor', torch.float),)
        code = """
        extern "C" void launch(float* tensor, int& __return_code) {
            __return_code = 0;
        }
        """
        
        runtime = compiler.build("test_validation", args, code)
        
        # Test with correct arguments
        tensor = torch.zeros(1, dtype=torch.float, device='cuda')
        result = runtime(tensor)
        self.assertEqual(result, 0, "Kernel execution failed")
        
        # Test with incorrect dtype
        tensor_wrong_dtype = torch.zeros(1, dtype=torch.int, device='cuda')
        with self.assertRaises(Exception):
            runtime(tensor_wrong_dtype)
            
        # Test with incorrect device
        tensor_cpu = torch.zeros(1, dtype=torch.float)
        with self.assertRaises(Exception):
            runtime(tensor_cpu)
            
        # Test with too few arguments
        with self.assertRaises(Exception):
            runtime()
            
        # Test with too many arguments
        with self.assertRaises(Exception):
            runtime(tensor, 42)
            
    @unittest.skipIf(not NVCC_AVAILABLE, "NVCC compiler not available")
    def test_caching_behavior(self):
        """Test that caching works correctly."""
        # Skip the actual test if NVCC is not available
        if not NVCC_AVAILABLE:
            self.skipTest("NVCC compiler not available")
            
        # Create a simple kernel
        args = (('a', int),)
        code = """
        extern "C" void launch(int a, int& __return_code) {
            __return_code = a;
        }
        """
        
        # First build should compile
        runtime1 = compiler.build("test_cache", args, code)
        
        # Second build with same parameters should use cache
        runtime2 = compiler.build("test_cache", args, code)
        
        # The two runtimes should have the same path
        self.assertEqual(runtime1.path, runtime2.path, 
                        "Second build should return cached runtime")
        
        # But they should be different objects
        self.assertIsNot(runtime1, runtime2, 
                        "Should return new Runtime object even when path is cached")
        
        # Both should work
        self.assertEqual(runtime1(42), 42, "First runtime failed")
        self.assertEqual(runtime2(42), 42, "Second runtime failed")


if __name__ == '__main__':
    unittest.main() 