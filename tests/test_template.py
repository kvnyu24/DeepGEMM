import os
import unittest
import torch
import tempfile
import shutil
from pathlib import Path

import deep_gemm
from deep_gemm.jit import template
from deep_gemm.jit.template import TemplateError

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()


class TemplateTests(unittest.TestCase):
    """Test the enhanced template module functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_cpp_format(self):
        """Test the cpp_format function."""
        # Simple case
        template_str = "Hello {name}!"
        keys = {"name": "World"}
        result = template.cpp_format(template_str, keys)
        self.assertEqual(result, "Hello World!")
        
        # C++ braces
        template_str = "for (int i = 0; i < {size}; i++) { doSomething(); }"
        keys = {"size": 10}
        result = template.cpp_format(template_str, keys)
        self.assertEqual(result, "for (int i = 0; i < 10; i++) { doSomething(); }")
        
        # Missing key (should warn but not error)
        template_str = "Hello {name}!"
        keys = {"different_key": "World"}
        result = template.cpp_format(template_str, keys)
        self.assertEqual(result, "Hello {name}!")
        
        # Unused key (should warn but not error)
        template_str = "Hello World!"
        keys = {"name": "Ignored"}
        result = template.cpp_format(template_str, keys)
        self.assertEqual(result, "Hello World!")
        
    @unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
    def test_map_ctype(self):
        """Test the map_ctype function."""
        # Test basic types
        self.assertIsNotNone(template.map_ctype(True))
        self.assertIsNotNone(template.map_ctype(42))
        self.assertIsNotNone(template.map_ctype(3.14))
        
        # Test tensor type
        tensor = torch.zeros(1, dtype=torch.float, device='cuda')
        self.assertIsNotNone(template.map_ctype(tensor))
        
        # Test stream type
        stream = torch.cuda.Stream()
        self.assertIsNotNone(template.map_ctype(stream))
        
        # Test unsupported type
        with self.assertRaises(TemplateError):
            template.map_ctype("string")
            
        # Test CPU tensor (should fail)
        cpu_tensor = torch.zeros(1, dtype=torch.float)
        with self.assertRaises(TemplateError):
            template.map_ctype(cpu_tensor)
    
    def test_map_ctype_basic(self):
        """Test the map_ctype function with basic types (non-CUDA)."""
        # Test basic types that don't require CUDA
        self.assertIsNotNone(template.map_ctype(True))
        self.assertIsNotNone(template.map_ctype(42))
        self.assertIsNotNone(template.map_ctype(3.14))
        
        # Test unsupported type
        with self.assertRaises(TemplateError):
            template.map_ctype("string")
            
    def test_validate_includes(self):
        """Test the validate_includes function."""
        # Valid includes
        includes = ['<cuda.h>', '"myheader.h"']
        validated = template.validate_includes(includes)
        self.assertEqual(validated, includes)
        
        # None includes
        validated = template.validate_includes(None)
        self.assertEqual(validated, [])
        
        # Non-list/tuple
        with self.assertRaises(TemplateError):
            template.validate_includes("not a list")
            
        # Non-string include
        with self.assertRaises(TemplateError):
            template.validate_includes([123])
            
        # Improperly formatted include (should warn but not error)
        includes = ['cuda.h']  # Missing < >
        validated = template.validate_includes(includes)
        self.assertEqual(validated, includes)
        
    def test_validate_arg_defs(self):
        """Test the validate_arg_defs function."""
        # Valid arg defs
        arg_defs = [('a', int), ('b', torch.float)]
        validated = template.validate_arg_defs(arg_defs)
        self.assertEqual(validated, arg_defs)
        
        # None arg defs
        with self.assertRaises(TemplateError):
            template.validate_arg_defs(None)
            
        # Non-list/tuple
        with self.assertRaises(TemplateError):
            template.validate_arg_defs("not a list")
            
        # Invalid arg def (not a tuple)
        with self.assertRaises(TemplateError):
            template.validate_arg_defs(["not a tuple"])
            
        # Invalid arg def (wrong length)
        with self.assertRaises(TemplateError):
            template.validate_arg_defs([('a', int, 'extra')])
            
        # Invalid arg name
        with self.assertRaises(TemplateError):
            template.validate_arg_defs([(123, int)])
            
        # Invalid arg type
        with self.assertRaises(TemplateError):
            template.validate_arg_defs([('a', list)])
            
    def test_generate(self):
        """Test the generate function."""
        # Basic test
        includes = ['<cuda.h>']
        arg_defs = [('a', int)]
        body = "int result = a * 2;\n__return_code = result;"
        code = template.generate(includes, arg_defs, body)
        
        # Check that key components are present
        self.assertIn('#include <cuda.h>', code)
        self.assertIn('extern "C" void launch(int a, int& __return_code)', code)
        self.assertIn('int result = a * 2;', code)
        self.assertIn('__return_code = result;', code)
        self.assertIn('try {', code)  # Should have error handling
        self.assertIn('} catch', code)
        
        # Invalid body
        with self.assertRaises(TemplateError):
            template.generate(includes, arg_defs, None)
            
    def test_generate_with_includes_from_file(self):
        """Test the generate_with_includes_from_file function."""
        # Create a test file with includes
        include_file = os.path.join(self.temp_dir, 'test_includes.h')
        with open(include_file, 'w') as f:
            f.write('#include <cuda.h>\n')
            f.write('#include "myheader.h"\n')
            f.write('// Not an include\n')
            
        # Generate code from the file
        arg_defs = [('a', int)]
        body = "__return_code = a;"
        code = template.generate_with_includes_from_file([include_file], arg_defs, body)
        
        # Check that includes were properly extracted
        self.assertIn('#include <cuda.h>', code)
        self.assertIn('#include "myheader.h"', code)
        
        # Test with non-existent file (should warn but not error)
        code = template.generate_with_includes_from_file(['nonexistent.h'], arg_defs, body)
        self.assertIn('extern "C" void launch', code)  # Should still generate code


if __name__ == '__main__':
    unittest.main() 