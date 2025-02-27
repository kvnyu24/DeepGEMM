import copy
import ctypes
import os
import logging
import torch

from typing import Any, Iterable, Dict, Tuple, Union, List, Optional

# Configure logging
logger = logging.getLogger("deep_gemm.jit.template")

class TemplateError(Exception):
    """Exception raised for errors in the template module."""
    pass


# Name map for Python `eval`
typename_map: Dict[Any, str] = {
    **{t: t.__name__ for t in (bool, int, float)},
    torch.int: 'torch.int',
    torch.float: 'torch.float',
    torch.bfloat16: 'torch.bfloat16',
    torch.float8_e4m3fn: 'torch.float8_e4m3fn',
    torch.cuda.Stream: 'torch.cuda.Stream',
}

# `ctype` map for Python casting
ctype_map: Dict[Any, Any] = {
    **{t: getattr(ctypes, f'c_{t.__name__}') for t in (bool, int, float)},
    **{t: ctypes.c_void_p for t in (torch.int, torch.float, torch.bfloat16, torch.float8_e4m3fn, torch.cuda.Stream)},
}


# Type map for both Python API and source code usages
genc_map = {
    bool: ('bool', 'bool'),
    int: ('int', 'int'),
    float: ('float', 'float'),
    torch.int: ('void*', 'int*'),
    torch.float: ('void*', 'float*'),
    torch.bfloat16: ('void*', '__nv_bfloat16*'),
    torch.float8_e4m3fn: ('void*', '__nv_fp8_e4m3*'),
    torch.cuda.Stream: ('void*', 'cudaStream_t'),
}


def map_ctype(value: Any) -> Any:
    """
    Map a Python value to its corresponding C type for FFI.
    
    Args:
        value: The Python value to map
        
    Returns:
        The mapped C type value
        
    Raises:
        TemplateError: If the value cannot be mapped
    """
    try:
        if isinstance(value, torch.Tensor):
            if not value.is_cuda:
                raise TemplateError("Cannot map CPU tensor to C type, CUDA tensor required")
            return ctype_map[value.dtype](value.data_ptr())
        elif isinstance(value, torch.cuda.Stream):
            return ctype_map[type(value)](value.cuda_stream)
        elif type(value) in ctype_map:
            return ctype_map[type(value)](value)
        else:
            raise TemplateError(f"Unsupported type for C type mapping: {type(value)}")
    except Exception as e:
        if isinstance(e, TemplateError):
            raise
        raise TemplateError(f"Failed to map value to C type: {e}")


def cpp_format(template: str, keys: Dict[str, Any]) -> str:
    """
    Format a C++ template string with the given keys.
    
    This function is safer than str.format() for C++ code that may contain braces.
    
    Args:
        template: The template string with {key} placeholders
        keys: Dictionary of key-value pairs to substitute in the template
        
    Returns:
        The formatted string
        
    Raises:
        TemplateError: If a key is not found in the template
    """
    # We don't use `str.format` because it's not safe for C++ {} braces
    new_template = copy.deepcopy(template)
    keys_used = set()
    
    for key, value in keys.items():
        placeholder = f'{{{key}}}'
        if placeholder not in new_template:
            logger.warning(f"Key '{key}' not found in template")
        else:
            keys_used.add(key)
            new_template = new_template.replace(placeholder, f'{value}')
    
    # Check if all keys were used
    unused_keys = set(keys.keys()) - keys_used
    if unused_keys:
        logger.warning(f"Unused keys in template: {unused_keys}")
    
    return new_template


def validate_includes(includes: Iterable[str]) -> List[str]:
    """
    Validate include statements.
    
    Args:
        includes: List of include statements
        
    Returns:
        Validated list of include statements
        
    Raises:
        TemplateError: If an include statement is invalid
    """
    if includes is None:
        return []
        
    if not isinstance(includes, (list, tuple)):
        raise TemplateError("Includes must be a list or tuple")
        
    valid_includes = []
    for include in includes:
        if not isinstance(include, str):
            raise TemplateError(f"Include must be a string, got {type(include)}")
            
        # Check if include is properly formatted
        if not (include.startswith('<') and include.endswith('>')) and \
           not (include.startswith('"') and include.endswith('"')):
            logger.warning(f"Include '{include}' is not properly formatted, should be <...> or \"...\"")
            
        valid_includes.append(include)
        
    return valid_includes


def validate_arg_defs(arg_defs: Iterable[Tuple]) -> List[Tuple[str, Any]]:
    """
    Validate argument definitions.
    
    Args:
        arg_defs: List of argument definitions as (name, type) tuples
        
    Returns:
        Validated list of argument definitions
        
    Raises:
        TemplateError: If an argument definition is invalid
    """
    if arg_defs is None:
        raise TemplateError("Argument definitions cannot be None")
        
    if not isinstance(arg_defs, (list, tuple)):
        raise TemplateError("Argument definitions must be a list or tuple")
        
    valid_arg_defs = []
    for arg_def in arg_defs:
        if not isinstance(arg_def, tuple) or len(arg_def) != 2:
            raise TemplateError(f"Invalid argument definition: {arg_def}")
            
        arg_name, arg_type = arg_def
        
        if not isinstance(arg_name, str) or not arg_name:
            raise TemplateError(f"Invalid argument name: {arg_name}")
            
        if arg_type not in genc_map:
            raise TemplateError(f"Unsupported argument type: {arg_type}")
            
        valid_arg_defs.append((arg_name, arg_type))
        
    return valid_arg_defs


def generate(includes: Iterable[str], arg_defs: Iterable[Tuple], body: str) -> str:
    """
    Generate CUDA C++ code from templates.
    
    Args:
        includes: List of include statements
        arg_defs: List of argument definitions as (name, type) tuples
        body: The function body
        
    Returns:
        The generated code
        
    Raises:
        TemplateError: If generation fails
    """
    try:
        # Validate inputs
        includes = validate_includes(includes)
        arg_defs = validate_arg_defs(arg_defs)
        
        if not isinstance(body, str):
            raise TemplateError("Body must be a string")
            
        # Common prefix
        code = '// DeepGEMM auto-generated JIT CUDA source file\n\n'

        # Includes
        preload_sys_includes = ['<cuda.h>', '<cuda_fp8.h>', '<cuda_runtime.h>', '<iostream>']
        preload_package_includes = ['"cutlass/cutlass.h"']

        sys_includes = sorted(list(set(preload_sys_includes + [include for include in includes if include.startswith('<')])))
        package_includes = sorted(list(set(preload_package_includes + [include for include in includes if include.startswith('"')])))
        code += '\n'.join(f'#include {include}' for include in sys_includes) + '\n\n'
        code += '\n'.join(f'#include {include}' for include in package_includes) + '\n\n'

        # Function signature
        raw = '__raw_'
        get_def = lambda n, t: f'{genc_map[t][0]} ' + (raw if genc_map[t][0] != genc_map[t][1] else '') + n
        code += f'extern "C" void launch('
        code += ', '.join([get_def(*arg_def) for arg_def in arg_defs] + ['int& __return_code', ])
        code += ') {\n'

        # Cast raw types
        code += '    // Cast raw types (if needed)\n'
        for arg_name, arg_type in arg_defs:
            if genc_map[arg_type][0] != genc_map[arg_type][1]:
                code += f'    auto {arg_name} = reinterpret_cast<{genc_map[arg_type][1]}>({raw}{arg_name});\n'

        # Function body - add try/catch to handle CUDA errors
        code += '    try {\n'
        body_lines = body.split('\n')
        # Add proper indentation for each line
        code += '\n'.join(['        ' + (line if line else '') for line in body_lines])
        
        # Add error handling
        code += '\n    } catch (const std::exception& e) {\n'
        code += '        std::cerr << "CUDA kernel error: " << e.what() << std::endl;\n'
        code += '        __return_code = 1;\n'
        code += '    } catch (...) {\n'
        code += '        std::cerr << "Unknown CUDA kernel error" << std::endl;\n'
        code += '        __return_code = 2;\n'
        code += '    }\n'

        # End the function
        code += '}\n\n'

        # Debug print
        if os.getenv('DG_JIT_DEBUG', None):
            logger.debug(f'Generated code:\n{code}')

        return code
        
    except Exception as e:
        if isinstance(e, TemplateError):
            raise
        raise TemplateError(f"Failed to generate code: {e}")


def generate_with_includes_from_file(
    include_files: List[str], 
    arg_defs: Iterable[Tuple], 
    body: str
) -> str:
    """
    Generate CUDA C++ code with includes from files.
    
    This helper function reads include directives from source files and adds them
    to the generated code.
    
    Args:
        include_files: List of files to scan for include statements
        arg_defs: List of argument definitions as (name, type) tuples
        body: The function body
        
    Returns:
        The generated code
        
    Raises:
        TemplateError: If generation fails
    """
    import re
    
    # Regex to match include directives
    include_pattern = re.compile(r'^\s*#\s*include\s+([<"][^>"]+[>"])')
    includes = []
    
    # Scan files for include statements
    for file_path in include_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    match = include_pattern.match(line)
                    if match:
                        includes.append(match.group(1))
        except Exception as e:
            logger.warning(f"Failed to read includes from {file_path}: {e}")
            
    # Generate code with collected includes
    return generate(includes, arg_defs, body)
