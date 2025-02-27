import hashlib
import functools
import os
import re
import subprocess
import uuid
import logging
from torch.utils.cpp_extension import CUDA_HOME
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

from . import interleave_ffma
from .runtime import Runtime, RuntimeCache
from .template import typename_map

# Configure logging
logging.basicConfig(level=logging.INFO if os.getenv('DG_JIT_DEBUG') else logging.WARNING)
logger = logging.getLogger("deep_gemm.jit")

# Global runtime cache
runtime_cache = RuntimeCache()

# Custom exceptions for better error handling
class JITCompilationError(Exception):
    """Exception raised for errors during JIT compilation."""
    pass

class ValidationError(Exception):
    """Exception raised for parameter validation errors."""
    pass

def hash_to_hex(s: str) -> str:
    """Generate a short hex hash from a string for uniquely identifying compiled kernels."""
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()[0:12]

@functools.lru_cache(maxsize=None)
def get_jit_include_dir() -> str:
    """Get the include directory for the JIT compiler."""
    include_dir = f'{os.path.dirname(os.path.abspath(__file__))}/../include'
    if not os.path.exists(include_dir):
        raise ValidationError(f"Include directory {include_dir} does not exist")
    return include_dir

@functools.lru_cache(maxsize=None)
def get_deep_gemm_version() -> str:
    """Calculate a version hash based on the included .cuh files and interleave_ffma.py."""
    # Update include directories
    include_dir = f'{get_jit_include_dir()}/deep_gemm'
    if not os.path.exists(include_dir):
        raise ValidationError(f'Cannot find GEMM include directory {include_dir}')
    
    md5 = hashlib.md5()
    
    # Hash all .cuh files in the include directory
    for filename in filter(lambda x: x.endswith('.cuh'), sorted(os.listdir(include_dir))):
        try:
            with open(f'{include_dir}/{filename}', 'rb') as f:
                md5.update(f.read())
        except IOError as e:
            logger.warning(f"Could not read file {filename}: {e}")

    # Update `interleave_ffma.py`
    try:
        with open(f'{os.path.dirname(os.path.realpath(__file__))}/interleave_ffma.py', 'rb') as f:
            md5.update(f.read())
    except IOError as e:
        logger.warning(f"Could not read interleave_ffma.py: {e}")
        
    return md5.hexdigest()[0:12]

@functools.lru_cache(maxsize=None)
def get_nvcc_compiler() -> Tuple[str, str]:
    """
    Find a suitable NVCC compiler and its version.
    
    Returns:
        Tuple[str, str]: The path to the NVCC compiler and its version
    
    Raises:
        ValidationError: If no suitable NVCC compiler is found or version is too low
    """
    paths = []
    if os.getenv('DG_NVCC_COMPILER'):
        paths.append(os.getenv('DG_NVCC_COMPILER'))
    
    # Add CUDA_HOME path if available
    if CUDA_HOME:
        paths.append(f'{CUDA_HOME}/bin/nvcc')
    
    # Add system path if the above fails
    paths.append('nvcc')  # Try to find nvcc in PATH

    # Try to find the first available NVCC compiler
    least_version_required = '12.3'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    
    for path in paths:
        try:
            result = subprocess.run([path, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=False)
            
            if result.returncode != 0:
                logger.debug(f"NVCC at {path} failed with: {result.stderr}")
                continue
                
            match = version_pattern.search(result.stdout)
            if not match:
                logger.debug(f"Could not parse NVCC version from output: {result.stdout}")
                continue
                
            version = match.group(1)
            if version < least_version_required:
                logger.debug(f"NVCC {path} version {version} is lower than required {least_version_required}")
                continue
                
            logger.info(f"Using NVCC compiler: {path} (version {version})")
            return path, version
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Error checking NVCC at {path}: {e}")
            continue
            
    raise ValidationError(f'Cannot find any available NVCC compiler with version >= {least_version_required}')

@functools.lru_cache(maxsize=None)
def get_default_user_dir() -> str:
    """Get the default user directory for cache files."""
    if 'DG_CACHE_DIR' in os.environ:
        path = os.getenv('DG_CACHE_DIR')
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError as e:
            logger.warning(f"Could not create cache directory {path}: {e}")
            # Fall back to default
            
    home_dir = os.path.expanduser('~')
    default_path = f'{home_dir}/.deep_gemm'
    try:
        os.makedirs(default_path, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create default cache directory {default_path}: {e}")
        # Try a temporary directory as last resort
        import tempfile
        default_path = tempfile.mkdtemp(prefix='deep_gemm_')
        logger.warning(f"Using temporary directory for cache: {default_path}")
        
    return default_path

@functools.lru_cache(maxsize=None)
def get_tmp_dir() -> str:
    """Get the temporary directory for intermediate files."""
    return f'{get_default_user_dir()}/tmp'

@functools.lru_cache(maxsize=None)
def get_cache_dir() -> str:
    """Get the cache directory for compiled kernels."""
    return f'{get_default_user_dir()}/cache'

def make_tmp_dir() -> str:
    """Create and return the temporary directory."""
    tmp_dir = get_tmp_dir()
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
    except OSError as e:
        logger.error(f"Could not create temporary directory {tmp_dir}: {e}")
        raise JITCompilationError(f"Failed to create temporary directory: {e}")

def put(path: str, data: Union[str, bytes], is_binary: bool = False) -> None:
    """
    Write data to a file with atomic replace.
    
    Args:
        path: Target file path
        data: Content to write
        is_binary: Whether to write in binary mode
    
    Raises:
        JITCompilationError: If writing the file fails
    """
    try:
        # Write and do POSIX atomic replace
        tmp_file_path = f'{make_tmp_dir()}/file.tmp.{str(uuid.uuid4())}.{hash_to_hex(path)}'
        with open(tmp_file_path, 'wb' if is_binary else 'w') as f:
            f.write(data)
        os.replace(tmp_file_path, path)
    except (IOError, OSError) as e:
        logger.error(f"Failed to write file {path}: {e}")
        raise JITCompilationError(f"Failed to write file {path}: {e}")

def validate_build_params(name: str, arg_defs: tuple, code: str) -> None:
    """
    Validate parameters for the build function.
    
    Args:
        name: Name of the kernel
        arg_defs: Argument definitions
        code: CUDA code to compile
        
    Raises:
        ValidationError: If any parameter validation fails
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Kernel name must be a non-empty string")
        
    if not arg_defs or not isinstance(arg_defs, tuple):
        raise ValidationError("Argument definitions must be a non-empty tuple")
        
    for arg_def in arg_defs:
        if not isinstance(arg_def, tuple) or len(arg_def) != 2:
            raise ValidationError(f"Invalid argument definition: {arg_def}")
        arg_name, arg_type = arg_def
        if not isinstance(arg_name, str) or not arg_name:
            raise ValidationError(f"Invalid argument name: {arg_name}")
        if arg_type not in typename_map:
            raise ValidationError(f"Unsupported argument type: {arg_type}")
            
    if not code or not isinstance(code, str):
        raise ValidationError("Code must be a non-empty string")

def get_compiler_flags() -> Dict[str, List[str]]:
    """Get the compiler flags for NVCC compilation."""
    nvcc_flags = [
        '-std=c++17', 
        '-shared', 
        '-O3', 
        '--expt-relaxed-constexpr', 
        '--expt-extended-lambda',
        '-gencode=arch=compute_90a,code=sm_90a',
        '--ptxas-options=--register-usage-level=10' + 
        (',--verbose' if 'DG_PTXAS_VERBOSE' in os.environ else ''),
        # Suppress some unnecessary warnings
        '--diag-suppress=177,174,940'
    ]
    
    cxx_flags = [
        '-fPIC', 
        '-O3', 
        '-Wno-deprecated-declarations', 
        '-Wno-abi'
    ]
    
    return {
        'nvcc_flags': nvcc_flags,
        'cxx_flags': cxx_flags,
        'combined': [*nvcc_flags, f'--compiler-options={",".join(cxx_flags)}']
    }

def find_in_cache(name: str, signature: str, path: str, 
                  fallback_callback: Optional[Callable] = None) -> Optional[Runtime]:
    """
    Try to find a cached runtime, either in memory or on disk.
    
    Args:
        name: Name of the kernel
        signature: Kernel signature for cache lookup
        path: Cache directory path
        fallback_callback: Function to call if the kernel is not found in cache
        
    Returns:
        Runtime object if found, None otherwise
    """
    global runtime_cache
    
    # Check in-memory cache first
    if runtime_cache[path] is not None:
        logger.debug(f'Using cached JIT runtime {name} from memory')
        return runtime_cache[path]
    
    # Check if already compiled on disk
    if os.path.exists(path) and Runtime.is_path_valid(path):
        try:
            runtime = Runtime(path)
            runtime_cache[path] = runtime
            logger.debug(f'Using cached JIT runtime {name} from disk')
            return runtime
        except Exception as e:
            logger.warning(f"Error loading cached runtime from {path}: {e}")
            # Continue to recompilation
    
    # Not found in cache, call fallback if provided
    if fallback_callback:
        try:
            return fallback_callback()
        except Exception as e:
            logger.error(f"Fallback compilation failed: {e}")
            raise JITCompilationError(f"Compilation failed and no valid cache was found: {e}")
    
    return None

def compile_kernel(src_path: str, tmp_so_path: str, flags: List[str], 
                  include_dirs: List[str]) -> None:
    """
    Compile a CUDA kernel using NVCC.
    
    Args:
        src_path: Path to source file
        tmp_so_path: Path for output shared object
        flags: Compiler flags
        include_dirs: Include directories
        
    Raises:
        JITCompilationError: If compilation fails
    """
    command = [
        get_nvcc_compiler()[0],
        src_path, 
        '-o', tmp_so_path,
        *flags,
        *[f'-I{d}' for d in include_dirs]
    ]
    
    if os.getenv('DG_JIT_DEBUG', None) or os.getenv('DG_JIT_PRINT_NVCC_COMMAND', False):
        logger.info(f'Compiling with command: {" ".join(command)}')
    
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        if os.getenv('DG_JIT_DEBUG', None):
            if process.stdout:
                logger.debug(f"Compiler stdout: {process.stdout}")
                
    except subprocess.CalledProcessError as e:
        error_msg = f"NVCC compilation failed with code {e.returncode}:\n{e.stderr}"
        logger.error(error_msg)
        raise JITCompilationError(error_msg)

def build(name: str, arg_defs: tuple, code: str) -> Runtime:
    """
    Build a CUDA kernel using JIT compilation.
    
    Args:
        name: Name of the kernel
        arg_defs: Argument definitions
        code: CUDA code to compile
        
    Returns:
        Runtime object for the compiled kernel
        
    Raises:
        ValidationError: If parameter validation fails
        JITCompilationError: If compilation fails
    """
    # Validate parameters
    validate_build_params(name, arg_defs, code)
    
    # Get compiler flags
    flags = get_compiler_flags()['combined']
    include_dirs = [get_jit_include_dir()]

    # Build signature and prepare paths
    nvcc_compiler_path, nvcc_version = get_nvcc_compiler()
    enable_sass_opt = nvcc_version <= '12.8' and int(os.getenv('DG_DISABLE_FFMA_INTERLEAVE', 0)) == 0
    
    signature = f'{name}$${get_deep_gemm_version()}$${code}$${nvcc_compiler_path}$${flags}$${enable_sass_opt}'
    hashed_name = f'kernel.{name}.{hash_to_hex(signature)}'
    path = f'{get_cache_dir()}/{hashed_name}'
    
    # Check cache first
    def compile_fallback():
        # Create cache directory
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create cache directory {path}: {e}")
            raise JITCompilationError(f"Failed to create cache directory: {e}")
            
        # Write the code and arguments
        args_path = f'{path}/kernel.args'
        src_path = f'{path}/kernel.cu'
        put(args_path, ', '.join([f"('{arg_def[0]}', {typename_map[arg_def[1]]})" for arg_def in arg_defs]))
        put(src_path, code)

        # Compile into a temporary SO file
        so_path = f'{path}/kernel.so'
        tmp_so_path = f'{make_tmp_dir()}/nvcc.tmp.{str(uuid.uuid4())}.{hash_to_hex(so_path)}.so'

        # Compile
        logger.info(f'Compiling JIT runtime {hashed_name}')
        compile_kernel(src_path, tmp_so_path, flags, include_dirs)

        # Apply FFMA interleaving optimization if enabled
        if enable_sass_opt:
            try:
                interleave_ffma.process(tmp_so_path)
                logger.debug("Applied FFMA interleaving optimization")
            except Exception as e:
                logger.warning(f"FFMA interleaving failed, continuing with unoptimized binary: {e}")

        # Atomic replace SO file
        try:
            os.replace(tmp_so_path, so_path)
        except OSError as e:
            logger.error(f"Failed to replace SO file {so_path}: {e}")
            raise JITCompilationError(f"Failed to finalize compilation: {e}")

        # Create and cache runtime
        global runtime_cache
        runtime = Runtime(path)
        runtime_cache[path] = runtime
        return runtime
    
    cached_runtime = find_in_cache(hashed_name, signature, path, compile_fallback)
    if cached_runtime is None:
        raise JITCompilationError(f"Failed to build runtime {name}")
        
    return cached_runtime

def clear_cache() -> int:
    """
    Clear the JIT kernel cache.
    
    Returns:
        Number of cache entries removed
    """
    global runtime_cache
    cache_count = len(runtime_cache.cache)
    runtime_cache.cache.clear()
    
    cache_dir = get_cache_dir()
    if os.path.exists(cache_dir):
        try:
            import shutil
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path) and item.startswith('kernel.'):
                    shutil.rmtree(item_path)
        except OSError as e:
            logger.warning(f"Failed to clear cache directory {cache_dir}: {e}")
            
    return cache_count
