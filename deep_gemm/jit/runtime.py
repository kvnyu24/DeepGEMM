import ctypes
import os
import logging
import torch
from typing import Optional, Dict, Tuple, List, Any

from .template import map_ctype

# Configure logging
logger = logging.getLogger("deep_gemm.jit.runtime")

class RuntimeError(Exception):
    """Exception raised for errors in the Runtime class."""
    pass

class Runtime:
    """
    A class that manages the loading and execution of a compiled CUDA kernel.
    
    This class handles loading the shared object file, parsing the argument
    definitions, and invoking the kernel with the correct argument types.
    """
    
    def __init__(self, path: str) -> None:
        """
        Initialize a Runtime object for a compiled kernel.
        
        Args:
            path: Path to the directory containing the compiled kernel files
            
        Raises:
            RuntimeError: If the path is invalid or the kernel cannot be loaded
        """
        self.path = path
        self.lib = None
        self.args = None
        self.so_path = os.path.join(self.path, 'kernel.so')
        self.args_path = os.path.join(self.path, 'kernel.args')

        if not self.is_path_valid(self.path):
            raise RuntimeError(f"Invalid runtime path: {self.path}")
            
        logger.debug(f"Runtime initialized for path: {self.path}")

    @staticmethod
    def is_path_valid(path: str) -> bool:
        """
        Check if a path contains valid compiled kernel files.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path contains all necessary files, False otherwise
        """
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            logger.debug(f"Path does not exist or is not a directory: {path}")
            return False

        # Contains all necessary files
        files = ['kernel.cu', 'kernel.args', 'kernel.so']
        for file in files:
            file_path = os.path.join(path, file)
            if not os.path.exists(file_path):
                logger.debug(f"Required file missing: {file_path}")
                return False
            # Check if kernel.so is readable and executable
            if file == 'kernel.so':
                if not os.access(file_path, os.R_OK | os.X_OK):
                    logger.debug(f"Kernel.so is not readable or executable: {file_path}")
                    return False
        
        return True

    def load(self) -> None:
        """
        Load the shared object file and argument definitions.
        
        Raises:
            RuntimeError: If loading fails
        """
        if self.lib is None:
            try:
                self.lib = ctypes.CDLL(self.so_path)
                logger.debug(f"Loaded shared object: {self.so_path}")
            except Exception as e:
                logger.error(f"Failed to load shared object: {self.so_path}: {e}")
                raise RuntimeError(f"Failed to load shared object: {e}")
                
        if self.args is None:
            try:
                with open(self.args_path, 'r') as f:
                    self.args = eval(f.read())
                logger.debug(f"Loaded arguments: {self.args}")
            except Exception as e:
                logger.error(f"Failed to load argument definitions from {self.args_path}: {e}")
                raise RuntimeError(f"Failed to load argument definitions: {e}")

    def validate_args(self, args: Tuple) -> None:
        """
        Validate the arguments passed to the kernel.
        
        Args:
            args: Arguments to validate
            
        Raises:
            RuntimeError: If arguments are invalid
        """
        if len(args) != len(self.args):
            raise RuntimeError(f'Expected {len(self.args)} arguments, got {len(args)}')
            
        for i, (arg, (name, dtype)) in enumerate(zip(args, self.args)):
            if isinstance(arg, torch.Tensor):
                if arg.dtype != dtype:
                    raise RuntimeError(f'Expected tensor dtype `{dtype}` for argument {i} (`{name}`), got `{arg.dtype}`')
                if not arg.is_cuda:
                    raise RuntimeError(f'Expected CUDA tensor for argument {i} (`{name}`), got CPU tensor')
            else:
                if not isinstance(arg, dtype):
                    raise RuntimeError(f'Expected built-in type `{dtype}` for argument {i} (`{name}`), got `{type(arg)}`')

    def __call__(self, *args) -> int:
        """
        Call the compiled kernel with the given arguments.
        
        Args:
            *args: Arguments to pass to the kernel
            
        Returns:
            Return code from the kernel
            
        Raises:
            RuntimeError: If calling the kernel fails
        """
        # Load SO file and args if needed
        if self.lib is None or self.args is None:
            self.load()

        # Validate arguments
        self.validate_args(args)

        # Convert arguments to C types
        cargs = []
        for arg, (name, dtype) in zip(args, self.args):
            try:
                cargs.append(map_ctype(arg))
            except Exception as e:
                logger.error(f"Failed to convert argument `{name}` to C type: {e}")
                raise RuntimeError(f"Failed to convert argument `{name}` to C type: {e}")

        # Call the kernel
        return_code = ctypes.c_int(0)
        try:
            self.lib.launch(*cargs, ctypes.byref(return_code))
            return return_code.value
        except Exception as e:
            logger.error(f"Failed to launch kernel: {e}")
            raise RuntimeError(f"Failed to launch kernel: {e}")


class RuntimeCache:
    """
    A cache for compiled kernel runtimes.
    
    This class manages a dictionary of Runtime objects, indexed by path.
    It provides methods to get and set cache entries, and to check if a
    runtime exists on disk.
    """
    
    def __init__(self) -> None:
        """Initialize an empty runtime cache."""
        self.cache: Dict[str, Runtime] = {}

    def __getitem__(self, path: str) -> Optional[Runtime]:
        """
        Get a runtime from the cache, or load it from disk if not in cache.
        
        Args:
            path: Path to the compiled kernel directory
            
        Returns:
            Runtime object if found, None otherwise
        """
        # In Python runtime
        if path in self.cache:
            logger.debug(f"Cache hit for path: {path}")
            return self.cache[path]

        # Try to load from disk
        if os.path.exists(path):
            try:
                if Runtime.is_path_valid(path):
                    logger.debug(f"Loading runtime from disk: {path}")
                    runtime = Runtime(path)
                    self.cache[path] = runtime
                    return runtime
            except Exception as e:
                logger.warning(f"Failed to load runtime from disk: {path}: {e}")
                
        logger.debug(f"No cached runtime found for path: {path}")
        return None

    def __setitem__(self, path: str, runtime: Runtime) -> None:
        """
        Add a runtime to the cache.
        
        Args:
            path: Path to use as the cache key
            runtime: Runtime object to cache
        """
        logger.debug(f"Adding runtime to cache: {path}")
        self.cache[path] = runtime
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "cached_paths": list(self.cache.keys())
        }
