import logging

from .compiler import build, clear_cache, get_nvcc_compiler, JITCompilationError, ValidationError
from .runtime import Runtime, RuntimeError
from .template import generate, cpp_format, TemplateError

# Set up logging
logging.basicConfig(level=logging.INFO if __import__('os').getenv('DG_JIT_DEBUG') else logging.WARNING)

# Export public API
__all__ = [
    'build', 'generate', 'get_nvcc_compiler', 'cpp_format', 'clear_cache',
    'Runtime', 'JITCompilationError', 'ValidationError', 'RuntimeError', 'TemplateError'
]
