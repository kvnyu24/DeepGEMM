import torch

from . import jit
from .jit_kernels import (
    gemm_fp8_fp8_bf16_nt,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    ceil_div,
    set_num_sms, get_num_sms,
    get_col_major_tma_aligned_tensor,
    get_m_alignment_for_contiguous_layout
)
from .utils import bench, bench_kineto, calc_diff

# Export error handling classes from jit module
from .jit import (
    JITCompilationError, ValidationError, 
    RuntimeError as JITRuntimeError, 
    TemplateError
)

__all__ = [
    # Core functions
    'gemm_fp8_fp8_bf16_nt',
    'm_grouped_gemm_fp8_fp8_bf16_nt_contiguous',
    'm_grouped_gemm_fp8_fp8_bf16_nt_masked',
    
    # Utility functions
    'ceil_div',
    'set_num_sms',
    'get_num_sms',
    'get_col_major_tma_aligned_tensor',
    'get_m_alignment_for_contiguous_layout',
    'bench',
    'bench_kineto',
    'calc_diff',
    
    # Error classes
    'JITCompilationError',
    'ValidationError',
    'JITRuntimeError',
    'TemplateError',
    
    # Sub-modules
    'jit'
]

# Package version
__version__ = '1.0.1'
