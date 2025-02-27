import os
import unittest
import torch
from typing import Any

from deep_gemm import jit

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()


class Capture:
    def __init__(self) -> None:
        self.read_fd = None
        self.write_fd = None
        self.saved_stdout = None
        self.captured = None

    def __enter__(self) -> Any:
        self.read_fd, self.write_fd = os.pipe()
        self.saved_stdout = os.dup(1)
        os.dup2(self.write_fd, 1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        os.dup2(self.saved_stdout, 1)
        os.close(self.write_fd)
        with os.fdopen(self.read_fd, 'r') as f:
            self.captured = f.read()

    def capture(self) -> str:
        return self.captured


@unittest.skipIf(not CUDA_AVAILABLE, "CUDA not available")
class JITTests(unittest.TestCase):
    def test_nvcc_find(self) -> None:
        with Capture() as c:
            print(jit.get_nvcc_compiler())
        
        # There should be no error here, so the output should not be empty
        assert len(c.capture()) > 0


if __name__ == "__main__":
    unittest.main()
