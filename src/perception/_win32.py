"""Win32 kernel32 access layer.

Isolates platform-specific ctypes calls behind a sys.platform guard.
On Windows, loads the real kernel32 DLL. On other platforms, provides
None stubs so the module is importable for type checking without
requiring type: ignore annotations in consumer code.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

if sys.platform == "win32":
    kernel32: Any = ctypes.WinDLL("kernel32", use_last_error=True)

    def get_last_error() -> int:
        return ctypes.get_last_error()
else:
    kernel32: Any = None

    def get_last_error() -> int:
        return 0


PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
