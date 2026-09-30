"""Compatibility shims for old pins (timm==0.3.2) on modern PyTorch."""

import collections.abc
import sys
import types


def _ensure_torch_six():
    if "torch._six" in sys.modules:
        return
    six = types.ModuleType("torch._six")
    six.container_abcs = collections.abc
    sys.modules["torch._six"] = six


_ensure_torch_six()
