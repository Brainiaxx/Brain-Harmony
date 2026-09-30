import importlib.metadata
from functools import lru_cache

import torch
from packaging import version


def _flash_attn_version():
    try:
        return version.parse(importlib.metadata.version("flash_attn"))
    except importlib.metadata.PackageNotFoundError:
        return None


def is_flash_attn_2_available():
    if not torch.version.cuda:
        return False
    installed = _flash_attn_version()
    return installed is not None and installed >= version.parse("2.1.0")


@lru_cache()
def is_flash_attn_greater_or_equal(library_version: str):
    installed = _flash_attn_version()
    return installed is not None and installed >= version.parse(library_version)


@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    return is_flash_attn_greater_or_equal("2.1.0")
