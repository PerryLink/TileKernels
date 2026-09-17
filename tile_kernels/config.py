import functools
from typing import Optional

import torch

_num_sms = 0


@functools.lru_cache(maxsize=None)
def get_device_num_sms(device_index: Optional[int] = None) -> int:
    # The device index is part of the cache key: properties are looked up on the
    # requested device and a second device never reuses the first device's value.
    if device_index is None:
        device_index = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device_index)
    return prop.multi_processor_count


def set_num_sms(num_sms: int) -> None:
    global _num_sms
    assert 0 < num_sms <= get_device_num_sms()
    _num_sms = num_sms


def get_num_sms(device_index: Optional[int] = None) -> int:
    # `_num_sms` set through `set_num_sms` is a process-global override; it wins
    # over the per-device value.
    global _num_sms
    if _num_sms == 0:
        return get_device_num_sms(device_index)
    return _num_sms


@functools.lru_cache(maxsize=None)
def get_max_smem_per_sm(device_index: Optional[int] = None) -> int:
    if device_index is None:
        device_index = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device_index)
    return prop.shared_memory_per_multiprocessor
