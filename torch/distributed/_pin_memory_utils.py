# mypy: allow-untyped-defs
import logging
from logging import getLogger

import torch

logger = getLogger()
logger.setLevel(logging.INFO)


def pin_shared_mem(data_ptr: int, size: int) -> None:
    cudart = torch.cuda.cudart()

    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )

    if succ != 0:
        raise RuntimeError(f"Registering shared memory failed with cudaError: {succ}\n")


def unpin_memory(data_ptr) -> None:
    succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    assert succ == 0, f"Unpinning shared memory failed with error-code: {succ}"
