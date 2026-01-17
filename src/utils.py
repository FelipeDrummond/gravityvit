"""Utility functions for GravityViT."""

import torch


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate PyTorch device based on configuration and availability.

    Args:
        device_config: Device configuration string.
            - "auto": Automatically detect best available device (cuda > mps > cpu)
            - "cuda": Use NVIDIA GPU
            - "mps": Use Apple Silicon GPU
            - "cpu": Use CPU only

    Returns:
        torch.device: The selected device.

    Raises:
        ValueError: If specified device is not available.
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda")

    if device_config == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return torch.device("mps")

    if device_config == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device config: {device_config}")


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        dict: Information about available devices.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_available": True,
    }

    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info
