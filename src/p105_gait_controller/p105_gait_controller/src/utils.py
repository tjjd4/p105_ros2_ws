from typing import Tuple
import numpy as np

type Vec3 = np.ndarray  # (3,)
type Mat3 = np.ndarray  # (3,3)

def _is_finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))

def _v3(x: float = 0.0) -> Vec3:
    """Create a 3D vector."""
    return np.array([x, x, x], dtype=np.float64)


def _m3(x: float = 0.0) -> Mat3:
    """Create a 3x3 matrix (diagonal if x != 0)."""
    if x == 0.0:
        return np.zeros((3, 3), dtype=np.float64)
    return np.diag([x, x, x]).astype(np.float64)

def _as_v3(x) -> Vec3:
    a = np.asarray(x, dtype=np.float64).reshape(3,)
    return a


def _check_shape(v: np.ndarray, shape: Tuple[int, ...], name: str) -> None:
    if v.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {v.shape}")


