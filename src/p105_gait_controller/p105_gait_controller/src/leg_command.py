from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List
import numpy as np
from .utils import _v3, _m3

class ControlMode(IntEnum):
    DISABLED = 0      # No Control
    FORCE_ONLY = 1      # Torque Control / Force Control
    JOINT_PD = 2        # Joint Space
    CARTESIAN_PD = 3    # Cartesian Space

@dataclass
class LegCommand:
    """
    - Cartesian targets: p_des, v_des, kp_cart, kd_cart (+ optional force_ff)
    - Joint targets: q_des, qd_des, kp_joint, kd_joint (+ optional tau_ff)
    """
    control_mode: ControlMode = ControlMode.DISABLED

    tau_ff: np.ndarray = field(default_factory=lambda: _v3(0.0))      # (3,)
    force_ff: np.ndarray = field(default_factory=lambda: _v3(0.0))    # (3,)

    q_des: np.ndarray = field(default_factory=lambda: _v3(0.0))       # (3,)
    qd_des: np.ndarray = field(default_factory=lambda: _v3(0.0))      # (3,)

    p_des: np.ndarray = field(default_factory=lambda: _v3(0.0))       # (3,)
    v_des: np.ndarray = field(default_factory=lambda: _v3(0.0))       # (3,)

    kp_cart: np.ndarray = field(default_factory=lambda: _m3(0.0))     # (3,3)
    kd_cart: np.ndarray = field(default_factory=lambda: _m3(0.0))     # (3,3)

    kp_joint: np.ndarray = field(default_factory=lambda: _m3(0.0))    # (3,3)
    kd_joint: np.ndarray = field(default_factory=lambda: _m3(0.0))    # (3,3)

    # set to zero
    def zero(self) -> None:
        self.control_mode = ControlMode.DISABLED
        self.tau_ff[:] = 0.0
        self.force_ff[:] = 0.0
        self.q_des[:] = 0.0
        self.qd_des[:] = 0.0
        self.p_des[:] = 0.0
        self.v_des[:] = 0.0
        self.kp_cart[:, :] = 0.0
        self.kd_cart[:, :] = 0.0
        self.kp_joint[:, :] = 0.0
        self.kd_joint[:, :] = 0.0

    def set_joint_gains_diag(self, kp: float, kd: float) -> None:
        self.kp_joint[:, :] = 0.0
        self.kd_joint[:, :] = 0.0
        np.fill_diagonal(self.kp_joint, kp)
        np.fill_diagonal(self.kd_joint, kd)

    def set_cart_gains_diag(self, kp: float, kd: float) -> None:
        self.kp_cart[:, :] = 0.0
        self.kd_cart[:, :] = 0.0
        np.fill_diagonal(self.kp_cart, kp)
        np.fill_diagonal(self.kd_cart, kd)


@dataclass
class LegData:
    q: np.ndarray = field(default_factory=lambda: _v3(0.0))           # (3,)
    qd: np.ndarray = field(default_factory=lambda: _v3(0.0))          # (3,)

    p: np.ndarray = field(default_factory=lambda: _v3(0.0))           # (3,) foot position
    v: np.ndarray = field(default_factory=lambda: _v3(0.0))           # (3,) foot velocity

    J: np.ndarray = field(default_factory=lambda: _m3(0.0))           # (3,3) Jacobian

    tau_est: np.ndarray = field(default_factory=lambda: _v3(0.0))     # (3,) for debug

    def zero(self) -> None:
        self.q[:] = 0.0
        self.qd[:] = 0.0
        self.p[:] = 0.0
        self.v[:] = 0.0
        self.J[:, :] = 0.0
        self.tau_est[:] = 0.0

# ----------------------------
# Helper functions
# ----------------------------

def make_leg_commands(num_legs: int = 4) -> List[LegCommand]:
    return [LegCommand() for _ in range(num_legs)]


def make_leg_datas(num_legs: int = 4) -> List[LegData]:
    return [LegData() for _ in range(num_legs)]
