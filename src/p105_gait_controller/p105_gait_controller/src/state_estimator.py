from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Protocol
import numpy as np

from .robot_state import RobotState
from .gait_scheduler import LegPhase

type Twist3 = Tuple[float, float, float]  # (vx, vy, wz) in body frame
type Contacts4 = Tuple[bool, bool, bool, bool]  # (FL, FR, RL, RR)


@dataclass(frozen=True)
class EstimatedState:
    """
    base_twist:
      Base linear velocity and yaw rate in body frame.

    com_height:
      Approx COM height (or base height proxy). For Route-1 you can keep a nominal constant.

    contacts:
      Per-leg contact flags (FL, FR, RL, RR).
    """
    base_twist: Twist3
    com_height: float
    contacts: Contacts4


class StateEstimator(Protocol):
    """
    Estimator interface. Controllers should depend on this, not on ROS topics.
    """

    def set_cmd_vel(self, cmd_vel: Twist3) -> None: ...
    def set_phases(self, phases: Tuple[LegPhase, LegPhase, LegPhase, LegPhase]) -> None: ...

    def push_odom_twist(self, odom_twist: Twist3, stamp_sec: Optional[float] = None) -> None: ...
    def push_imu_gyro(self, gyro_xyz: Tuple[float, float, float], stamp_sec: Optional[float] = None) -> None: ...

    def update(self, robot_state: RobotState, dt: float) -> None: ...
    def get_estimated_state(self) -> EstimatedState: ...


class LowPassFilter3:
    """
    Simple first-order low-pass filter for 3D signals.
    y = y + alpha*(x - y), alpha = dt/(tau+dt)

    tau: time constant in seconds. Larger => smoother, more delay.
    """
    def __init__(self, tau: float = 0.08):
        self.tau = float(max(tau, 1e-6))
        self.y = np.zeros(3, dtype=np.float64)
        self.initialized = False

    def reset(self, x: Tuple[float, float, float]) -> None:
        self.y[:] = np.asarray(x, dtype=np.float64)
        self.initialized = True

    def step(self, x: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        dt = float(max(dt, 1e-6))
        if not self.initialized:
            self.reset((float(x_arr[0]), float(x_arr[1]), float(x_arr[2])))
        alpha = dt / (self.tau + dt)
        self.y += alpha * (x_arr - self.y)
        return (float(self.y[0]), float(self.y[1]), float(self.y[2]))


class DummyEstimator(StateEstimator):
    """
    Route-1 friendly estimator:
      - base_twist := cmd_vel (disables Raibert feedback but keeps interface correct)
      - contacts := phases.is_stance (or default True if phases not set)
      - com_height := nominal constant

    Use this first to get walking. Swap later with OdomEstimator/EKF without touching controllers.
    """
    def __init__(self, nominal_height: float = 0.15):
        self._cmd_vel: Twist3 = (0.0, 0.0, 0.0)
        self._contacts: Contacts4 = (True, True, True, True)
        self._com_height = float(nominal_height)
        self._est = EstimatedState(base_twist=self._cmd_vel, com_height=self._com_height, contacts=self._contacts)

    def set_cmd_vel(self, cmd_vel: Twist3) -> None:
        self._cmd_vel = (float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2]))

    def set_phases(self, phases: Tuple[LegPhase, LegPhase, LegPhase, LegPhase]) -> None:
        self._contacts = tuple(bool(p.is_stance) for p in phases)  # type: ignore

    def push_odom_twist(self, odom_twist: Twist3, stamp_sec: Optional[float] = None) -> None:
        pass

    def push_imu_gyro(self, gyro_xyz: Tuple[float, float, float], stamp_sec: Optional[float] = None) -> None:
        pass

    def update(self, robot_state: RobotState, dt: float) -> None:
        # [TODO] for now base_twist = cmd_vel
        self._est = EstimatedState(base_twist=self._cmd_vel, com_height=self._com_height, contacts=self._contacts)

    def get_estimated_state(self) -> EstimatedState:
        return self._est


class OdomEstimator(StateEstimator):
    """
    Estimator using external odom twist as v_meas:
      - base_twist := filtered odom_twist (vx, vy, wz) in body frame
      - contacts := phases.is_stance (simple proxy)
      - com_height := nominal constant (or later derive from pose)

    You feed odom in via push_odom_twist() from your ROS subscriber callback.
    """
    def __init__(self, nominal_height: float = 0.15, twist_lpf_tau: float = 0.08):
        self._cmd_vel: Twist3 = (0.0, 0.0, 0.0)
        self._contacts: Contacts4 = (True, True, True, True)
        self._com_height = float(nominal_height)

        self._odom_twist_latest: Optional[Twist3] = None
        self._odom_stamp_latest: Optional[float] = None
        self._twist_lpf = LowPassFilter3(tau=twist_lpf_tau)

        self._est = EstimatedState(base_twist=(0.0, 0.0, 0.0), com_height=self._com_height, contacts=self._contacts)

    def set_cmd_vel(self, cmd_vel: Twist3) -> None:
        self._cmd_vel = (float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2]))

    def set_phases(self, phases: Tuple[LegPhase, LegPhase, LegPhase, LegPhase]) -> None:
        self._contacts = tuple(bool(p.is_stance) for p in phases)  # type: ignore

    def push_odom_twist(self, odom_twist: Twist3, stamp_sec: Optional[float] = None) -> None:
        self._odom_twist_latest = (float(odom_twist[0]), float(odom_twist[1]), float(odom_twist[2]))
        self._odom_stamp_latest = None if stamp_sec is None else float(stamp_sec)

    def push_imu_gyro(self, gyro_xyz: Tuple[float, float, float], stamp_sec: Optional[float] = None) -> None:
        # optional: you can fuse yaw rate later. For now we ignore or use odom's wz.
        pass

    def update(self, robot_state: RobotState, dt: float) -> None:
        # choose measured twist source
        if self._odom_twist_latest is None:
            # fallback: use cmd_vel (disables feedback)
            base_twist = self._cmd_vel
        else:
            base_twist = self._twist_lpf.step(self._odom_twist_latest, dt)

        self._est = EstimatedState(base_twist=base_twist, com_height=self._com_height, contacts=self._contacts)

    def get_estimated_state(self) -> EstimatedState:
        return self._est
