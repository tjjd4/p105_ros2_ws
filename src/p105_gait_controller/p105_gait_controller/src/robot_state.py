from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading
import numpy as np


DEFAULT_JOINT_ORDER: List[str] = [
    # ros2_control: FL -> FR -> RL -> RR (3 joints each)
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

LEG_NAMES = ["FL", "FR", "RL", "RR"]


@dataclass(frozen=True)
class RobotStateSnapshot:
    """
    Immutable snapshot for control loop usage.
    """
    stamp_sec: float
    q: np.ndarray            # (12,)
    qd: np.ndarray           # (12,)

    joint_names: Tuple[str, ...] = tuple(DEFAULT_JOINT_ORDER)

    def leg_q(self, leg_index: int) -> np.ndarray:
        i = leg_index * 3
        return self.q[i:i+3].copy()

    def leg_qd(self, leg_index: int) -> np.ndarray:
        i = leg_index * 3
        return self.qd[i:i+3].copy()


@dataclass
class RobotState:
    joint_order: List[str] = field(default_factory=lambda: list(DEFAULT_JOINT_ORDER))

    _q: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64), init=False)
    _qd: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64), init=False)
    _have_state: bool = field(default=False, init=False)

    _last_stamp_sec: float = field(default=0.0, init=False)
    _last_update_wall_sec: float = field(default=0.0, init=False)

    _name_to_index: Dict[str, int] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._rebuild_index_map()

    def _rebuild_index_map(self) -> None:
        self._name_to_index = {name: i for i, name in enumerate(self.joint_order)}
        if len(self._name_to_index) != len(self.joint_order):
            raise ValueError("joint_order contains duplicate joint names")

        if len(self.joint_order) != 12:
            raise ValueError(f"Expected 12 joints in joint_order, got {len(self.joint_order)}")


    def set_joint_order(self, joint_order: List[str]) -> None:
        """
        Change the expected joint ordering
        """
        with self._lock:
            self.joint_order = list(joint_order)
            self._rebuild_index_map()
            # reset
            self._q[:] = 0.0
            self._qd[:] = 0.0
            self._have_state = False
            self._last_stamp_sec = 0.0
            self._last_update_wall_sec = 0.0

    def update_from_joint_state(
        self,
        *,
        names: List[str],
        positions: List[float],
        velocities: Optional[List[float]],
        stamp_sec: float,
        wall_time_sec: float,
    ) -> bool:
        """
        Update internal q/qd from a JointState message fields.

        Returns:
            True if update succeeded (all joints found),
            False if message is missing required joints.
        """
        # fast path: build msg name -> msg idx
        msg_index = {n: i for i, n in enumerate(names)}

        with self._lock:
            required = self._name_to_index.keys()

            # ensure required joints exist
            for jn in required:
                if jn not in msg_index:
                    return False
            for jn, target_i in self._name_to_index.items():
                mi = msg_index[jn]
                self._q[target_i] = float(positions[mi])

                if velocities is not None and len(velocities) == len(names):
                    self._qd[target_i] = float(velocities[mi])
                else:
                    # If velocity not provided, keep previous (or zero if never had state)
                    if not self._have_state:
                        self._qd[target_i] = 0.0

            self._have_state = True
            self._last_stamp_sec = float(stamp_sec)
            self._last_update_wall_sec = float(wall_time_sec)

        return True

    def have_state(self) -> bool:
        with self._lock:
            return self._have_state

    def age_sec(self, wall_time_sec: float) -> float:
        """
        Age since last update in wall time (useful for timeout safety).
        """
        with self._lock:
            if not self._have_state:
                return float("inf")
            return max(0.0, float(wall_time_sec) - self._last_update_wall_sec)

    def snapshot(self) -> Optional[RobotStateSnapshot]:
        """
        Get an immutable snapshot for control loop usage.
        Returns:
            RobotStateSnapshot if robot have state,
            None if no state received yet.
        """
        with self._lock:
            if not self._have_state:
                return None
            return RobotStateSnapshot(
                stamp_sec=self._last_stamp_sec,
                q=self._q.copy(),
                qd=self._qd.copy(),
                joint_names=tuple(self.joint_order),
            )

    # ----------------------------
    # Helper functions
    # ----------------------------

    def leg_slice(self, leg_index: int) -> slice:
        if leg_index < 0 or leg_index > 3:
            raise ValueError("leg_index must be 0..3 (FL, FR, RL, RR)")
        i = leg_index * 3
        return slice(i, i + 3)

    def joint_indices_for_leg(self, leg_name: str) -> Tuple[int, int, int]:
        """
        Returns indices in [0..11] for (hip, thigh, calf) for the given leg.
        """
        leg_name = leg_name.upper()
        if leg_name not in LEG_NAMES:
            raise ValueError(f"leg_name must be one of {LEG_NAMES}")

        base = LEG_NAMES.index(leg_name) * 3
        return (base + 0, base + 1, base + 2)
