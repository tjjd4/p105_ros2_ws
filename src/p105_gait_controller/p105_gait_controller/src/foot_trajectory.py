from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .leg_command import LegData
from .gait_scheduler import LegPhase
from .params import FootTrajectoryParams
from .utils import _as_v3, _v3


type Vec3 = np.ndarray  # (3,)


def _bezier3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, s: float) -> Vec3:
    """
    Cubic Bezier curve point at parameter s in [0,1].
    """
    s = float(np.clip(s, 0.0, 1.0))
    u = 1.0 - s
    return (u*u*u)*p0 + 3*(u*u*s)*p1 + 3*(u*s*s)*p2 + (s*s*s)*p3


def _d_bezier3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, s: float) -> Vec3:
    """
    Cubic Bezier derivative (dp/ds).
    v_des = dp/ds * ds/dt
    速度= Bezier 點對 s 的微分 * s 對時間的微分
    """
    s = float(np.clip(s, 0.0, 1.0))
    u = 1.0 - s
    # derivative of cubic Bezier
    return 3*(u*u)*(p1 - p0) + 6*(u*s)*(p2 - p1) + 3*(s*s)*(p3 - p2)


@dataclass
class SwingState:
    """
    Per-leg state cached at the beginning of swing across control steps to make swing stable.
    """
    active: bool = False  # is currently in swing
    p0: Optional[Vec3] = None  # lift-off foot position
    p1: Optional[Vec3] = None  # desired touchdown position


class FootTrajectoryGenerator:
    """
    Generates per-leg foot targets p_des, v_des in hip frame.
    Uses cubic Bezier curves for swing trajectory.
    """

    def __init__(self, params: FootTrajectoryParams = FootTrajectoryParams(), *, num_legs: int = 4):
        self.params = params
        self.num_legs = int(num_legs)
        self._swing_states: List[SwingState] = [SwingState() for _ in range(self.num_legs)]

    def reset(self) -> None:
        self._swing_states = [SwingState() for _ in range(self.num_legs)]

    def update(
        self,
        *,
        phases: List[LegPhase],
        leg_datas: List[LegData],
        frequency_hz: float,
        duty_factor: float,
        # command velocity for moving in body/hip frame
        cmd_vel: Optional[Tuple[float, float, float]] = None,  # (vx, vy, yaw_rate)
        # Optional: measured velocity of robot body CoM
        com_vel: Optional[Tuple[float, float, float]] = None,  # (vx, vy, yaw_rate)
        # Optional: allow you to provide touchdown targets externally (planner/WBC)
        touchdown_targets: Optional[List[Vec3]] = None,
        # Optional: nominal stance targets if stance need to go to a fixed pose
        stance_targets: Optional[List[Vec3]] = None,
    ) -> Tuple[List[Vec3], List[Vec3]]:
        """
        Given per-leg phases and leg data, compute desired foot p_des, v_des.

        - swing duration = (1-duty_factor)/frequency_hz
        - v_des is dp/dt (meters/sec) in the same frame as p_des

        Returns:
          p_des_list: position List[Vec3] of num_legs
          v_des_list: velocity List[Vec3] of num_legs
        """
        if len(phases) != self.num_legs:
            raise ValueError(f"phases length must be {self.num_legs}")
        if len(leg_datas) != self.num_legs:
            raise ValueError(f"leg_datas length must be {self.num_legs}")
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be > 0")
        if not (0.0 < duty_factor < 1.0):
            raise ValueError("duty_factor must be in (0,1)")
        if touchdown_targets is not None and len(touchdown_targets) != self.num_legs:
            raise ValueError(f"touchdown_targets length must be {self.num_legs}")
        if stance_targets is not None and len(stance_targets) != self.num_legs:
            raise ValueError(f"stance_targets length must be {self.num_legs}")
        if cmd_vel is not None and len(cmd_vel) != 3:
            raise ValueError("cmd_vel must be length 3 (vx, vy, yaw_rate)")
        if com_vel is not None and len(com_vel) != 3:
            raise ValueError("com_vel must be length 3 (vx, vy, yaw_rate)")

        # compute stance/swing times
        stance_time = float(duty_factor) / float(frequency_hz)
        swing_time = (1.0 - float(duty_factor)) / float(frequency_hz)
        stance_time = max(stance_time, 1e-4)
        swing_time = max(swing_time, 1e-4)


        p_des_list: List[Vec3] = []
        v_des_list: List[Vec3] = []

        for leg in range(self.num_legs):

            ph = phases[leg]
            ld = leg_datas[leg]
            st = self._swing_states[leg]

            # Nominal stance targets (default to current foot positions if stance_targets not provided)
            stance_p_des: Vec3 = ld.p.copy()
            if stance_targets is not None:
                stance_p_des = _as_v3(stance_targets[leg])

            # touchdown target
            td: Optional[Vec3] = None
            # per-leg touchdown targets if:
            # - user provided touchdown_targets
            if touchdown_targets is not None:
                td = _as_v3(touchdown_targets[leg])
            # else if cmd_vel is provided
            elif cmd_vel is not None:
                vx, vy, wz = float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2])
                td = self._compute_touchdown_target(
                    stance_p_des=stance_p_des, vx=vx, vy=vy, wz=wz,
                    com_vel=com_vel,
                    stance_time=stance_time,
                )
            # else: td remains None, will default to nominal stance target later
            
            if ph.is_stance:
                # Touchdown edge: clear swing state
                if ph.just_touchdown:
                    st.active = False
                    st.p0 = None
                    st.p1 = None

                if self.params.stance_hold:
                    p_des = stance_p_des
                    v_des = _v3(0.0)
                else:
                    # [TODO] if not holding, implement target tracking
                    p_des = stance_p_des
                    v_des = _v3(0.0)

            else:
                # Swing
                if ph.just_liftoff or (not st.active):
                    # latch lift-off position from measured foot position
                    st.active = True
                    st.p0 = ld.p.copy()

                    # if no touchdown target provided, default is "return to stance target"
                    st.p1 = td.copy() if td is not None else stance_p_des.copy()

                if st.p0 is not None and st.p1 is not None:
                    p_des, v_des = self._swing_bezier(
                        p0=st.p0,
                        p1=st.p1,
                        swing_phase=ph.swing_phase,
                        swing_time=swing_time,
                    )
                else:
                    # Fallback (should never happen in normal operation)
                    p_des = stance_p_des
                    v_des = _v3(0.0)

            p_des_list.append(p_des)
            v_des_list.append(v_des)

        return p_des_list, v_des_list

    def _compute_touchdown_target(
        self,
        *,
        stance_p_des: Vec3,
        vx: float,
        vy: float,
        wz: float,
        stance_time: float,
        com_vel: Optional[Tuple[float, float, float]] = None,
    ) -> Vec3:
        """
        Compute touchdown target position using Raibert Heuristic foot point algorithm.
        """
        if com_vel is not None:
            com_vx, com_vy, com_wz = com_vel
        else:
            com_vx, com_vy, com_wz = vx, vy, wz

        placement_time_fraction = float(self.params.placement_time_fraction)
        K_vel = float(self.params.vel_gain)

        T = max(1e-4, placement_time_fraction * stance_time)
        x_nom = stance_p_des[0]
        y_nom = stance_p_des[1]
        vx_cmd_leg = vx - wz * y_nom
        vy_cmd_leg = vy + wz * x_nom

        vx_meas_leg = com_vx - com_wz * y_nom
        vy_meas_leg = com_vy + com_wz * x_nom

        dx = vx_meas_leg * T
        dy = vy_meas_leg * T

        dx_feedback = K_vel * (vx_meas_leg - vx_cmd_leg)
        dy_feedback = K_vel * (vy_meas_leg - vy_cmd_leg)


        dx_total = dx + dx_feedback
        dy_total = dy + dy_feedback
        # Apply max step limits
        max_xy = float(self.params.max_step_xy)
        dx_total = float(np.clip(dx_total, -max_xy, +max_xy))
        dy_total = float(np.clip(dy_total, -max_xy, +max_xy))

        td_x = stance_p_des[0] + dx_total
        td_y = stance_p_des[1] + dy_total
        td_z = stance_p_des[2]  # keep nominal height
        

        return np.array([td_x, td_y, td_z], dtype=np.float64)

    def _swing_bezier(self, *, p0: Vec3, p1: Vec3, swing_phase: float, swing_time: float) -> Tuple[Vec3, Vec3]:
        """
        Smooth swing arc using cubic Bezier.

        Control points:
          P0 = p0 (lift-off)
          P3 = p1 (touchdown)
          P1/P2 are elevated by swing_height.
        """
        p0 = _as_v3(p0)
        p1 = _as_v3(p1)

        s = float(np.clip(swing_phase, 0.0, 1.0))

        # Place control points along the line with an added vertical clearance
        # fraction controls how early/late the foot lifts/descends
        frac = float(np.clip(self.params.control_point_fraction, 0.05, 0.95))
        P0 = p0
        P3 = p1

        P1 = (1.0 - frac) * P0 + frac * P3
        P2 = frac * P0 + (1.0 - frac) * P3

        # Add swing height along +z (if hip frame z is up, else change `+=` to `-=`)
        P1 = P1.copy()
        P2 = P2.copy()
        P1[2] += float(self.params.swing_height)
        P2[2] += float(self.params.swing_height)

        p = _bezier3(P0, P1, P2, P3, s)
        dp_ds = _d_bezier3(P0, P1, P2, P3, s)

        # Convert derivative wrt s to derivative wrt time: dp/dt = dp/ds * ds/dt
        # s goes from 0..1 over swing_time seconds, so ds/dt = 1/swing_time
        v = dp_ds * (1.0 / float(swing_time))

        return p, v
