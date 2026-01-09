from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class TopicsParams:
    joint_states: str = "/joint_states"
    cmd_vel: str = "/cmd_vel"
    odom: str = "/odom"
    command_out: str = "/forward_position_controller/commands"

@dataclass(frozen=True)
class ModelParams:
    """
    Robot model parameters.
    - nominal_height: nominal base height (meters)
    - l1, l2, l3, l4: leg link lengths (hip, thigh, calf, knee offset)
    """
    nominal_height: float = 0.15  # meters
    l1: float = 0.083  # hip link length
    l2: float = 0.213  # thigh link length
    l3: float = 0.213  # calf link length
    l4: float = 0.0    # knee lateral offset

@dataclass(frozen=True)
class GaitControllerParams:
    """
    Gait controller parameters.
    - control_dt: control loop time step (seconds)
    - cart_kp: Cartesian PD position gain
    - cart_kd: Cartesian PD damping gain
    """
    control_dt: float = 0.005  # seconds
    cart_kp: float = 100.0  # Cartesian position gain
    cart_kd: float = 5.0    # Cartesian damping gain

@dataclass(frozen=True)
class GaitSchedulerParams:
    """
    frequency_hz:
      Gait cycle frequency. Example: 2.0 Hz -> one gait cycle = 0.5 s

    duty_factor:
      Fraction of gait cycle in stance (0..1). 站立週期比例(0..1)
      Example:
        0.6 -> 60% stance, 40% swing.

    phase_offsets:
      Per-leg phase offsets in [0,1). Example trotting:
        FL = 0.0, RR = 0.0
        FR = 0.5, RL = 0.5
    """
    cycle_frequency_hz: float = 2.0
    duty_factor: float = 0.6
    phase_offsets: Tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)

    def __post_init__(self):
        if self.cycle_frequency_hz <= 0.0:
            raise ValueError("cycle_frequency_hz must be > 0")
        if not (0.0 < self.duty_factor < 1.0):
            raise ValueError("duty_factor must be in (0,1)")
        for o in self.phase_offsets:
            if not (0.0 <= o < 1.0):
                raise ValueError("phase_offsets must be in [0,1)")


@dataclass(frozen=True)
class FootTrajectoryParams:
    """
    Swing foot trajectory parameters (timing + shape).

    - swing_height: lift height (meters) added along +z of hip frame
    - clearance_fraction: where to place the peak (0.5 means mid-swing)

    Raibert-style touchdown planning parameters:
    - vel_gain: to adjust foot placement
    - placement_time_fraction: fraction of swing time ahead to place foot
    """
    swing_height: float = 0.05  # meters
    control_point_fraction: float = 0.5  # (0..1) location of control points along swing path
    # stance behavior (hold current position vs track a target)
    stance_hold: bool = True

    # Raibert foot placement gains (for adjusting touchdown targets)
    vel_gain: float = 1.0  # K
    placement_time_fraction: float = 0.5  # T = placement_time_fraction * stance_time

    # Safety caps (meters)
    max_step_xy: float = 0.08  # max step update meters in x/y direction


@dataclass(frozen=True)
class LegInterfaceParams:
    """
    Parameters for leg interface (position output).
    """

    # IK solver (damped least squares)
    ik_mode: str = "position"    # "position" or "velocity" - IK control mode
    ik_lambda: float = 1e-3      # damping (DLS 阻尼係數)
    ik_step_scale: float = 1.0   # update step size scaling on dq (步伐更新比例)

    # Safety limits
    max_dq_per_step: float = 0.15  # max update rad per control step (per joint)
    joint_pos_min: Tuple[float, float, float] = (-2.5, -2.5, -2.8)
    joint_pos_max: Tuple[float, float, float] = ( 2.5,  2.5,  0.2)

    # When state is stale, hold last command (or output zeros if not available)
    state_timeout_sec: float = 0.15

    # [Optional] position control outputs q_des only; kp, kd for torque control
    default_hold_kp: float = 0.0
    default_hold_kd: float = 0.0

    # Nominal stance foot x,y offsets in hip frame [x, y] for each leg
    # z will be computed from model.nominal_height
    nominal_stance_fl: Tuple[float, float] = (0.15, 0.12)
    nominal_stance_fr: Tuple[float, float] = (0.15, -0.12)
    nominal_stance_rl: Tuple[float, float] = (-0.15, 0.12)
    nominal_stance_rr: Tuple[float, float] = (-0.15, -0.12)

@dataclass(frozen=True)
class LocomotionParams:
    topics_params: TopicsParams = TopicsParams()
    model_params: ModelParams = ModelParams()
    gait_controller_params: GaitControllerParams = GaitControllerParams()
    gait_scheduler_params: GaitSchedulerParams = GaitSchedulerParams()
    foot_trajectory_params: FootTrajectoryParams = FootTrajectoryParams()
    leg_interface_params: LegInterfaceParams = LegInterfaceParams()