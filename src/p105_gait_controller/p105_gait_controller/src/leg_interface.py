from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .leg_command import LegCommand, LegData, make_leg_commands, make_leg_datas, ControlMode
from .params import LegInterfaceParams, ModelParams
from .kinematics import AnalyticKinematics, QuadrupedGeom, make_default_geom
from .gait_scheduler import LegPhase
from .utils import _is_finite, Vec3

class LegInterface:
    """
    leg interface (position output)

    Inputs (each control cycle):
      - LegData[4]: q, qd, p, v, J
      - LegCommand[4]: desired p/v (cartesian) or q/qd (joint), plus gains (optional)

    Output:
      - q_des_full: np.ndarray shape (12,)
    """
    def __init__(self, model_params: ModelParams, params: LegInterfaceParams = LegInterfaceParams()):
        self.params = params
        self.model_params = model_params
        self.commands: List[LegCommand] = make_leg_commands(4)
        self.leg_datas: List[LegData] = make_leg_datas(4)

        # Initialize kinematics module
        geom = make_default_geom(
            l1=model_params.l1,
            l2=model_params.l2,
            l3=model_params.l3,
            l4=model_params.l4,
        )
        self.kinematics = AnalyticKinematics(geom)

        self._enabled: bool = False
        self._last_q_des: Optional[np.ndarray] = None  # (12,)

        # cached joint limits (per-leg)
        self._q_min_leg = np.array(self.params.joint_pos_min, dtype=np.float64).reshape(3,)
        self._q_max_leg = np.array(self.params.joint_pos_max, dtype=np.float64).reshape(3,)

        # Nominal stance targets: compute z from nominal_height
        stance_z = -model_params.nominal_height  # feet below body
        self._nominal_stance_targets = [
            np.array([self.params.nominal_stance_fl[0], self.params.nominal_stance_fl[1], stance_z], dtype=np.float64),
            np.array([self.params.nominal_stance_fr[0], self.params.nominal_stance_fr[1], stance_z], dtype=np.float64),
            np.array([self.params.nominal_stance_rl[0], self.params.nominal_stance_rl[1], stance_z], dtype=np.float64),
            np.array([self.params.nominal_stance_rr[0], self.params.nominal_stance_rr[1], stance_z], dtype=np.float64),
        ]


    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def zero_command(self) -> None:
        """
        Clear all commands every cycle before filling new ones.
        """
        for c in self.commands:
            c.zero()

    def set_cartesian_target(
        self,
        leg: int,
        *,
        p_des: Vec3,
        v_des: Optional[Vec3] = None,
        kp: float = 0.0,
        kd: float = 0.0,
        force_ff: Optional[Vec3] = None,
    ) -> None:
        """
        Fill one leg's `cartesian` target and diagonal gains. (by position, velocity)
        """
        c = self.commands[leg]
        c.control_mode = ControlMode.CARTESIAN_PD
        c.p_des[:] = np.asarray(p_des, dtype=np.float64).reshape(3,)
        if v_des is not None:
            c.v_des[:] = np.asarray(v_des, dtype=np.float64).reshape(3,)
        else:
            c.v_des[:] = 0.0

        c.set_cart_gains_diag(kp=kp, kd=kd)
        if force_ff is not None:
            c.force_ff[:] = np.asarray(force_ff, dtype=np.float64).reshape(3,)
        else:
            c.force_ff[:] = 0.0

    def set_joint_target(
        self,
        leg: int,
        *,
        q_des: Vec3,
        qd_des: Optional[Vec3] = None,
        kp: float = 0.0,
        kd: float = 0.0,
        tau_ff: Optional[Vec3] = None,
    ) -> None:
        """
        Fill one leg's `joint` target and diagonal gains. (by angle, angle velocity)
        """
        c = self.commands[leg]
        c.control_mode = ControlMode.JOINT_PD
        c.q_des[:] = np.asarray(q_des, dtype=np.float64).reshape(3,)
        if qd_des is not None:
            c.qd_des[:] = np.asarray(qd_des, dtype=np.float64).reshape(3,)
        else:
            c.qd_des[:] = 0.0

        c.set_joint_gains_diag(kp=kp, kd=kd)
        if tau_ff is not None:
            c.tau_ff[:] = np.asarray(tau_ff, dtype=np.float64).reshape(3,)
        else:
            c.tau_ff[:] = 0.0

    def update(
        self,
        *,
        leg_datas: List[LegData],
        phases: List[LegPhase],
    ) -> np.ndarray:
        """
        [Main] Compute q_des(12) for position controller.

        - If disabled: output last q_des (hold) or current q (if available)
        - If state is stale: hold last q_des
        - Otherwise:
            For each leg:
              1. If joint target was explicitly set -> use q_des directly
              2. Else if cartesian target was set -> do Jacobian IK step from current q
              3. Apply per-step dq limit + joint limits
        """
        # If no leg data, safest fallback
        if leg_datas is None or len(leg_datas) != 4:
            return self._fallback_output()

        # Disabled: hold posture (do not keep applying new steps)
        if not self._enabled:
            return self._hold_or_current(leg_datas)

        q_des_full = np.zeros(12, dtype=np.float64)

        # Compute per-leg q_des
        for leg in range(4):
            ld = leg_datas[leg]
            cmd = self.commands[leg]

            # Basic sanity checks
            if not (_is_finite(ld.q) and _is_finite(ld.p) and _is_finite(ld.J) and _is_finite(cmd.q_des) and _is_finite(cmd.p_des)):
                # If something is NaN or Inf, hold current
                q_leg_des = ld.q.copy()
            else:
                q_leg_des = self._compute_leg_q_des(leg=leg, ld=ld, lp=phases[leg], cmd=cmd)

            # Clamp to joint limits
            q_leg_des = np.clip(q_leg_des, self._q_min_leg, self._q_max_leg)

            q_des_full[leg * 3: leg * 3 + 3] = q_leg_des

        self._last_q_des = q_des_full.copy()
        return q_des_full


    def _compute_leg_q_des(self, *, leg: int, ld: LegData, lp: LegPhase, cmd: LegCommand) -> np.ndarray:
        if cmd.control_mode == ControlMode.JOINT_PD:
            return cmd.q_des.copy()

        elif cmd.control_mode == ControlMode.CARTESIAN_PD:
            if lp.is_stance:
                # stance: position-incremental IK (small safe steps)
                return self._ik_step_dls(ld=ld, cmd=cmd)
            else:
                # swing: analytic IK (direct solve) + rate limit + fallback
                return self._ik_analytic(leg=leg, ld=ld, cmd=cmd)

        else:  # DISABLED
            return ld.q.copy()

    def _ik_step_dls(self, *, ld: LegData, cmd: LegCommand) -> np.ndarray:
        """
        One-step damped least squares Jacobian IK with position-based incremental mode:

        delta_p = position_gain * (p_des - p)
        dq = J^T * (J*J^T + λ^2 I)^-1 * delta_p
        """
        p = ld.p.reshape(3,)
        v = ld.v.reshape(3,)
        J = ld.J.reshape(3, 3)
        q = ld.q.reshape(3,)

        # Position error with exponential convergence
        # Use ik_step_scale as position gain (0.3 = move 30% of error per step)
        p_error = cmd.p_des.reshape(3,) - p
        position_gain = float(self.params.ik_step_scale)
        delta_p = position_gain * p_error

        if not _is_finite(delta_p):
            return q.copy()

        lam = float(self.params.ik_lambda)
        A = J @ J.T + (lam * lam) * np.eye(3, dtype=np.float64)

        # Solve for joint-space delta: dq = J^T * (J*J^T + λ^2*I)^-1 * delta_p
        try:
            y = np.linalg.solve(A, delta_p)
        except np.linalg.LinAlgError:
            return q.copy()

        dq = J.T @ y  # No step_scale here - already applied to delta_p

        # Per-step dq limit (safety clipping)
        dq = np.clip(dq, -self.params.max_dq_per_step, self.params.max_dq_per_step)

        q_next = q + dq
        return q_next
    
    def _ik_analytic(self, *, leg: int, ld: LegData, cmd: LegCommand) -> np.ndarray:
        p_des = cmd.p_des.reshape(3,)
        q = ld.q.reshape(3,)

        # analytic IK
        try:
            q_sol = self.kinematics.ik_leg(leg_index=leg, p_des=p_des, q_seed=q)
        except Exception:
            q_sol = None

        if (q_sol is None) or (not _is_finite(q_sol)):
            # fallback: do incremental step instead of hard failing
            return self._ik_step_dls(ld=ld, cmd=cmd)

        q_sol = np.asarray(q_sol, dtype=np.float64).reshape(3,)

        # rate-limit per tick to avoid jumps (important for analytic IK)
        if self._last_q_des is not None:
            q_last = self._last_q_des[leg*3:leg*3+3].copy()
        else:
            q_last = q.copy()

        dq = q_sol - q_last
        dq = np.clip(dq, -self.params.max_dq_per_step, self.params.max_dq_per_step)
        q_next = q_last + dq
        return q_next

    def _hold_or_current(self, leg_datas: List[LegData]) -> np.ndarray:
        """
        Hold last q_des if available; otherwise hold current q.
        """
        if self._last_q_des is not None:
            return self._last_q_des.copy()

        q_full = np.zeros(12, dtype=np.float64)
        for leg in range(4):
            q_full[leg*3:leg*3+3] = leg_datas[leg].q.copy()
        return q_full

    def _fallback_output(self) -> np.ndarray:
        """
        safest output: zeros or last command.
        """
        if self._last_q_des is not None:
            return self._last_q_des.copy()
        return np.zeros(12, dtype=np.float64)

    def compute_leg_data_from_joints(self, q: np.ndarray, qd: np.ndarray) -> List[LegData]:
        """
        Compute FK/Jacobian/velocity for all 4 legs from joint positions/velocities.
        
        Args:
            q: (12,) joint positions
            qd: (12,) joint velocities
        
        Returns:
            List[LegData] of length 4 with p, v, J filled
        """
        self.leg_datas = self.kinematics.update_leg_data(q, qd)
        return self.leg_datas

    def get_nominal_stance_targets(self) -> List[Vec3]:
        """
        Return nominal stance foot positions for all 4 legs in hip frame.
        
        Returns:
            List[Vec3] of length 4: [FL, FR, RL, RR]
        """
        return [t.copy() for t in self._nominal_stance_targets]
