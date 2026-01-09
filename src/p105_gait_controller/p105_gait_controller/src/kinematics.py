from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, override
import numpy as np

from .leg_command import LegData
from .utils import _as_v3, _is_finite, _check_shape, Vec3, Mat3


@dataclass(frozen=True)
class LegGeom:
    """
    Per-leg geometry parameters used by the analytic FK/J model.

    Notes:
    - l1: hip link length
    - l2: thigh link length
    - l3: calf link length
    - l4: knee y-offset (optional lateral offset)
    - side_sign: +1 for left legs, -1 for right legs

    - frame convention: "hip frame" (same orientation as body frame),
      origin at abad pivot.
    """
    l1: float
    l2: float
    l3: float
    l4: float
    side_sign: float  # +1 left, -1 right


@dataclass(frozen=True)
class QuadrupedGeom:
    """
    Whole-robot geometry needed for kinematics.
    - leg_geoms: [FL, FR, RL, RR]
    """
    leg_geoms: Tuple[LegGeom, LegGeom, LegGeom, LegGeom]


# ----------------------------
# Analytic FK/J
# ----------------------------

def foot_position_and_jacobian(q_leg: Vec3, geom: LegGeom) -> Tuple[Vec3, Mat3]:
    """
    Compute foot position p and Jacobian J in the hip/leg frame for one 3DOF leg.

    q_leg: [q_abad, q_hip, q_knee] (rad)
    returns:
      p: (3,)
      J: (3,3) such that v = J * qd
    """
    q_leg = _as_v3(q_leg)

    l1, l2, l3, l4 = geom.l1, geom.l2, geom.l3, geom.l4
    sign = geom.side_sign

    q1, q2, q3 = q_leg

    s1, s2, s3 = np.sin(q1), np.sin(q2), np.sin(q3)
    c1, c2, c3 = np.cos(q1), np.cos(q2), np.cos(q3)

    c23 = c2 * c3 - s2 * s3
    s23 = s2 * c3 + c2 * s3

    # Jacobian
    J = np.zeros((3, 3), dtype=np.float64)
    J[0, 0] = 0.0
    J[0, 1] = l3 * c23 + l2 * c2
    J[0, 2] = l3 * c23

    J[1, 0] = l3 * c1 * c23 + l2 * c1 * c2 - (l1 + l4) * sign * s1
    J[1, 1] = -l3 * s1 * s23 - l2 * s1 * s2
    J[1, 2] = -l3 * s1 * s23

    J[2, 0] = l3 * s1 * c23 + l2 * c2 * s1 + (l1 + l4) * sign * c1
    J[2, 1] = l3 * c1 * s23 + l2 * c1 * s2
    J[2, 2] = l3 * c1 * s23

    # Foot position
    p = np.zeros(3, dtype=np.float64)
    p[0] = l3 * s23 + l2 * s2
    p[1] = (l1 + l4) * sign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
    p[2] = (l1 + l4) * sign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

    return p, J


class IKinematics(Protocol):
    """
    Interface for kinematics module.
    """
    def update_leg_data(self, q: np.ndarray, qd: np.ndarray) -> List[LegData]:
        """
        Given full q(12), qd(12), fill LegData for 4 legs with:
          - q, qd (3)
          - p, v (3)
          - J (3,3)
        """
        return []


class AnalyticKinematics(IKinematics):
    """
    FK/J, IK for a quadruped with 3DOF legs.
    """
    def __init__(self, geom: QuadrupedGeom):
        self._geom = geom

    @override
    def update_leg_data(self, q: np.ndarray, qd: np.ndarray) -> List[LegData]:
        q = np.asarray(q, dtype=np.float64).reshape(-1,)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1,)

        _check_shape(q, (12,), "q")
        _check_shape(qd, (12,), "qd")

        leg_datas: List[LegData] = [LegData() for _ in range(4)]

        for leg in range(4):
            sl = slice(leg * 3, leg * 3 + 3)
            q_leg = q[sl]
            qd_leg = qd[sl]

            p, J = foot_position_and_jacobian(q_leg, self._geom.leg_geoms[leg])
            v = J @ qd_leg

            ld = leg_datas[leg]
            ld.q[:] = q_leg
            ld.qd[:] = qd_leg
            ld.p[:] = p
            ld.v[:] = v
            ld.J[:, :] = J

        return leg_datas
    
    def ik_leg(
        self,
        *,
        leg_index: int,
        p_des: Vec3,
        q_seed: Optional[Vec3] = None,
    ) -> Optional[Vec3]:
        if leg_index < 0 or leg_index > 3:
            raise ValueError("leg_index must be 0..3")
        
        p = _as_v3(p_des)
        geom = self._geom.leg_geoms[leg_index]
        l1, l2, l3, l4 = float(geom.l1), float(geom.l2), float(geom.l3), float(geom.l4)
        sign = float(geom.side_sign)
        x, y, z = float(p[0]), float(p[1]), float(p[2])

        # 1: solve q1 (abad) from lateral plane geometry
        R = (l1 + l4) * sign
        r_yz = np.hypot(y, z)

        if r_yz < abs(R) - 1e-6: # unreachable
            return None  

        A_mag_sq = max(0.0, r_yz * r_yz - R * R)
        A_mag = float(np.sqrt(A_mag_sq))
        A_candidates = [A_mag, -A_mag]

        solutions: List[np.ndarray] = []

        # "A" plays role of forward projection
        for A in A_candidates:
            # q1 from angle addition
            phi = float(np.arctan2(z, y))
            q1 = phi + float(np.arctan2(A, R))

            px = x
            pz = A
            d = float(np.hypot(px, pz))

            if d > (l2 + l3) + 1e-6 or d < abs(l2 - l3) - 1e-6:
                continue

            cos_q3 = (d * d - l2 * l2 - l3 * l3) / (2.0 * l2 * l3)
            cos_q3 = float(np.clip(cos_q3, -1.0, 1.0))
            q3a = float(np.arccos(cos_q3))
            q3b = -q3a
            for q3 in (q3a, q3b):
                k1 = l2 + l3 * np.cos(q3)
                k2 = l3 * np.sin(q3)
                q2 = float(np.arctan2(px, pz) - np.arctan2(k2, k1))

            q_leg = np.array([q1, q2, q3], dtype=np.float64)
            if _is_finite(q_leg):
                solutions.append(q_leg)

        if not solutions:
            return None

        # 3: choose solution closest to q_seed to avoid branch jumping
        if q_seed is None:
            return solutions[0]

        q_seed = _as_v3(q_seed)
        best = None
        best_cost = float("inf")

        for s in solutions:
            # wrap angle differences to [-pi, pi] for fair comparison
            dq = s - q_seed
            dq = (dq + np.pi) % (2.0 * np.pi) - np.pi
            cost = float(dq @ dq)
            if cost < best_cost:
                best_cost = cost
                best = s

        return best

def make_default_geom(
    *,
    l1: float,
    l2: float,
    l3: float,
    l4: float = 0.0,
) -> QuadrupedGeom:
    fl = LegGeom(l1=l1, l2=l2, l3=l3, l4=l4, side_sign=+1.0)
    fr = LegGeom(l1=l1, l2=l2, l3=l3, l4=l4, side_sign=-1.0)
    rl = LegGeom(l1=l1, l2=l2, l3=l3, l4=l4, side_sign=+1.0)
    rr = LegGeom(l1=l1, l2=l2, l3=l3, l4=l4, side_sign=-1.0)
    return QuadrupedGeom(leg_geoms=(fl, fr, rl, rr))
