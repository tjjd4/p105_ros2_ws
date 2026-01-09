from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple
import numpy as np
from .params import GaitSchedulerParams


@dataclass
class LegPhase:
    """
    Per-leg phase output.
    """
    phase: float                 # global phase with offset (0..1)
    is_swing: bool
    is_stance: bool
    swing_phase: float           # (0..1) valid only if is_swing else 0
    stance_phase: float          # (0..1) valid only if is_stance else 0

    # Helpful events (edges)
    just_touchdown: bool = False
    just_liftoff: bool = False


class GaitScheduler:
    """
    Timing-only gait scheduler.

    - Maintains a global phase (0..1)
    - Applies per-leg phase offsets
    - Splits stance/swing according to duty_factor
    """
    def __init__(self, params: GaitSchedulerParams = GaitSchedulerParams()):
        self.params = params
        self._global_phase: float = 0.0  # (0..1)
        self._last_is_swing: List[bool] = [False, False, False, False]

    def reset(self, phase: float = 0.0) -> None:
        self._global_phase = float(phase) % 1.0
        self._last_is_swing = [False, False, False, False]

    def step(self, dt: float) -> List[LegPhase]:
        """
        Advance the gait phase by dt seconds and return per-leg phases.

        This is called by the control loop at fixed rate.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")

        # advance global phase
        self._global_phase = (self._global_phase + dt * self.params.cycle_frequency_hz) % 1.0

        # stance window: [0, duty_factor)
        df = self.params.duty_factor

        outputs: List[LegPhase] = []
        for i in range(4):
            ph = (self._global_phase + self.params.phase_offsets[i]) % 1.0

            is_stance = (ph < df)
            is_swing = not is_stance

            if is_stance:
                stance_phase = ph / df  # normalized 0..1
                swing_phase = 0.0
            else:
                # swing part is [df, 1)
                swing_phase = (ph - df) / (1.0 - df)
                stance_phase = 0.0

            # edge events
            last_swing = self._last_is_swing[i]
            just_liftoff = (not last_swing) and is_swing
            just_touchdown = last_swing and (not is_swing)

            outputs.append(LegPhase(
                phase=ph,
                is_swing=is_swing,
                is_stance=is_stance,
                swing_phase=float(swing_phase),
                stance_phase=float(stance_phase),
                just_touchdown=just_touchdown,
                just_liftoff=just_liftoff,
            ))

            self._last_is_swing[i] = is_swing

        return outputs
    
    def standing_phases(self) -> List[LegPhase]:
        """
        Return per-leg phases for standing (all stance).
        """
        outputs: List[LegPhase] = []
        for i in range(4):
            ph = (self._global_phase + self.params.phase_offsets[i]) % 1.0
            outputs.append(LegPhase(
                phase=ph,
                is_swing=False,
                is_stance=True,
                swing_phase=0.0,
                stance_phase=1.0,
                just_touchdown=False,
                just_liftoff=False,
            ))
        return outputs

    @property
    def global_phase(self) -> float:
        return self._global_phase

    def set_frequency(self, frequency_hz: float) -> None:
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be > 0")
        self.params = replace(self.params, frequency_hz=float(frequency_hz))

    def set_duty_factor(self, duty_factor: float) -> None:
        if not (0.0 < duty_factor < 1.0):
            raise ValueError("duty_factor must be in (0,1)")
        self.params = replace(self.params, duty_factor=float(duty_factor))

    def set_phase_offsets(self, offsets: Tuple[float, float, float, float]) -> None:
        if len(offsets) != 4:
            raise ValueError("offsets must have length 4")
        for o in offsets:
            if not (0.0 <= o < 1.0):
                raise ValueError("phase_offsets must be in [0,1)")
        self.params = replace(self.params, phase_offsets=tuple(float(x) for x in offsets))
