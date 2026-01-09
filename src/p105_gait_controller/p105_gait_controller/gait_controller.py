from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ---- your modules ----
from .src.robot_state import RobotState
from .src.state_estimator import DummyEstimator, OdomEstimator
from .src.gait_scheduler import GaitScheduler, LegPhase
from .src.foot_trajectory import FootTrajectoryGenerator
from .src.leg_interface import LegInterface
from .src.leg_command import LegCommand
from .src.params import GaitControllerParams, LocomotionParams, ModelParams, TopicsParams, GaitSchedulerParams, FootTrajectoryParams, LegInterfaceParams


type Twist3 = Tuple[float, float, float]  # (vx, vy, wz)


class GaitControllerNode(Node):
    """
    dataflow:
    1. JointState -> RobotState (q, qd)
    2. Kinematics/LegInterface (per-leg p,v,J ; IK)
    3. GaitScheduler -> phases -> FootTrajectory (p_des,v_des, Raibert touchdown)
    4. LegCommand (control-mode = CARTESIAN_PD or JOINT_POS)
    5. LegInterface -> q_des (12,) -> ros2_control position cmd
    """

    def __init__(self):
        super().__init__("gait_controller")
        self._declare_parameters()
        self.loco_params: LocomotionParams = self._load_params()

        self.robot_state = RobotState()

        self.estimator = DummyEstimator(nominal_height=self.loco_params.model_params.nominal_height)
        self.get_logger().info("Estimator: DummyEstimator")

        self.gait = GaitScheduler(params=self.loco_params.gait_scheduler_params)
        self.foot_traj = FootTrajectoryGenerator(params=self.loco_params.foot_trajectory_params)
        self.leg_if = LegInterface(
            params=self.loco_params.leg_interface_params,
            model_params=self.loco_params.model_params
        )

        # command velocity
        self._cmd_vel: Twist3 = (0.0, 0.0, 0.0)

        t = self.loco_params.topics_params
        self.sub_js = self.create_subscription(
            JointState, t.joint_states, self._on_joint_state, 50
        )
        self.sub_cmd = self.create_subscription(
            Twist, t.cmd_vel, self._on_cmd_vel, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, t.odom, self._on_odom, 20
        )
        self.pub_cmd = self.create_publisher(Float64MultiArray, t.command_out, 10)

        # important variables
        self.control_dt = self.loco_params.gait_controller_params.control_dt
        self.timeout = self.loco_params.leg_interface_params.state_timeout_sec

        self._last_loop_wall = time.time()
        self._timer = self.create_timer(self.control_dt, self._on_control_tick)

        self.get_logger().info(
            f"Running control loop at {1.0 / self.control_dt:.1f} Hz, "
            f"gait freq {self.loco_params.gait_scheduler_params.cycle_frequency_hz:.2f} Hz, "
            f"duty {self.loco_params.gait_scheduler_params.duty_factor:.2f}"
        )

    def _declare_parameters(self) -> None:
        # topics
        self.declare_parameter("topics.joint_states", "/joint_states")
        self.declare_parameter("topics.cmd_vel", "/cmd_vel")
        self.declare_parameter("topics.odom", "/odom")
        self.declare_parameter("topics.command_out", "/forward_command_controller/commands")

        # model
        self.declare_parameter("model.nominal_height", 0.15)
        self.declare_parameter("model.l1", 0.083)
        self.declare_parameter("model.l2", 0.213)
        self.declare_parameter("model.l3", 0.213)
        self.declare_parameter("model.l4", 0.0)

        # gait_controller
        self.declare_parameter("gait_controller.control_dt", 0.005)
        self.declare_parameter("gait_controller.cart_kp", 100.0)
        self.declare_parameter("gait_controller.cart_kd", 5.0)

        # gait_scheduler
        self.declare_parameter("gait_scheduler.frequency_hz", 2.0)
        self.declare_parameter("gait_scheduler.duty_factor", 0.5)
        self.declare_parameter("gait_scheduler.phase_offsets", (0.0, 0.5, 0.5, 0.0))

        # foot_trajectory
        self.declare_parameter("foot_trajectory.swing_height", 0.05)
        self.declare_parameter("foot_trajectory.control_point_fraction", 0.5)
        self.declare_parameter("foot_trajectory.stance_hold", True)

        self.declare_parameter("foot_trajectory.placement_time_fraction", 0.5)
        self.declare_parameter("foot_trajectory.vel_gain", 1.0)
        self.declare_parameter("foot_trajectory.max_step_xy", 0.08)

        # leg_interface
        self.declare_parameter("leg_interface.ik_lambda", 1e-3)
        self.declare_parameter("leg_interface.ik_step_scale", 1.0)

        self.declare_parameter("leg_interface.max_dq_per_step", 0.15)
        self.declare_parameter("leg_interface.joint_pos_min", (-2.5, -2.5, -2.8))
        self.declare_parameter("leg_interface.joint_pos_max", (2.5, 2.5, 0.2))

        self.declare_parameter("leg_interface.state_timeout_sec", 0.15)
        self.declare_parameter("leg_interface.default_hold_kp", 0.0)
        self.declare_parameter("leg_interface.default_hold_kd", 0.0)
        
        self.declare_parameter("leg_interface.nominal_stance_fl", (0.15, 0.12))
        self.declare_parameter("leg_interface.nominal_stance_fr", (0.15, -0.12))
        self.declare_parameter("leg_interface.nominal_stance_rl", (-0.15, 0.12))
        self.declare_parameter("leg_interface.nominal_stance_rr", (-0.15, -0.12))



    def _load_params(self) -> LocomotionParams:
        topics_params = TopicsParams(
            joint_states=str(self.get_parameter("topics.joint_states").get_parameter_value().string_value),
            cmd_vel=str(self.get_parameter("topics.cmd_vel").get_parameter_value().string_value),
            odom=str(self.get_parameter("topics.odom").get_parameter_value().string_value),
            command_out=str(self.get_parameter("topics.command_out").get_parameter_value().string_value),
        )

        model_params = ModelParams(
            nominal_height=float(self.get_parameter("model.nominal_height").get_parameter_value().double_value),
            l1=float(self.get_parameter("model.l1").get_parameter_value().double_value),
            l2=float(self.get_parameter("model.l2").get_parameter_value().double_value),
            l3=float(self.get_parameter("model.l3").get_parameter_value().double_value),
            l4=float(self.get_parameter("model.l4").get_parameter_value().double_value),
        )

        gait_controller_params = GaitControllerParams(
            control_dt=float(self.get_parameter("gait_controller.control_dt").get_parameter_value().double_value),
            cart_kp=float(self.get_parameter("gait_controller.cart_kp").get_parameter_value().double_value),
            cart_kd=float(self.get_parameter("gait_controller.cart_kd").get_parameter_value().double_value),
        )

        phase_offsets = tuple(float(x) for x in self.get_parameter("gait_scheduler.phase_offsets").get_parameter_value().double_array_value)
        if len(phase_offsets) != 4:
            raise ValueError("gait_scheduler.phase_offsets must have 4 elements (FL, FR, RL, RR)")

        gait_scheduler_params = GaitSchedulerParams(
            cycle_frequency_hz=float(self.get_parameter("gait_scheduler.frequency_hz").get_parameter_value().double_value),
            duty_factor=float(self.get_parameter("gait_scheduler.duty_factor").get_parameter_value().double_value),
            phase_offsets=(phase_offsets[0], phase_offsets[1], phase_offsets[2], phase_offsets[3]),
        )

        foot_trajectory_params = FootTrajectoryParams(
            swing_height=float(self.get_parameter("foot_trajectory.swing_height").get_parameter_value().double_value),
            control_point_fraction=float(self.get_parameter("foot_trajectory.control_point_fraction").get_parameter_value().double_value),
            stance_hold=bool(self.get_parameter("foot_trajectory.stance_hold").get_parameter_value().bool_value),
            placement_time_fraction=float(self.get_parameter("foot_trajectory.placement_time_fraction").get_parameter_value().double_value),
            vel_gain=float(self.get_parameter("foot_trajectory.vel_gain").get_parameter_value().double_value),
            max_step_xy=float(self.get_parameter("foot_trajectory.max_step_xy").get_parameter_value().double_value),
        )

        jmin = tuple(float(x) for x in self.get_parameter("leg_interface.joint_pos_min").get_parameter_value().double_array_value)
        jmax = tuple(float(x) for x in self.get_parameter("leg_interface.joint_pos_max").get_parameter_value().double_array_value)
        if len(jmin) != 3 or len(jmax) != 3:
            raise ValueError("leg_interface.joint_pos_min/max must have 3 elements (hip, thigh, calf)")

        stance_fl = tuple(float(x) for x in self.get_parameter("leg_interface.nominal_stance_fl").get_parameter_value().double_array_value)
        stance_fr = tuple(float(x) for x in self.get_parameter("leg_interface.nominal_stance_fr").get_parameter_value().double_array_value)
        stance_rl = tuple(float(x) for x in self.get_parameter("leg_interface.nominal_stance_rl").get_parameter_value().double_array_value)
        stance_rr = tuple(float(x) for x in self.get_parameter("leg_interface.nominal_stance_rr").get_parameter_value().double_array_value)
        if len(stance_fl) != 2 or len(stance_fr) != 2 or len(stance_rl) != 2 or len(stance_rr) != 2:
            raise ValueError("leg_interface.nominal_stance_* must have 2 elements [x, y]")

        leg_interface_params = LegInterfaceParams(
            ik_lambda=float(self.get_parameter("leg_interface.ik_lambda").get_parameter_value().double_value),
            ik_step_scale=float(self.get_parameter("leg_interface.ik_step_scale").get_parameter_value().double_value),
            max_dq_per_step=float(self.get_parameter("leg_interface.max_dq_per_step").get_parameter_value().double_value),
            joint_pos_min=(jmin[0], jmin[1], jmin[2]),
            joint_pos_max=(jmax[0], jmax[1], jmax[2]),
            state_timeout_sec=float(self.get_parameter("leg_interface.state_timeout_sec").get_parameter_value().double_value),
            default_hold_kp=float(self.get_parameter("leg_interface.default_hold_kp").get_parameter_value().double_value),
            default_hold_kd=float(self.get_parameter("leg_interface.default_hold_kd").get_parameter_value().double_value),
            nominal_stance_fl=(stance_fl[0], stance_fl[1]),
            nominal_stance_fr=(stance_fr[0], stance_fr[1]),
            nominal_stance_rl=(stance_rl[0], stance_rl[1]),
            nominal_stance_rr=(stance_rr[0], stance_rr[1]),
        )

        return LocomotionParams(
            topics_params=topics_params,
            model_params=model_params,
            gait_controller_params=gait_controller_params,
            gait_scheduler_params=gait_scheduler_params,
            foot_trajectory_params=foot_trajectory_params,
            leg_interface_params=leg_interface_params,
        )

    #
    # Callbacks: inputs
    #
    def _on_joint_state(self, msg: JointState) -> None:
        now = time.time()
        # msg.header.stamp may be 0 in simulation sometimes; use wall if missing
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = now

        ok = self.robot_state.update_from_joint_state(
            names=list(msg.name),
            positions=list(msg.position),
            velocities=list(msg.velocity) if msg.velocity else None,
            stamp_sec=stamp,
            wall_time_sec=now,
        )
        if not ok:
            # Only warn occasionally to avoid spam
            self.get_logger().warn("JointState missing required joints (check names/order).", throttle_duration_sec=2.0)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
        self.estimator.set_cmd_vel(self._cmd_vel)

    def _on_odom(self, msg: Odometry) -> None:
        # feed odom twist to estimator (if estimator ignores, it's fine)
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        wz = float(msg.twist.twist.angular.z)

        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        self.estimator.push_odom_twist((vx, vy, wz), stamp_sec=stamp)

    #
    # Control loop
    #
    def _on_control_tick(self) -> None:
        wall = time.time()
        dt = max(1e-4, wall - self._last_loop_wall)
        self._last_loop_wall = wall

        snap = self.robot_state.snapshot()
        if snap is None:
            return

        # Optional safety: if joint_states stale, don't command
        if self.robot_state.age_sec(wall) > self.timeout:
            self.get_logger().warn(f"JointState stale (>{self.timeout:.1f}s), skipping command.", throttle_duration_sec=1.0)
            return

        # 1: compute per-leg data from current joints (FK/J/v)
        leg_datas = self.leg_if.compute_leg_data_from_joints(snap.q, snap.qd)

        # Check if we should be in standing mode (no velocity command)
        vx, vy, wz = self._cmd_vel
        cmd_vel_magnitude = np.sqrt(vx**2 + vy**2 + wz**2)

        stance_targets = self.leg_if.get_nominal_stance_targets()

        is_standing = cmd_vel_magnitude < 0.01  # threshold for "zero" velocity
        if is_standing:
            # Standing mode: converge to nominal stance without gait scheduling
            phases: List[LegPhase] = self.gait.standing_phases()
            
            # Set all legs to nominal stance with zero velocity
            self.leg_if.zero_command()
            for leg in range(4):
                self.leg_if.set_cartesian_target(
                    leg=leg,
                    p_des=stance_targets[leg],
                    v_des=np.zeros(3),
                    kp=self.loco_params.gait_controller_params.cart_kp,
                    kd=self.loco_params.gait_controller_params.cart_kd,
                )
            
            # Convert to joint targets (IK) and publish
            self.leg_if.enable()
            q_des_12 = self.leg_if.update(leg_datas=leg_datas, phases=phases)
        else:
            # 2: gait phases (stance/swing schedule)
            phases: List[LegPhase] = self.gait.step(dt=dt)

            # 3: estimator (contacts from phases, base_twist)
            self.estimator.set_phases(tuple(phases))  # type: ignore
            self.estimator.update(robot_state=snap, dt=dt)  # type: ignore
            est = self.estimator.get_estimated_state()
            com_vel = est.base_twist
            cmd_vel = self.estimator._cmd_vel  # should be same as self._cmd_vel

            # 4: foot trajectory -> desired foot pos/vel
            p_des_list, v_des_list = self.foot_traj.update(
                phases=phases,
                leg_datas=leg_datas,
                frequency_hz=self.loco_params.gait_scheduler_params.cycle_frequency_hz,
                duty_factor=self.loco_params.gait_scheduler_params.duty_factor,
                cmd_vel=cmd_vel,
                com_vel=com_vel,
                touchdown_targets=None,
                stance_targets=stance_targets,
            )

            # 5: Fill leg commands with Cartesian targets
            self.leg_if.zero_command()
            for leg in range(4):
                self.leg_if.set_cartesian_target(
                    leg=leg,
                    p_des=p_des_list[leg],
                    v_des=v_des_list[leg],
                    kp=self.loco_params.gait_controller_params.cart_kp,
                    kd=self.loco_params.gait_controller_params.cart_kd,
                )

            # 6: convert to joint targets (IK) for position interface
            self.leg_if.enable()
            q_des_12 = self.leg_if.update(leg_datas=leg_datas, phases=phases)
        
        # 7: publish to ros2_control forward_position_controller
        out = Float64MultiArray()
        out.data = [float(x) for x in q_des_12.tolist()]
        self.pub_cmd.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GaitControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # send zeros (optional)
        msg = Float64MultiArray()
        msg.data = [0.0] * 12
        node.pub_cmd.publish(msg)
        node.destroy_node()
        rclpy.shutdown()
