import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

MODEL_PATH = "assets/p105.xml"

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.sub_cmd = self.create_subscription(
            Float64MultiArray,
            '/forward_command_controller/commands',
            self.cmd_callback,
            10
        )
        
        self.pub_state = self.create_publisher(JointState, '/joint_states', 10)

        self.current_ctrl = self.data.ctrl.copy()
        
        self.get_logger().info("MuJoCo Bridge activated, waiting for /joint_commands...")

    def cmd_callback(self, msg):
        """
        When ROS2 sends a command, this function is triggered
        Expected msg.data is an array of length 12 (corresponding to 12 motors)
        """
        if len(msg.data) != 12:
            self.get_logger().warn(f"Received incorrect command length: {len(msg.data)}, should be 12")
            return
        
        # Update control target (will be written to MuJoCo in the main loop)
        self.current_ctrl = np.array(msg.data, dtype=np.float64)

    def publish_joint_states(self):
        """
        Read MuJoCo's physical state and package it into a ROS message to send
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        
        # Read data from MuJoCo
        # Get joint angles (note: skip the first 7 freejoint variables)
        joint_qpos = self.data.qpos[7:] 
        joint_qvel = self.data.qvel[6:] 

        msg.position = joint_qpos.tolist()
        msg.velocity = joint_qvel.tolist()
        
        self.pub_state.publish(msg)

    def run_simulation(self):
        """
        Main loop: integrate MuJoCo Viewer with ROS2 Spin
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            time.sleep(0.1)
            
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()

                # A. Process ROS2 events (receive commands)
                rclpy.spin_once(self, timeout_sec=0.0)

                # B. Write control commands to MuJoCo
                self.data.ctrl[:] = self.current_ctrl

                # C. Execute physical simulation (Step)
                mujoco.mj_step(self.model, self.data)

                # D. Publish state back to ROS2
                self.publish_joint_states()

                # E. Update viewer
                viewer.sync()

                # F. Control speed (synchronize with timestep)
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main(args=None):
    rclpy.init(args=args)
    bridge = MujocoRosBridge()
    
    try:
        bridge.run_simulation()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()