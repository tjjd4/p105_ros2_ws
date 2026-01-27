import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

# ROS2 相關套件
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# === 設定區 ===
MODEL_PATH = "assets/mjmodel.xml"

# 定義馬達名稱與順序 (必須與您的 XML <actuator> 順序完全一致)
# 這裡對應 XML 中的 actuator
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        
        # 1. 初始化 MuJoCo
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        
        # 重置並載入 Keyframe (站立姿勢)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # 2. 建立 ROS2 介面
        # 訂閱: 接收 12 個馬達的目標角度
        # 格式: Float64MultiArray (簡單陣列)
        self.sub_cmd = self.create_subscription(
            Float64MultiArray,
            '/forward_command_controller/commands',
            self.cmd_callback,
            10
        )
        
        # 發布: 回傳現在的關節狀態 (給 Rviz 或 State Estimator 用)
        self.pub_state = self.create_publisher(JointState, '/joint_states', 10)

        # 緩衝區：用來存最新的控制指令 (預設為當前站立姿勢)
        # 注意：我們直接讀取 keyframe 載入後的 ctrl 作為預設值，避免一啟動就歸零跌倒
        self.current_ctrl = self.data.ctrl.copy()
        
        self.get_logger().info("MuJoCo Bridge 已啟動！等待 /joint_commands 指令...")

    def cmd_callback(self, msg):
        """
        當 ROS2 送來指令時觸發
        預期 msg.data 是一個長度 12 的陣列 (對應 12 個馬達)
        """
        if len(msg.data) != 12:
            self.get_logger().warn(f"接收到錯誤的指令長度: {len(msg.data)}, 應為 12")
            return
        
        # 更新控制目標 (會在主迴圈寫入 MuJoCo)
        self.current_ctrl = np.array(msg.data, dtype=np.float64)

    def publish_joint_states(self):
        """
        讀取 MuJoCo 的物理狀態，打包成 ROS 訊息發送
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        
        # 從 MuJoCo 讀取數據
        # qpos 通常前 7 個是 base (x,y,z, quat)，後面才是 joint
        # qvel 同理，前 6 個是 base (v, w)
        
        # 取得關節角度 (注意：要跳過前 7 個 freejoint 變數)
        # 寫法依賴於您的 XML 結構，如果只有一個 freejoint，通常 joint 從 index 7 開始
        joint_qpos = self.data.qpos[7:] 
        joint_qvel = self.data.qvel[6:] 

        msg.position = joint_qpos.tolist()
        msg.velocity = joint_qvel.tolist()
        
        # 發布
        self.pub_state.publish(msg)

    def run_simulation(self):
        """
        主迴圈：整合 MuJoCo Viewer 與 ROS2 Spin
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # 讓 viewer 稍微初始化
            time.sleep(0.1)
            
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()

                # A. 處理 ROS2 事件 (接收指令)
                rclpy.spin_once(self, timeout_sec=0.0)

                # B. 寫入控制指令到 MuJoCo
                self.data.ctrl[:] = self.current_ctrl

                # C. 執行物理模擬 (Step)
                mujoco.mj_step(self.model, self.data)

                # D. 發布狀態回 ROS2
                self.publish_joint_states()

                # E. 更新畫面
                viewer.sync()

                # F. 控速 (與 timestep 同步)
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