# P105 ROS2 Workspace

ROS2 workspace for quadruped robot, including simulation, control, and gait planning.

## Requirements

- ROS2 (Jazzy in this repo)
- Python 3.10+
- MuJoCo (for physics simulation)

## Build

```bash
cd ~/p105_ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Packages

### p105_description

Robot URDF/Xacro model description package, including:

- Robot model files (URDF/Xacro)
- 3D Mesh files
- RViz configuration

**Launch model visualization:**

```bash
ros2 launch p105_description display_model.launch.py
```

---

### p105_control

ros2_control configuration package, including:

- Controller configuration
- Hardware Interface settings

**Launch control test:**

```bash
ros2 launch p105_control test_control.launch.py
```

---

### p105_gait_controller

Gait controller package for quadruped locomotion and motion control, including:

- Gait Scheduler
- Foot Trajectory planner
- Inverse Kinematics Leg Interface
- Raibert foothold planning

**Launch gait controller:**

```bash
ros2 launch p105_gait_controller gait_controller.launch.py
```

**Main Topics:**

| Topic | Type | Description |
| ----- | ---- | ----------- |
| `/joint_states` | `sensor_msgs/JointState` | Joint state input |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command input |
| `/odom` | `nav_msgs/Odometry` | Odometry input |
| `/forward_command_controller/commands` | `std_msgs/Float64MultiArray` | Motor command output |

---

### p105_mujoco_gym

MuJoCo physics simulation bridge package, providing:

- MuJoCo simulator to ROS2 connection
- Joint state publishing
- Control command receiving

**Run the simulator:**

```bash
cd ~/p105_ros2_ws/src/p105_mujoco_gym
source .venv/bin/activate
python3 p105_mujoco_gym/mujoco_bridge.py
```

---

## Quick Start

1. **Model visualization only (RViz):**

   ```bash
   ros2 launch p105_description display_model.launch.py
   ```

2. **Run MuJoCo simulation + Gait control:**

   ```bash
   # Terminal 1: Start MuJoCo simulator
   cd ~/p105_ros2_ws/src/p105_mujoco_gym
   source .venv/bin/activate
   python3 p105_mujoco_gym/mujoco_bridge.py

   # Terminal 2: Start gait controller
   ros2 launch p105_gait_controller gait_controller.launch.py

   # Terminal 3: Send velocity command
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
   ```
