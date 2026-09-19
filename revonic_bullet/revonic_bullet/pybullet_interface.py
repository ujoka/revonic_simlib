import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
import math
import pybullet as p
import pybullet_data


class PyBulletRosWrapper(Node):
    def __init__(self):
        super().__init__("pybullet_interface")

        # ----------------- Parameters -----------------
        self.declare_parameter("rover_model", "eva2")
        self.declare_parameter("loop_rate", 100.0)
        self.declare_parameter("urdf_spawn_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("terrain_path", "")
        self.declare_parameter("tracking_cam", False)
        self.declare_parameter("cam_angle", [4.5, 45.0, -45.0])
        self.declare_parameter("max_torque", 16.0)
        self.declare_parameter(
            "joint_control_mode", 1
        )  # 0: position, 1: velocity, 2: torque

        # Load parameters
        self._rover_model = self.get_parameter("rover_model").value
        self.loop_rate = self.get_parameter("loop_rate").value
        self.urdf_spawn_pose = self.get_parameter("urdf_spawn_pose").value
        self.terrain_path = self.get_parameter("terrain_path").value or None
        self.tracking_cam = self.get_parameter("tracking_cam").value
        self.cam_angle = self.get_parameter("cam_angle").value
        self.max_torque = self.get_parameter("max_torque").value
        self.joint_control_mode = self.get_parameter("joint_control_mode").value
        self.joint_control_mode_mapping = {
            0: p.POSITION_CONTROL,
            1: p.VELOCITY_CONTROL,
            2: p.TORQUE_CONTROL,
        }

        # ----------------- PyBullet Setup -----------------
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setTimeStep(1.0 / self.loop_rate)
        p.setGravity(0, 0, -9.81)

        self.robot_id = None
        self.num_dofs = 0
        self.pause_simulation = False

        self.set_urdf_path()
        self.load_terrain()
        self.load_urdf()

        # ----------------- Command Placeholders -----------------
        self.command_positions = [0.0] * self.num_dofs
        self.command_velocities = [0.0] * self.num_dofs
        self.command_efforts = [
            self.max_torque if self.joint_control_mode != 2 else 0.0
        ] * self.num_dofs

        # ----------------- Publishers & Subscribers -----------------
        self.joint_state_pub = self.create_publisher(
            JointState, "/revonic/joint_states", 100
        )
        self.joint_state = JointState()
        self.joint_state.name = list(self.revolute_joint_idx.values())

        self.joint_command_sub = self.create_subscription(
            JointState, "/revonic/joint_commands", self.joint_command_callback, 10
        )

        # ----------------- Services -----------------
        self.create_service(Empty, "reset_simulation", self.handle_reset_simulation)
        self.create_service(Empty, "pause_physics", self.handle_pause_physics)
        self.create_service(Empty, "unpause_physics", self.handle_unpause_physics)

        # ----------------- Timer for Control Loop -----------------
        self.timer = self.create_timer(1.0 / self.loop_rate, self.control_loop)

    # ----------------- Main Control Loop -----------------
    def control_loop(self):
        self.send_joint_commands()

        # Update joint states
        time_now = self.get_clock().now().to_msg()
        self.joint_state.header.stamp = time_now
        (
            self.joint_state.position,
            self.joint_state.velocity,
            self.joint_state.effort,
        ) = self.get_feedback()
        self.joint_state_pub.publish(self.joint_state)

        # Update camera if enabled
        if self.tracking_cam:
            self.update_cam_view(self.cam_angle)

        # Step simulation
        if not self.pause_simulation:
            p.stepSimulation()

    # ----------------- Subscriber Callback -----------------
    def joint_command_callback(self, msg: JointState):
        if len(msg.position) != self.num_dofs:
            self.get_logger().warn("Received joint command with wrong DOF count, ignoring")
            return
        # TODO: Simulation breaks when receiving NAN values.
        #       should positions also be set to zero?
        positions = list(msg.position)
        velocities = (
            [0.0 if math.isnan(v) else v for v in msg.velocity]
            if len(msg.velocity) == self.num_dofs
            else [0.0] * self.num_dofs
        )
        efforts = (
            [0.0 if math.isnan(v) else v for v in msg.effort]
            if len(msg.effort) == self.num_dofs
            else [self.max_torque] * self.num_dofs
        )

        # Update commands based on control mode
        if self.joint_control_mode == 0:  # POSITION_CONTROL
            self.command_positions = positions
            self.command_velocities = velocities
            self.command_efforts = efforts
        elif self.joint_control_mode == 1:  # VELOCITY_CONTROL
            self.command_positions = [0.0] * self.num_dofs
            self.command_velocities = velocities
            self.command_efforts = efforts
        elif self.joint_control_mode == 2:  # TORQUE_CONTROL
            self.command_positions = [0.0] * self.num_dofs
            self.command_velocities = [0.0] * self.num_dofs
            self.command_efforts = efforts

    # ----------------- Send Commands to PyBullet -----------------
    def send_joint_commands(self):
        if self.robot_id is None or self.num_dofs == 0:
            return
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_idx,
            self.joint_control_mode_mapping[self.joint_control_mode],
            targetPositions=self.command_positions,
            targetVelocities=self.command_velocities,
            forces=self.command_efforts,
        )

    # ----------------- PyBullet Helpers -----------------
    def get_feedback(self):
        positions = []
        velocities = []
        efforts = []
        for idx in self.joint_idx:
            joint_p, joint_v, _, joint_torque = p.getJointState(self.robot_id, idx)
            positions.append(joint_p)
            velocities.append(joint_v)
            efforts.append(joint_torque)
        return positions, velocities, efforts

    def set_urdf_path(self):
        base_path = get_package_share_directory("revonic_description")
        if self._rover_model == "eva2":
            self.urdf_path = f"{base_path}/eva_models/revonic_eva2/revonic_eva2.urdf"
        else:
            self.urdf_path = None

    def load_urdf(self):
        self.robot_id = p.loadURDF(
            self.urdf_path,
            self.urdf_spawn_pose,
            useFixedBase=0,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        # Map joint info
        (
            self.revolute_joint_idx,
            self.prismatic_joint_idx,
            self.fixed_joint_idx,
            self.link_name_idx,
        ) = self.get_properties()
        self.num_dofs = len(self.revolute_joint_idx)
        self.joint_idx = list(self.revolute_joint_idx.keys())

        # Apply physics constraints
        # ----------------- Contraints Class -----------------
        if self._rover_model == "eva2":
            self.rev2_constaints = RevonicEva2Constraints(self.robot_id)
            self.rev2_constaints._setup_closed_loops()

    def get_properties(self):
        revolute_joint_idx = {}
        prismatic_joint_idx = {}
        fixed_joint_idx = {}
        link_name_idx = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_type = info[2]
            name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            link_name_idx[link_name] = i
            if joint_type == p.JOINT_REVOLUTE:
                revolute_joint_idx[i] = name
            elif joint_type == p.JOINT_PRISMATIC:
                prismatic_joint_idx[i] = name
            else:
                fixed_joint_idx[i] = name
        return revolute_joint_idx, prismatic_joint_idx, fixed_joint_idx, link_name_idx

    def load_terrain(self):
        p.loadURDF("plane.urdf")
        if self.terrain_path is not None:
            try:
                p.loadURDF(self.terrain_path, useFixedBase=1)
            except p.error as e:
                self.get_logger().error(f"Failed to load terrain: {e}")

    def update_cam_view(self, cam_angle):
        dist, yaw, pitch = cam_angle
        mid_idx = int(self.num_dofs / 2) - 1
        pos, *_ = p.getLinkState(self.robot_id, self.joint_idx[mid_idx])
        pos = list(pos)
        pos[2] = 0
        p.resetDebugVisualizerCamera(dist, yaw, pitch, tuple(pos))

    # ----------------- Services -----------------
    def handle_reset_simulation(self, req, response=None):
        self.pause_simulation = True
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.load_terrain()
        self.load_urdf()
        self.command_positions = [0.0] * self.num_dofs
        self.command_velocities = [0.0] * self.num_dofs
        self.command_efforts = [
            self.max_torque if self.joint_control_mode != 2 else 0.0
        ] * self.num_dofs
        self.pause_simulation = False
        return response

    def handle_pause_physics(self, req, response=None):
        self.pause_simulation = True
        return response

    def handle_unpause_physics(self, req, response=None):
        self.pause_simulation = False
        return response


class RevonicEva2Constraints:
    def __init__(self, robot_id):
        self._robot_id = robot_id

    def _setup_closed_loops(self):
        """Internal helper to apply PyBullet constraints unique to differential ball-joints."""
        left_bottom_cid = p.createConstraint(
            parentBodyUniqueId=self._robot_id,
            parentLinkIndex=17, # "diff_left_ball_joint"
            childBodyUniqueId=self._robot_id,
            childLinkIndex=1, # "left_suspension_joint"
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[-0.3060, 0, 0.06],
        )
        right_bottom_cid = p.createConstraint(
            parentBodyUniqueId=self._robot_id,
            parentLinkIndex=23, # "diff_right_ball_joint"
            childBodyUniqueId=self._robot_id,
            childLinkIndex=6, # "right_suspension_joint"
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[-0.3060, 0, 0.06],
        )
        
        for cid in (left_bottom_cid, right_bottom_cid):
            p.changeConstraint(cid, maxForce=1000.0, erp=0.5)


def main():
    rclpy.init()
    node = PyBulletRosWrapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()