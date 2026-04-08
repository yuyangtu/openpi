import logging
import time
import dataclasses
import threading
import queue
from typing import Tuple

import numpy as np
import tyro
import rospy

from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from openpi_client import websocket_client_policy as _websocket_client_policy

from tams_diana7_tools.dual_diana_helper import DualDianaHelper
from tams_diana7_tools.gripper_interfaces import GripperType

from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint

# MoveIt state validity service
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest


# -----------------------
# Args
# -----------------------
@dataclasses.dataclass
class Args:
    host: str = "134.100.39.19"
    port: int = 8000
    num_steps: int = 2000
    prompt: str = "pick up the spoon and feed the person"

    rotate_wrist_180: bool = False
    convert_to_rgb: bool = True
    resize_hw: int = 224

    # Policy horizon H and timing
    horizon: int = 16
    waypoint_dt: float = 0.20  # seconds per step

    # Execute only s steps each loop (rolling execution)
    exec_steps: int = 8       # s: 每次只执行 s 步（建议 8~15）
    manual_d: int = 8         # d: 砍掉前缀多少步（你主要调这个）

    # MoveIt smoothing (no obstacle avoidance, just time parameterization)
    enable_moveit_retime: bool = True
    vel_scale: float = 0.05
    acc_scale: float = 0.05

    # Collision checking (self-collision + planning scene)
    collision_check_stride: int = 4  # 抽查点间隔：1最严谨但慢
    collision_check_timeout: float = 0.15

    # Fallback planning (only if collision detected)
    fallback_use_plan: bool = True
    fallback_planning_time: float = 0.5

    # Async buffer (queue) configuration
    buffer_size: int = 2
    max_wait_next_chunk: float = 0.2  # 主线程最多等下一chunk多久（秒）

    # Debug
    print_timing: bool = True


# -----------------------
# Sensor cache (常驻订阅)
# -----------------------
class SensorCache:
    def __init__(self, rotate_wrist_180: bool, convert_to_rgb: bool, resize_hw: int):
        self.rotate_wrist_180 = rotate_wrist_180
        self.convert_to_rgb = convert_to_rgb
        self.resize_hw = resize_hw

        self._lock = threading.Lock()
        self._bridge = CvBridge()

        self._right_state_8 = np.zeros((8,), dtype=np.float32)
        self._top_img = None
        self._wrist_img = None
        self._got_any = False

        self._sub_js = rospy.Subscriber("/joint_states", JointState, self._joint_cb, queue_size=1)
        self._sub_top = rospy.Subscriber("/top_view/color/image_raw", Image, self._top_cb, queue_size=1)
        self._sub_wrist = rospy.Subscriber("/diana_R_view/color/image_raw", Image, self._wrist_cb, queue_size=1)

    def _joint_cb(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        arm_r_names = [n for n in msg.name if "arm_r" in n]
        gr_names = [n for n in msg.name if "q_gripper_r" in n]

        if len(arm_r_names) >= 7:
            arm_r_names = arm_r_names[:7]
            right7 = np.array([msg.position[name_to_idx[n]] for n in arm_r_names], dtype=np.float32)
        else:
            right7 = np.array(msg.position[:7], dtype=np.float32)

        gr = float(msg.position[name_to_idx[gr_names[0]]]) if len(gr_names) >= 1 else 0.0

        with self._lock:
            self._right_state_8[:7] = right7
            self._right_state_8[7] = np.float32(gr)
            self._got_any = True

    def _top_cb(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._top_img = img
            self._got_any = True

    def _wrist_cb(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._wrist_img = img
            self._got_any = True

    def _process_image(self, img, do_rotate_180: bool):
        import cv2
        if img is None:
            img = np.zeros((self.resize_hw, self.resize_hw, 3), dtype=np.uint8)
        else:
            if img.shape[0] != self.resize_hw or img.shape[1] != self.resize_hw:
                img = cv2.resize(img, (self.resize_hw, self.resize_hw))
        if do_rotate_180:
            img = cv2.flip(img, -1)
        if self.convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def wait_ready(self, timeout: float = 3.0) -> bool:
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            with self._lock:
                ok = (self._top_img is not None) and (self._wrist_img is not None) and self._got_any
            if ok:
                return True
            rospy.sleep(0.02)
        return False

    def get_observation(self, prompt: str) -> dict:
        with self._lock:
            state8 = self._right_state_8.copy()
            top = None if self._top_img is None else self._top_img.copy()
            wrist = None if self._wrist_img is None else self._wrist_img.copy()

        top_img = self._process_image(top, do_rotate_180=False)
        wrist_img = self._process_image(wrist, do_rotate_180=self.rotate_wrist_180)

        return {
            "observation/state": state8.astype(np.float32),
            "observation/image": top_img,
            "observation/wrist_image": wrist_img,
            "prompt": prompt,
        }


# -----------------------
# MoveIt helpers
# -----------------------
def _get_arms_joint_names(arms_group) -> list:
    joints = arms_group.get_active_joints()
    filtered = [jn for jn in joints if ("arm_l" in jn or "arm_r" in jn)]
    return filtered[:14] if len(filtered) >= 14 else joints[:14]


def _robot_state_from_joints(joint_names, joint_positions):
    rs = RobotState()
    rs.joint_state = JointState()
    rs.joint_state.name = list(joint_names)
    rs.joint_state.position = [float(x) for x in joint_positions]
    return rs


def build_traj_from_right_waypoints(
    robot: DualDianaHelper,
    left_fixed_7: np.ndarray,
    right_waypoints_7: np.ndarray,
    waypoint_dt: float,
    vel_scale: float,
    acc_scale: float,
    enable_moveit_retime: bool,
) -> RobotTrajectory:
    group = robot.arms_group
    joint_names = _get_arms_joint_names(group)

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(joint_names)

    # 起点：当前关节（14）
    cur = group.get_current_joint_values()
    cur14 = np.array(cur[:14], dtype=np.float64)

    p0 = JointTrajectoryPoint()
    p0.positions = [float(x) for x in cur14]
    p0.time_from_start = rospy.Duration.from_sec(float(waypoint_dt))
    traj.joint_trajectory.points.append(p0)

    for i in range(right_waypoints_7.shape[0]):
        target14 = np.concatenate([left_fixed_7, right_waypoints_7[i]], axis=0)
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in target14]
        p.velocities = []
        p.accelerations = []
        p.effort = []
        p.time_from_start = rospy.Duration.from_sec(float((i + 2) * waypoint_dt))
        traj.joint_trajectory.points.append(p)

    if enable_moveit_retime:
        try:
            traj = group.retime_trajectory(
                robot.robot.get_current_state(),
                traj,
                velocity_scaling_factor=float(vel_scale),
                acceleration_scaling_factor=float(acc_scale),
                algorithm="time_optimal_trajectory_generation",
            )
        except Exception as e:
            rospy.logwarn(f"retime_trajectory failed, execute raw traj: {e}")

    return traj


# -----------------------
# Collision checking via /check_state_validity
# -----------------------
class StateValidityChecker:
    def __init__(self, group_name: str, timeout: float = 0.15):
        self.group_name = group_name
        srv_name = "/check_state_validity"
        rospy.loginfo(f"Waiting for service {srv_name} ...")
        rospy.wait_for_service(srv_name, timeout=5.0)
        self._srv = rospy.ServiceProxy(srv_name, GetStateValidity)
        self.timeout = timeout

    def is_state_valid(self, joint_names, joint_positions) -> Tuple[bool, str]:
        req = GetStateValidityRequest()
        req.group_name = self.group_name
        req.robot_state = _robot_state_from_joints(joint_names, joint_positions)
        try:
            resp = self._srv(req)
            return bool(resp.valid), ("valid" if resp.valid else "collision_or_invalid")
        except rospy.ServiceException as e:
            return False, f"svc_error: {e}"

    def is_trajectory_valid(self, traj: RobotTrajectory, stride: int = 2) -> Tuple[bool, int, str]:
        jn = traj.joint_trajectory.joint_names
        pts = traj.joint_trajectory.points
        if not pts:
            return False, -1, "empty_traj"

        stride = max(1, int(stride))
        for idx in range(0, len(pts), stride):
            ok, msg = self.is_state_valid(jn, pts[idx].positions)
            if not ok:
                return False, idx, msg

        # 末端点也查一次
        if (len(pts) - 1) % stride != 0:
            ok, msg = self.is_state_valid(jn, pts[-1].positions)
            if not ok:
                return False, len(pts) - 1, msg

        return True, -1, "valid"


# -----------------------
# Fallback: one MoveIt plan to final target (only on collision)
# -----------------------
def fallback_plan_and_execute_to_target(
    robot: DualDianaHelper,
    left_fixed_7: np.ndarray,
    right_target_7: np.ndarray,
    planning_time: float,
    vel_scale: float,
    acc_scale: float,
) -> bool:
    group = robot.arms_group
    group.set_planning_time(float(planning_time))
    group.set_max_velocity_scaling_factor(float(vel_scale))
    group.set_max_acceleration_scaling_factor(float(acc_scale))

    joint_names = _get_arms_joint_names(group)
    target14 = np.concatenate([left_fixed_7, right_target_7], axis=0)

    group.set_start_state_to_current_state()
    group.set_joint_value_target({jn: float(p) for jn, p in zip(joint_names, target14)})

    plan_ret = group.plan()
    if isinstance(plan_ret, tuple):
        success, plan_msg, _, _ = plan_ret
        if (not success) or plan_msg is None or len(plan_msg.joint_trajectory.points) == 0:
            rospy.logwarn("Fallback MoveIt plan failed.")
            return False
        plan = plan_msg
    else:
        plan = plan_ret
        if plan is None or len(plan.joint_trajectory.points) == 0:
            rospy.logwarn("Fallback MoveIt plan failed.")
            return False

    ok = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()
    return bool(ok)


# -----------------------
# Async inference producer (background thread)
# Buffer 中存的就是 “可执行段 seg”，而不是整段 H
# -----------------------
class AsyncInferProducer(threading.Thread):
    def __init__(
        self,
        policy,
        sensor: SensorCache,
        prompt: str,
        horizon: int,
        exec_steps: int,
        manual_d: int,
        out_q: queue.Queue,
        stop_evt: threading.Event,
    ):
        super().__init__(daemon=True)
        self.policy = policy
        self.sensor = sensor
        self.prompt = prompt
        self.horizon = int(horizon)
        self.exec_steps = int(exec_steps)
        self.manual_d = int(manual_d)
        self.out_q = out_q
        self.stop_evt = stop_evt

    def run(self):
        H = self.horizon
        s = self.exec_steps
        d = self.manual_d

        while not rospy.is_shutdown() and not self.stop_evt.is_set():
            obs = self.sensor.get_observation(self.prompt)
            t0 = time.time()
            result = self.policy.infer(obs)
            t1 = time.time()

            if not isinstance(result, dict) or "actions" not in result:
                continue

            acts = np.asarray(result["actions"], dtype=np.float64)
            if acts.ndim != 2 or acts.shape[1] < 8:
                continue

            if acts.shape[0] < H:
                continue
            acts = acts[:H]  # (H, 8)

            # ---- cut prefix in producer (buffer里不存前缀) ----
            start = max(0, min(d, H - 1))
            end = min(start + s, H)
            if end <= start:
                continue

            seg = acts[start:end].copy()  # (<=s, 8)

            # 丢旧保新：避免延迟堆积
            try:
                if self.out_q.full():
                    _ = self.out_q.get_nowait()
                self.out_q.put_nowait((seg, t0, t1))
            except queue.Full:
                pass


# -----------------------
# Main
# -----------------------
def main(args: Args) -> None:
    rospy.init_node("openpi_moveit_cutbuffer", anonymous=True)

    # sanity checks
    H = int(args.horizon)
    s = int(args.exec_steps)
    d = int(args.manual_d)

    if s <= 0 or s > H:
        raise ValueError(f"exec_steps must be in [1, H]. Got s={s}, H={H}")
    if d < 0:
        raise ValueError(f"manual_d must be >=0. Got d={d}")
    if d > H - 1:
        rospy.logwarn(f"manual_d={d} too large for H={H}, clamp to H-1.")
        d = H - 1
        args.manual_d = d
    if d + 1 > H:
        raise ValueError("Invalid d/H.")
    if d + s > H:
        rospy.logwarn(f"d+s={d+s} > H={H}. Will execute shorter segment each time (end=min(d+s,H)).")

    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info(f"Server metadata: {policy.get_server_metadata()}")

    rospy.sleep(1.0)

    robot = DualDianaHelper(
        load_moveit=True,
        arm_l_active=True,
        arm_r_active=True,
        no_grippers=False,
        gripper_l=None,
        gripper_r=GripperType.Q_GRIPPER,
        fetch_joint_states=True,
        load_joint_trajectory_controller=True,
    )
    robot.load_position_trajectory_controllers()

    # init pose
    joint_goal_r = np.deg2rad([-32, 5, -18, 95, -8, -85, -1])
    joint_goal_l = np.deg2rad([-32, -23, -49, 97, 11, -98, -1])
    joint_goal_gripper_r = np.array([1.0], dtype=np.float32)  # open
    joint_goal_gripper_l = np.array([0.0], dtype=np.float32)

    joint_goal = np.concatenate([joint_goal_l, joint_goal_r, joint_goal_gripper_l, joint_goal_gripper_r])
    robot.move_to_joint_goal(joint_goal, use_moveit=True)

    left_fixed = np.array(joint_goal[:7], dtype=np.float64)

    # sensor resident
    sensor = SensorCache(args.rotate_wrist_180, args.convert_to_rgb, args.resize_hw)
    rospy.loginfo("Waiting for sensors to be ready...")
    if not sensor.wait_ready(timeout=3.0):
        rospy.logwarn("Sensors not ready, will continue with zero images until ready.")

    # warmup
    rospy.loginfo("Warming up policy inference...")
    _ = policy.infer(sensor.get_observation(args.prompt))

    # collision checker
    group = robot.arms_group
    checker = StateValidityChecker(group_name=group.get_name(), timeout=args.collision_check_timeout)

    # async producer
    q_actions: queue.Queue = queue.Queue(maxsize=args.buffer_size)
    stop_evt = threading.Event()
    producer = AsyncInferProducer(
        policy=policy,
        sensor=sensor,
        prompt=args.prompt,
        horizon=args.horizon,
        exec_steps=args.exec_steps,
        manual_d=args.manual_d,
        out_q=q_actions,
        stop_evt=stop_evt,
    )
    producer.start()

    rospy.loginfo(
        f"Running CUT-BUFFER rolling execution with H={H}, s={s}, manual_d={d} (dt={args.waypoint_dt}).\n"
        f"Queue stores only EXECUTABLE segment each time: seg = actions[d:d+s].\n"
        f"Tip: if you see rollback / pull-back, increase --manual_d.\n"
    )

    count = 0
    start_time = time.time()

    # get first seg
    rospy.loginfo("Waiting first executable segment from async producer...")
    seg, t_infer_start, t_infer_done = q_actions.get()
    rospy.loginfo("Got first executable segment.")

    while not rospy.is_shutdown() and count < args.num_steps:
        # seg shape: (<=s, 8)
        if seg is None or len(seg) == 0:
            try:
                seg, t_infer_start, t_infer_done = q_actions.get(timeout=args.max_wait_next_chunk)
                continue
            except queue.Empty:
                continue

        right_waypoints = seg[:, :7].copy()

        # ---- build traj for this segment ----
        t0 = time.time()
        traj = build_traj_from_right_waypoints(
            robot=robot,
            left_fixed_7=left_fixed,
            right_waypoints_7=right_waypoints,
            waypoint_dt=args.waypoint_dt,
            vel_scale=args.vel_scale,
            acc_scale=args.acc_scale,
            enable_moveit_retime=args.enable_moveit_retime,
        )
        t1 = time.time()

        # ---- collision check ----
        ok, bad_idx, msg = checker.is_trajectory_valid(traj, stride=args.collision_check_stride)
        t2 = time.time()

        if ok:
            exec_ok = group.execute(traj, wait=True)
            group.stop()
            if not exec_ok:
                rospy.logwarn("Execution failed, skipping this segment.")
        else:
            rospy.logwarn(f"[COLLISION] segment invalid at pt {bad_idx}: {msg}")
            if args.fallback_use_plan:
                right_target = right_waypoints[-1]
                fb_ok = fallback_plan_and_execute_to_target(
                    robot=robot,
                    left_fixed_7=left_fixed,
                    right_target_7=right_target,
                    planning_time=args.fallback_planning_time,
                    vel_scale=args.vel_scale,
                    acc_scale=args.acc_scale,
                )
                if not fb_ok:
                    rospy.logwarn("Fallback plan failed, skipping.")
            else:
                rospy.logwarn("Skipping due to collision risk.")

        # ---- gripper follow last step of seg ----
        rg = float(seg[-1, 7])
        if robot.gripper_r is not None:
            if rg < 0:
                robot.gripper_r.open()
            else:
                robot.gripper_r.close()

        seg_steps = int(seg.shape[0])
        count += seg_steps
        t3 = time.time()

        if args.print_timing:
            infer_dt = float(t_infer_done - t_infer_start)
            build_dt = float(t1 - t0)
            check_dt = float(t2 - t1)
            loop_dt = float(t3 - t0)
            rospy.loginfo(
                f"seg_steps={seg_steps} | "
                f"infer_dt={infer_dt:.3f}s build={build_dt:.3f}s check={check_dt:.3f}s loop={loop_dt:.3f}s | "
                f"manual_d={d} exec_steps={s}"
            )

        # ---- get newest next seg (queue already drops old) ----
        try:
            seg, t_infer_start, t_infer_done = q_actions.get(timeout=args.max_wait_next_chunk)
        except queue.Empty:
            rospy.logwarn("No new executable segment received in time; holding.")
            continue

        # optional interactive stop
        if count % 200 == 0:
            user_input = input("Continue executing? [y/n] ")
            if user_input.lower() != "y":
                rospy.loginfo("Execution stopped by user.")
                break

    stop_evt.set()
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))






