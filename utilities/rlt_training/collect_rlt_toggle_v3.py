from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from typing import Tuple

import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from moveit_msgs.msg import RobotState, RobotTrajectory
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from rlt_training.rollout_replay_buffer import RolloutKeyboard, RolloutReplayBuffer
from rlt_training.runtime_adapter import RLTActionRuntime
from tams_diana7_tools.dual_diana_helper import DualDianaHelper
from tams_diana7_tools.gripper_interfaces import GripperType


@dataclasses.dataclass
class Args:
    host: str = "134.100.39.19"
    port: int = 8000
    num_steps: int = 2000
    prompt: str = "pick up the spoon and feed the person"
    replay_dir: str = "./rlt_replay_toggle"
    rlt_checkpoint: str = ""
    rlt_device: str = "cuda"
    rlt_max_delta: float = 0.03
    rlt_risk_stop_threshold: float = 0.92
    explore_delta_std: float = 0.0
    explore_delta_max: float = 0.02
    rotate_wrist_180: bool = False
    convert_to_rgb: bool = True
    resize_hw: int = 224
    horizon: int = 16
    waypoint_dt: float = 0.20
    exec_steps: int = 8
    manual_d: int = 8
    enable_moveit_retime: bool = True
    vel_scale: float = 0.05
    acc_scale: float = 0.05
    collision_check_stride: int = 4
    collision_check_timeout: float = 0.15
    fallback_use_plan: bool = True
    fallback_planning_time: float = 0.5
    buffer_size: int = 2
    max_wait_next_chunk: float = 0.2
    label_every_chunk: bool = True
    print_timing: bool = True


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
            right7 = np.array([msg.position[name_to_idx[n]] for n in arm_r_names[:7]], dtype=np.float32)
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
        elif img.shape[0] != self.resize_hw or img.shape[1] != self.resize_hw:
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
        return {
            "observation/state": state8.astype(np.float32),
            "observation/image": self._process_image(top, do_rotate_180=False),
            "observation/wrist_image": self._process_image(wrist, do_rotate_180=self.rotate_wrist_180),
            "prompt": prompt,
        }


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


def build_traj_from_right_waypoints(robot, left_fixed_7, right_waypoints_7, waypoint_dt, vel_scale, acc_scale, enable_moveit_retime):
    group = robot.arms_group
    joint_names = _get_arms_joint_names(group)
    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(joint_names)
    cur14 = np.array(group.get_current_joint_values()[:14], dtype=np.float64)
    p0 = JointTrajectoryPoint()
    p0.positions = [float(x) for x in cur14]
    p0.time_from_start = rospy.Duration.from_sec(float(waypoint_dt))
    traj.joint_trajectory.points.append(p0)
    for i in range(right_waypoints_7.shape[0]):
        target14 = np.concatenate([left_fixed_7, right_waypoints_7[i]], axis=0)
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in target14]
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


class StateValidityChecker:
    def __init__(self, group_name: str, timeout: float = 0.15):
        srv_name = "/check_state_validity"
        rospy.loginfo(f"Waiting for service {srv_name} ...")
        rospy.wait_for_service(srv_name, timeout=5.0)
        self.group_name = group_name
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
        if (len(pts) - 1) % stride != 0:
            ok, msg = self.is_state_valid(jn, pts[-1].positions)
            if not ok:
                return False, len(pts) - 1, msg
        return True, -1, "valid"


def fallback_plan_and_execute_to_target(robot, left_fixed_7, right_target_7, planning_time, vel_scale, acc_scale) -> bool:
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
            return False
        plan = plan_msg
    else:
        plan = plan_ret
        if plan is None or len(plan.joint_trajectory.points) == 0:
            return False
    ok = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()
    return bool(ok)


class AsyncInferProducer(threading.Thread):
    def __init__(self, policy, sensor, prompt, horizon, exec_steps, manual_d, out_q, stop_evt):
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
            if acts.ndim != 2 or acts.shape[1] < 8 or acts.shape[0] < H:
                continue
            start = max(0, min(d, H - 1))
            end = min(start + s, H)
            if end <= start:
                continue
            seg = acts[:H][start:end].copy()
            try:
                if self.out_q.full():
                    _ = self.out_q.get_nowait()
                self.out_q.put_nowait((obs, seg, t0, t1))
            except queue.Full:
                pass


def apply_rlt_or_exploration(obs, pi_seg, keyboard, adapter, args):
    delta = np.zeros_like(pi_seg, dtype=np.float64)
    info = {"risk": 0.0, "value": 0.0, "source": "none", "stopped_by_risk": False}
    if not keyboard.rlt_enabled:
        return pi_seg.copy(), delta, info
    if adapter is not None:
        corrected, rlt_info = adapter.correct_observation_chunk(obs, pi_seg)
        delta = np.asarray(corrected, dtype=np.float64) - np.asarray(pi_seg, dtype=np.float64)
        info.update(rlt_info)
        info["source"] = "rlt"
        if rlt_info.get("stopped_by_risk", False):
            return pi_seg.copy(), np.zeros_like(pi_seg, dtype=np.float64), info
        return corrected, delta, info
    if args.explore_delta_std > 0:
        delta = np.random.normal(0.0, args.explore_delta_std, size=pi_seg.shape)
        delta[:, 7] = 0.0
        delta = np.clip(delta, -args.explore_delta_max, args.explore_delta_max)
        info["source"] = "random_explore"
        return (pi_seg + delta).astype(np.float64), delta.astype(np.float64), info
    info["source"] = "label_only"
    return pi_seg.copy(), delta, info


def main(args: Args) -> None:
    rospy.init_node("openpi_rlt_toggle_collector", anonymous=True)
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info(f"Server metadata: {policy.get_server_metadata()}")
    adapter = None
    if args.rlt_checkpoint:
        adapter = RLTActionRuntime(
            args.rlt_checkpoint,
            device=args.rlt_device,
            max_delta=args.rlt_max_delta,
            risk_stop_threshold=args.rlt_risk_stop_threshold,
        )
        rospy.loginfo(f"Loaded RLT checkpoint: {args.rlt_checkpoint}")

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
    joint_goal_r = np.deg2rad([-32, 5, -18, 95, -8, -85, -1])
    joint_goal_l = np.deg2rad([-32, -23, -49, 97, 11, -98, -1])
    joint_goal = np.concatenate([joint_goal_l, joint_goal_r, np.array([0.0]), np.array([1.0])])
    robot.move_to_joint_goal(joint_goal, use_moveit=True)
    left_fixed = np.array(joint_goal[:7], dtype=np.float64)

    sensor = SensorCache(args.rotate_wrist_180, args.convert_to_rgb, args.resize_hw)
    if not sensor.wait_ready(timeout=3.0):
        rospy.logwarn("Sensors not ready, will continue with zero images until ready.")
    _ = policy.infer(sensor.get_observation(args.prompt))
    group = robot.arms_group
    checker = StateValidityChecker(group_name=group.get_name(), timeout=args.collision_check_timeout)
    replay = RolloutReplayBuffer(args.replay_dir)
    keyboard = RolloutKeyboard()

    q_actions = queue.Queue(maxsize=args.buffer_size)
    stop_evt = threading.Event()
    producer = AsyncInferProducer(policy, sensor, args.prompt, args.horizon, args.exec_steps, args.manual_d, q_actions, stop_evt)
    producer.start()
    episode_id = f"ep_{int(time.time())}"
    chunk_id = 0
    count = 0
    obs, seg, t_infer_start, t_infer_done = q_actions.get()

    while not rospy.is_shutdown() and count < args.num_steps:
        if args.label_every_chunk:
            keyboard.poll_blocking()
            if keyboard.stop:
                break
            if keyboard.human_env_reset:
                episode_id = f"ep_{int(time.time())}"
                chunk_id = 0
                rospy.loginfo("Human environment reset marked. Starting a fresh post-reset segment.")
                obs, seg, t_infer_start, t_infer_done = q_actions.get()
        pi_seg = seg.copy()
        exec_seg, delta_seg, rlt_info = apply_rlt_or_exploration(obs, pi_seg, keyboard, adapter, args)
        if rlt_info.get("stopped_by_risk", False):
            rospy.logwarn(f"RLT risk gate held chunk: risk={rlt_info['risk']:.3f}")
            label = keyboard.make_label(exec_ok=False, collision_from_checker=False)
            replay.add_chunk(obs, pi_seg, pi_seg, label, episode_id, chunk_id, delta_actions=np.zeros_like(pi_seg))
            chunk_id += 1
            obs, seg, t_infer_start, t_infer_done = q_actions.get()
            continue

        right_waypoints = exec_seg[:, :7].copy()
        t0 = time.time()
        traj = build_traj_from_right_waypoints(robot, left_fixed, right_waypoints, args.waypoint_dt, args.vel_scale, args.acc_scale, args.enable_moveit_retime)
        t1 = time.time()
        traj_ok, bad_idx, msg = checker.is_trajectory_valid(traj, stride=args.collision_check_stride)
        t2 = time.time()
        exec_ok = False
        if traj_ok:
            exec_ok = bool(group.execute(traj, wait=True))
            group.stop()
        elif args.fallback_use_plan:
            rospy.logwarn(f"[COLLISION] segment invalid at pt {bad_idx}: {msg}")
            exec_ok = fallback_plan_and_execute_to_target(robot, left_fixed, right_waypoints[-1], args.fallback_planning_time, args.vel_scale, args.acc_scale)

        rg = float(exec_seg[-1, 7])
        if robot.gripper_r is not None:
            if rg < 0:
                robot.gripper_r.open()
            else:
                robot.gripper_r.close()

        label = keyboard.make_label(exec_ok=exec_ok, collision_from_checker=not traj_ok)
        replay.add_chunk(obs=obs, pi_actions=pi_seg, executed_actions=exec_seg, delta_actions=delta_seg, label=label, episode_id=episode_id, chunk_id=chunk_id)
        chunk_id += 1
        seg_steps = int(exec_seg.shape[0])
        count += seg_steps
        t3 = time.time()
        if args.print_timing:
            rospy.loginfo(
                f"saved chunk={chunk_id} steps={seg_steps} rlt={label.rlt_enabled} "
                f"src={rlt_info['source']} risk={rlt_info['risk']:.3f} value={rlt_info['value']:.3f} "
                f"delta_max={float(np.max(np.abs(delta_seg))):.4f} "
                f"infer={t_infer_done - t_infer_start:.3f}s build={t1 - t0:.3f}s "
                f"check={t2 - t1:.3f}s loop={t3 - t0:.3f}s "
                f"success={label.success} failure={label.failure} env_reset={label.human_env_reset}"
            )
        try:
            obs, seg, t_infer_start, t_infer_done = q_actions.get(timeout=args.max_wait_next_chunk)
        except queue.Empty:
            rospy.logwarn("No new executable segment received in time; holding.")
    stop_evt.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
