#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-prompt RLT replay collector for pi0.5 feeding.

This version is adapted from:
  - utilities/rlt_training/collect_rlt_toggle_v3.py
  - test_the_server_basic_v3_return.py

Key differences from collect_rlt_toggle_v3.py:
  1. Supports two prompts in the same policy server:
       FEED   : "pick up the spoon and feed the person"
       RETURN : "return the spoon to the start position"
  2. Keeps your old command semantics:
       r -> switch to RETURN prompt
       c -> switch back to FEED prompt
     Therefore RLT toggle is moved to:
       t -> toggle RLT residual/exploration on/off
     Collision label is moved to:
       x -> mark collision/unsafe behavior
  3. Uses robust right-arm state extraction and MoveIt joint-name mapping from
     your working two-prompt client.
  4. Uses synchronous chunk inference/execution to avoid stale async chunks after
     prompt switching.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from moveit_msgs.msg import RobotState, RobotTrajectory
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from rlt_training.rollout_replay_buffer import ChunkLabel, RolloutReplayBuffer
from rlt_training.runtime_adapter import RLTActionRuntime
from tams_diana7_tools.dual_diana_helper import DualDianaHelper
from tams_diana7_tools.gripper_interfaces import GripperType


ARM_L_NAMES = [
    "arm_l_joint_1",
    "arm_l_joint_2",
    "arm_l_joint_3",
    "arm_l_joint_4",
    "arm_l_joint_5",
    "arm_l_joint_6",
    "arm_l_joint_7",
]

ARM_R_NAMES = [
    "arm_r_joint_1",
    "arm_r_joint_2",
    "arm_r_joint_3",
    "arm_r_joint_4",
    "arm_r_joint_5",
    "arm_r_joint_6",
    "arm_r_joint_7",
]

GRIPPER_R_CANDIDATES = [
    "q_gripper_r_FJ",
    "q_gripper_r",
]


@dataclasses.dataclass
class Args:
    host: str = "134.100.39.19"
    port: int = 8000
    num_steps: int = 2000

    prompt: str = "pick up the spoon and feed the person"
    return_prompt: str = "return the spoon to the start position"
    replay_dir: str = "./rlt_replay_toggle_two_prompts"

    rlt_checkpoint: str = ""
    rlt_device: str = "cuda"
    rlt_max_delta: float = 0.03
    rlt_risk_stop_threshold: float = 0.92
    explore_delta_std: float = 0.0
    explore_delta_max: float = 0.02

    rotate_wrist_180: bool = False
    convert_to_rgb: bool = True
    resize_hw: int = 224

    horizon: int = 12
    exec_steps: int = 12
    manual_d: int = 0

    waypoint_dt: float = 0.10
    vel_scale: float = 0.05
    acc_scale: float = 0.05
    planning_time: float = 0.5
    allow_rlt_on_return: bool = False

    label_every_chunk: bool = True
    print_action_debug: bool = True
    print_timing: bool = True


class TwoPromptRolloutKeyboard:
    def __init__(self):
        self.rlt_enabled = False
        self.rlt_toggle_on = False
        self.rlt_toggle_off = False
        self.human_env_reset = False
        self.success = False
        self.failure = False
        self.collision = False
        self.stop = False
        self.reset_type = ""
        self.phase = "unknown"
        self.note = ""
        self.task_cmd: Optional[str] = None

    def poll_blocking(self, current_mode: str, current_rlt_allowed: bool) -> None:
        msg = (
            f"[cmd mode={current_mode} rlt={self.rlt_enabled and current_rlt_allowed}] "
            "Enter=continue | r=RETURN | c=FEED | t=toggle-RLT | "
            "e=env-reset | s=success | f=failure | x=collision | "
            "1=approach 2=grasp 3=lift 4=mouth 5=feed 6=retreat | q=quit > "
        )
        try:
            raw = input(msg).strip().lower()
        except EOFError:
            self.stop = True
            self.note = "eof_quit"
            return

        self.task_cmd = None

        if raw == "":
            self.note = ""
            return

        tokens = raw.split() if " " in raw else list(raw)

        for cmd in tokens:
            if cmd == "r":
                self.task_cmd = "RETURN"
                self.note = "switch_return"
            elif cmd == "c":
                self.task_cmd = "FEED"
                self.note = "switch_feed"
            elif cmd == "t":
                self.rlt_enabled = not self.rlt_enabled
                self.rlt_toggle_on = self.rlt_enabled
                self.rlt_toggle_off = not self.rlt_enabled
                self.note = "rlt_on" if self.rlt_enabled else "rlt_off"
            elif cmd == "e":
                self.human_env_reset = True
                self.reset_type = input("[cmd] reset type, e.g. spoon_reposition > ").strip()
                self.note = "human_env_reset"
            elif cmd == "s":
                self.success = True
                self.failure = False
                self.note = "success"
            elif cmd == "f":
                self.failure = True
                self.success = False
                self.note = "failure"
            elif cmd == "x":
                self.collision = True
                self.note = "collision"
            elif cmd == "q":
                self.stop = True
                self.note = "quit"
            elif cmd in {"1", "2", "3", "4", "5", "6"}:
                phases = {
                    "1": "approach_spoon",
                    "2": "grasp_spoon",
                    "3": "lift_food",
                    "4": "approach_mouth",
                    "5": "feed",
                    "6": "retreat",
                }
                self.phase = phases[cmd]
                self.note = self.phase
            else:
                rospy.logwarn(f"Unknown keyboard command ignored: {cmd}")

    def make_label(self, exec_ok: bool, collision_from_checker: bool, current_mode: str, rlt_allowed: bool) -> ChunkLabel:
        effective_rlt = bool(self.rlt_enabled and rlt_allowed)
        label = ChunkLabel(
            success=self.success,
            failure=self.failure,
            collision=self.collision or collision_from_checker,
            exec_ok=exec_ok,
            rlt_enabled=effective_rlt,
            rlt_toggle_on=self.rlt_toggle_on and rlt_allowed,
            rlt_toggle_off=self.rlt_toggle_off,
            human_env_reset=self.human_env_reset,
            cut_before_reset=self.human_env_reset,
            reset_type=self.reset_type,
            phase=self.phase,
            note=f"mode={current_mode};{self.note}" if self.note else f"mode={current_mode}",
        )
        self.success = False
        self.failure = False
        self.collision = False
        self.rlt_toggle_on = False
        self.rlt_toggle_off = False
        self.human_env_reset = False
        self.reset_type = ""
        self.note = ""
        return label


class SensorCache:
    def __init__(self, rotate_wrist_180: bool, convert_to_rgb: bool, resize_hw: int):
        import threading

        self.rotate_wrist_180 = rotate_wrist_180
        self.convert_to_rgb = convert_to_rgb
        self.resize_hw = resize_hw
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._right_state_8 = np.zeros((8,), dtype=np.float32)
        self._got_right_arm = False
        self._top_img = None
        self._wrist_img = None
        self._sub_js = rospy.Subscriber("/joint_states", JointState, self._joint_cb, queue_size=1)
        self._sub_top = rospy.Subscriber("/top_view/color/image_raw", Image, self._top_cb, queue_size=1)
        self._sub_wrist = rospy.Subscriber("/diana_R_view/color/image_raw", Image, self._wrist_cb, queue_size=1)

    def _joint_cb(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name) if i < len(msg.position)}
        state = self._right_state_8.copy()
        got_right = self._got_right_arm
        if all(n in name_to_idx for n in ARM_R_NAMES):
            state[:7] = np.array([msg.position[name_to_idx[n]] for n in ARM_R_NAMES], dtype=np.float32)
            got_right = True
        elif len(msg.position) >= 14:
            arm_r_names = [n for n in msg.name if "arm_r" in n]
            if len(arm_r_names) >= 7:
                try:
                    state[:7] = np.array([msg.position[name_to_idx[n]] for n in arm_r_names[:7]], dtype=np.float32)
                    got_right = True
                except Exception:
                    pass
            else:
                state[:7] = np.array(msg.position[7:14], dtype=np.float32)
                got_right = True
        for gn in GRIPPER_R_CANDIDATES:
            if gn in name_to_idx:
                state[7] = np.float32(msg.position[name_to_idx[gn]])
                break
        with self._lock:
            self._right_state_8 = state
            self._got_right_arm = got_right

    def _top_cb(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._top_img = img

    def _wrist_cb(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._wrist_img = img

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
                ok = (self._top_img is not None) and (self._wrist_img is not None) and self._got_right_arm
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


def _fmt_arr(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=3, suppress_small=True, separator=", ")


def _get_arms_joint_names(arms_group) -> list:
    joints = arms_group.get_active_joints()
    filtered = [jn for jn in joints if ("arm_l" in jn or "arm_r" in jn)]
    if len(filtered) < 14:
        rospy.logwarn(f"Less than 14 arm joints found by name. Fallback to joints[:14]. joints={joints}")
        filtered = joints[:14]
    else:
        filtered = filtered[:14]
    return filtered


def _robot_state_from_joints(joint_names, joint_positions):
    rs = RobotState()
    rs.joint_state = JointState()
    rs.joint_state.name = list(joint_names)
    rs.joint_state.position = [float(x) for x in joint_positions]
    return rs


def _make_target14_by_name(joint_names: List[str], left_fixed_7: np.ndarray, right_target_7: np.ndarray) -> List[float]:
    left_fixed_7 = np.asarray(left_fixed_7, dtype=np.float64).reshape(7)
    right_target_7 = np.asarray(right_target_7, dtype=np.float64).reshape(7)
    value_map: Dict[str, float] = {}
    for name, val in zip(ARM_L_NAMES, left_fixed_7):
        value_map[name] = float(val)
    for name, val in zip(ARM_R_NAMES, right_target_7):
        value_map[name] = float(val)
    if all(jn in value_map for jn in joint_names):
        return [value_map[jn] for jn in joint_names]
    rospy.logwarn_throttle(
        1.0,
        "MoveIt joint names do not fully match ARM_L_NAMES/ARM_R_NAMES. "
        f"Fallback to concatenate order. joint_names={joint_names}",
    )
    target14 = np.concatenate([left_fixed_7, right_target_7], axis=0)
    return [float(x) for x in target14]


def _concat_trajs(trajs, joint_names):
    full = RobotTrajectory()
    full.joint_trajectory.joint_names = list(joint_names)
    t_offset = 0.0
    last_pos = None
    for seg_idx, tr in enumerate(trajs):
        pts = tr.joint_trajectory.points
        if not pts:
            continue
        for i, p in enumerate(pts):
            if seg_idx > 0 and i == 0:
                continue
            pos = np.array(p.positions, dtype=np.float64)
            if last_pos is not None and np.max(np.abs(pos - last_pos)) < 1e-9:
                continue
            npnt = JointTrajectoryPoint()
            npnt.positions = list(p.positions)
            npnt.velocities = list(p.velocities) if p.velocities else []
            npnt.accelerations = list(p.accelerations) if p.accelerations else []
            npnt.effort = []
            npnt.time_from_start = rospy.Duration.from_sec(t_offset + p.time_from_start.to_sec())
            full.joint_trajectory.points.append(npnt)
            last_pos = pos
        if full.joint_trajectory.points:
            t_offset = full.joint_trajectory.points[-1].time_from_start.to_sec()
    return full


def _retime_uniform_dt(traj: RobotTrajectory, dt: float) -> RobotTrajectory:
    out = RobotTrajectory()
    out.joint_trajectory.joint_names = traj.joint_trajectory.joint_names
    for i, p in enumerate(traj.joint_trajectory.points):
        npnt = JointTrajectoryPoint()
        npnt.positions = list(p.positions)
        npnt.velocities = []
        npnt.accelerations = []
        npnt.effort = []
        npnt.time_from_start = rospy.Duration.from_sec((i + 1) * float(dt))
        out.joint_trajectory.points.append(npnt)
    return out


def plan_and_execute_horizon_with_arms_group(
    robot: DualDianaHelper,
    left_fixed_7: np.ndarray,
    right_waypoints_7: np.ndarray,
    planning_time: float,
    waypoint_dt: float,
    vel_scale: float,
    acc_scale: float,
) -> bool:
    if not hasattr(robot, "arms_group") or robot.arms_group is None:
        rospy.logerr("robot.arms_group is None. Need DualDianaHelper(load_moveit=True) and both arms active.")
        return False
    group = robot.arms_group
    group.set_planning_time(float(planning_time))
    group.set_max_velocity_scaling_factor(float(vel_scale))
    group.set_max_acceleration_scaling_factor(float(acc_scale))
    joint_names = _get_arms_joint_names(group)
    if len(joint_names) != 14:
        rospy.logwarn(f"arms joint_names length is {len(joint_names)}: {joint_names}")
    seg_trajs = []
    cur_start_14 = None
    for k in range(right_waypoints_7.shape[0]):
        target14 = _make_target14_by_name(joint_names, left_fixed_7, right_waypoints_7[k])
        if cur_start_14 is not None:
            group.set_start_state(_robot_state_from_joints(joint_names, cur_start_14))
        else:
            group.set_start_state_to_current_state()
        group.set_joint_value_target({jn: float(p) for jn, p in zip(joint_names, target14)})
        plan_ret = group.plan()
        if isinstance(plan_ret, tuple):
            success, plan_msg, _, _ = plan_ret
            if (not success) or plan_msg is None or len(plan_msg.joint_trajectory.points) == 0:
                rospy.logwarn(f"MoveIt plan failed at waypoint {k}")
                return False
            seg_trajs.append(plan_msg)
        else:
            plan_msg = plan_ret
            if plan_msg is None or len(plan_msg.joint_trajectory.points) == 0:
                rospy.logwarn(f"MoveIt plan failed at waypoint {k}")
                return False
            seg_trajs.append(plan_msg)
        cur_start_14 = np.array(seg_trajs[-1].joint_trajectory.points[-1].positions, dtype=np.float64)
    full = _concat_trajs(seg_trajs, joint_names)
    if len(full.joint_trajectory.points) == 0:
        rospy.logwarn("Concatenated trajectory is empty.")
        return False
    full = _retime_uniform_dt(full, dt=float(waypoint_dt))
    try:
        full = group.retime_trajectory(
            robot.robot.get_current_state(),
            full,
            velocity_scaling_factor=float(vel_scale),
            acceleration_scaling_factor=float(acc_scale),
            algorithm="time_optimal_trajectory_generation",
        )
    except Exception as e:
        rospy.logwarn(f"retime_trajectory failed, executing uniform-dt trajectory: {e}")
    ok = group.execute(full, wait=True)
    group.stop()
    group.clear_pose_targets()
    return bool(ok)


def crop_action_segment(actions: np.ndarray, horizon: int, manual_d: int, exec_steps: int) -> np.ndarray:
    acts = np.asarray(actions, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] < 8:
        raise ValueError(f"Expected actions with shape [T, >=8], got {acts.shape}")
    if acts.shape[0] < horizon:
        raise ValueError(f"Policy returned only {acts.shape[0]} actions, need horizon={horizon}")
    H = int(horizon)
    d = max(0, min(int(manual_d), H - 1))
    s = int(exec_steps)
    end = min(d + s, H)
    if end <= d:
        raise ValueError(f"Invalid crop: manual_d={manual_d}, exec_steps={exec_steps}, horizon={horizon}")
    return acts[:H][d:end].copy()


def apply_rlt_or_exploration(obs, pi_seg, keyboard: TwoPromptRolloutKeyboard, adapter, args: Args, current_mode: str):
    delta = np.zeros_like(pi_seg, dtype=np.float64)
    info = {"risk": 0.0, "value": 0.0, "source": "none", "stopped_by_risk": False}
    rlt_allowed = (current_mode == "FEED") or bool(args.allow_rlt_on_return)
    if not keyboard.rlt_enabled or not rlt_allowed:
        if keyboard.rlt_enabled and not rlt_allowed:
            info["source"] = "blocked_in_return"
        return pi_seg.copy(), delta, info, rlt_allowed
    if adapter is not None:
        corrected, rlt_info = adapter.correct_observation_chunk(obs, pi_seg)
        delta = np.asarray(corrected, dtype=np.float64) - np.asarray(pi_seg, dtype=np.float64)
        info.update(rlt_info)
        info["source"] = "rlt"
        if rlt_info.get("stopped_by_risk", False):
            return pi_seg.copy(), np.zeros_like(pi_seg, dtype=np.float64), info, rlt_allowed
        return corrected, delta, info, rlt_allowed
    if args.explore_delta_std > 0:
        delta = np.random.normal(0.0, args.explore_delta_std, size=pi_seg.shape)
        delta[:, 7] = 0.0
        delta = np.clip(delta, -args.explore_delta_max, args.explore_delta_max)
        info["source"] = "random_explore"
        return (pi_seg + delta).astype(np.float64), delta.astype(np.float64), info, rlt_allowed
    info["source"] = "label_only"
    return pi_seg.copy(), delta, info, rlt_allowed


def main(args: Args) -> None:
    rospy.init_node("openpi_rlt_two_prompt_collector", anonymous=True)
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
    joint_goal_gripper_l = np.array([0.0], dtype=np.float32)
    joint_goal_gripper_r = np.array([1.0], dtype=np.float32)
    joint_goal = np.concatenate([joint_goal_l, joint_goal_r, joint_goal_gripper_l, joint_goal_gripper_r])
    robot.move_to_joint_goal(joint_goal, use_moveit=True)
    left_fixed = np.array(joint_goal_l, dtype=np.float64)
    joint_names = _get_arms_joint_names(robot.arms_group)
    rospy.loginfo("MoveIt arm joint order:")
    for i, jn in enumerate(joint_names):
        rospy.loginfo(f"  [{i}] {jn}")
    sensor = SensorCache(args.rotate_wrist_180, args.convert_to_rgb, args.resize_hw)
    if not sensor.wait_ready(timeout=3.0):
        rospy.logwarn("Sensors not fully ready. Collector will continue, but replay may contain zero images/state until topics arrive.")
    replay = RolloutReplayBuffer(args.replay_dir)
    keyboard = TwoPromptRolloutKeyboard()
    current_mode = "FEED"
    current_prompt = args.prompt
    episode_id = f"ep_{current_mode}_{int(time.time())}"
    chunk_id = 0
    count = 0
    rospy.loginfo(
        "=== RUNNING TWO-PROMPT RLT COLLECTOR ===\n"
        f"H={args.horizon}, exec_steps={args.exec_steps}, manual_d={args.manual_d}, dt={args.waypoint_dt}\n"
        f"[FEED  ] prompt: {args.prompt}\n"
        f"[RETURN] prompt: {args.return_prompt}\n"
        "Commands per chunk:\n"
        "  r : switch to RETURN task prompt\n"
        "  c : switch back to FEED task prompt\n"
        "  t : toggle RLT residual/exploration on/off\n"
        "  x : mark collision/unsafe behavior\n"
    )
    rospy.loginfo("Warming up policy inference for FEED prompt...")
    policy.infer(sensor.get_observation(args.prompt))
    rospy.loginfo("Warming up policy inference for RETURN prompt...")
    policy.infer(sensor.get_observation(args.return_prompt))
    start_time = time.time()
    while not rospy.is_shutdown() and count < int(args.num_steps):
        rlt_allowed_for_mode = (current_mode == "FEED") or bool(args.allow_rlt_on_return)
        if args.label_every_chunk:
            keyboard.poll_blocking(current_mode=current_mode, current_rlt_allowed=rlt_allowed_for_mode)
            if keyboard.stop:
                break
            if keyboard.task_cmd == "RETURN":
                current_mode = "RETURN"
                current_prompt = args.return_prompt
                if not args.allow_rlt_on_return and keyboard.rlt_enabled:
                    keyboard.rlt_enabled = False
                    rospy.logwarn("[MODE] Switched to RETURN and disabled RLT because allow_rlt_on_return=False.")
                else:
                    rospy.logwarn(f"[MODE] Switched to RETURN. prompt='{current_prompt}'")
            elif keyboard.task_cmd == "FEED":
                current_mode = "FEED"
                current_prompt = args.prompt
                rospy.logwarn(f"[MODE] Switched to FEED. prompt='{current_prompt}'")
            if keyboard.human_env_reset:
                episode_id = f"ep_{current_mode}_{int(time.time())}"
                chunk_id = 0
                rospy.loginfo("Human environment reset marked. Starting a fresh post-reset segment.")
        obs = sensor.get_observation(current_prompt)
        if args.print_action_debug:
            rospy.loginfo(
                f"[OBS] mode={current_mode}, prompt='{current_prompt}', "
                f"state={_fmt_arr(obs['observation/state'])}"
            )
        t_infer0 = time.time()
        result = policy.infer(obs)
        t_infer1 = time.time()
        if not isinstance(result, dict) or "actions" not in result:
            rospy.logwarn(f"Policy output has no 'actions': type={type(result)}")
            continue
        try:
            pi_seg = crop_action_segment(result["actions"], args.horizon, args.manual_d, args.exec_steps)
        except ValueError as e:
            rospy.logwarn(str(e))
            continue
        exec_seg, delta_seg, rlt_info, rlt_allowed_for_mode = apply_rlt_or_exploration(
            obs=obs,
            pi_seg=pi_seg,
            keyboard=keyboard,
            adapter=adapter,
            args=args,
            current_mode=current_mode,
        )
        if args.print_action_debug:
            rospy.loginfo(
                f"[ACTION] mode={current_mode}, src={rlt_info['source']}, "
                f"first={_fmt_arr(exec_seg[0])}, last={_fmt_arr(exec_seg[-1])}, "
                f"infer_dt={t_infer1 - t_infer0:.3f}s, delta_max={float(np.max(np.abs(delta_seg))):.4f}"
            )
        if rlt_info.get("stopped_by_risk", False):
            rospy.logwarn(f"RLT risk gate held chunk: risk={rlt_info['risk']:.3f}")
            label = keyboard.make_label(exec_ok=False, collision_from_checker=False, current_mode=current_mode, rlt_allowed=rlt_allowed_for_mode)
            replay.add_chunk(obs, pi_seg, pi_seg, label, episode_id, chunk_id, delta_actions=np.zeros_like(pi_seg))
            chunk_id += 1
            continue
        right_waypoints = exec_seg[:, :7].copy()
        t_exec0 = time.time()
        exec_ok = plan_and_execute_horizon_with_arms_group(
            robot=robot,
            left_fixed_7=left_fixed,
            right_waypoints_7=right_waypoints,
            planning_time=args.planning_time,
            waypoint_dt=args.waypoint_dt,
            vel_scale=args.vel_scale,
            acc_scale=args.acc_scale,
        )
        t_exec1 = time.time()
        if exec_ok:
            rg = float(exec_seg[-1, 7])
            if robot.gripper_r is not None:
                if rg < 0:
                    robot.gripper_r.open()
                else:
                    robot.gripper_r.close()
        else:
            rospy.logwarn("Chunk planning/execution failed. Saving negative/failed replay label if marked by keyboard/exec_ok.")
        label = keyboard.make_label(exec_ok=exec_ok, collision_from_checker=False, current_mode=current_mode, rlt_allowed=rlt_allowed_for_mode)
        replay.add_chunk(
            obs=obs,
            pi_actions=pi_seg,
            executed_actions=exec_seg,
            delta_actions=delta_seg,
            label=label,
            episode_id=episode_id,
            chunk_id=chunk_id,
        )
        chunk_id += 1
        seg_steps = int(exec_seg.shape[0])
        if exec_ok:
            count += seg_steps
        if args.print_timing:
            rospy.loginfo(
                f"saved chunk={chunk_id} mode={current_mode} steps={seg_steps} total_steps={count}/{args.num_steps} "
                f"rlt={label.rlt_enabled} src={rlt_info['source']} "
                f"risk={rlt_info['risk']:.3f} value={rlt_info['value']:.3f} "
                f"delta_max={float(np.max(np.abs(delta_seg))):.4f} "
                f"infer={t_infer1 - t_infer0:.3f}s exec={t_exec1 - t_exec0:.3f}s "
                f"success={label.success} failure={label.failure} collision={label.collision} "
                f"env_reset={label.human_env_reset} phase={label.phase}"
            )
    end_time = time.time()
    rospy.loginfo(
        f"Finished. executed_steps={count}, total_time={end_time - start_time:.2f}s, "
        f"mode={current_mode}, replay_dir={args.replay_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
