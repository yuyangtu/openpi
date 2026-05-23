#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from moveit_msgs.msg import RobotState, RobotTrajectory
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image, JointState
from tams_diana7_tools.dual_diana_helper import DualDianaHelper
from tams_diana7_tools.gripper_interfaces import GripperType
from trajectory_msgs.msg import JointTrajectoryPoint

from rlt_training.episode_replay_buffer import EpisodeChunk, EpisodeReplayRecorder
from rlt_training.runtime_adapter import RLTActionRuntime


ARM_L_NAMES = [
    "arm_l_joint_1", "arm_l_joint_2", "arm_l_joint_3", "arm_l_joint_4",
    "arm_l_joint_5", "arm_l_joint_6", "arm_l_joint_7",
]
ARM_R_NAMES = [
    "arm_r_joint_1", "arm_r_joint_2", "arm_r_joint_3", "arm_r_joint_4",
    "arm_r_joint_5", "arm_r_joint_6", "arm_r_joint_7",
]
GRIPPER_R_CANDIDATES = ["q_gripper_r_FJ", "q_gripper_r"]


@dataclasses.dataclass
class Args:
    host: str = "134.100.39.19"
    port: int = 8000
    num_steps: int = 2000
    prompt: str = "pick up the spoon and feed the person"
    return_prompt: str = "return the spoon to the start position"
    replay_dir: str = "./rlt_episode_replay"
    episode_gamma: float = 0.97
    rlt_checkpoint: str = ""
    rlt_device: str = "cuda"
    rlt_max_delta: float = 0.03
    rlt_risk_stop_threshold: float = 0.95
    explore_delta_std: float = 0.0
    explore_delta_max: float = 0.02
    rlt_on_return: bool = False
    rotate_wrist_180: bool = False
    convert_to_rgb: bool = True
    resize_hw: int = 224
    horizon: int = 12
    waypoint_dt: float = 0.10
    vel_scale: float = 0.05
    acc_scale: float = 0.05
    planning_time: float = 0.5
    print_action_debug: bool = True


class CommandWatcher(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._cmd: Optional[str] = None
        self._stop_evt = threading.Event()

    def run(self):
        prompt = (
            "[cmd] Enter=continue, r=RETURN, c=FEED, t=toggle-RLT, "
            "s=episode-success, f=episode-failure, x=unsafe, "
            "e=env-reset, n=new-neutral, q=quit: "
        )
        while not self._stop_evt.is_set() and not rospy.is_shutdown():
            try:
                value = input(prompt).strip().lower()
            except EOFError:
                break
            with self._lock:
                self._cmd = value

    def pop(self) -> Optional[str]:
        with self._lock:
            value = self._cmd
            self._cmd = None
        return value

    def stop(self):
        self._stop_evt.set()


def _fmt_arr(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=3, suppress_small=True, separator=", ")


def observation_ball(prompt: str, timeout: float = 2.0, rotate_wrist_180: bool = True, convert_to_rgb: bool = True, resize_hw: int = 224) -> dict:
    latest = {"right_state_8": np.zeros((8,), dtype=np.float32), "got_right_arm": False, "image": None, "wrist_image": None}
    bridge = CvBridge()

    def joint_state_callback(msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name) if i < len(msg.position)}
        if all(n in name_to_idx for n in ARM_R_NAMES):
            latest["right_state_8"][:7] = np.array([msg.position[name_to_idx[n]] for n in ARM_R_NAMES], dtype=np.float32)
            latest["got_right_arm"] = True
        elif len(msg.position) >= 14:
            arm_r_names = [n for n in msg.name if "arm_r" in n]
            if len(arm_r_names) >= 7:
                try:
                    latest["right_state_8"][:7] = np.array([msg.position[name_to_idx[n]] for n in arm_r_names[:7]], dtype=np.float32)
                    latest["got_right_arm"] = True
                except Exception:
                    pass
            else:
                latest["right_state_8"][:7] = np.array(msg.position[7:14], dtype=np.float32)
                latest["got_right_arm"] = True
        for name in GRIPPER_R_CANDIDATES:
            if name in name_to_idx:
                latest["right_state_8"][7] = np.float32(msg.position[name_to_idx[name]])
                break

    def image_callback(msg: Image):
        latest["image"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def wrist_image_callback(msg: Image):
        latest["wrist_image"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    joint_state_sub = rospy.Subscriber("/joint_states", JointState, joint_state_callback, queue_size=1)
    image_state_sub = rospy.Subscriber("/top_view/color/image_raw", Image, image_callback, queue_size=1)
    wrist_image_state_sub = rospy.Subscriber("/diana_R_view/color/image_raw", Image, wrist_image_callback, queue_size=1)
    start_time = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        got_img = latest["image"] is not None
        got_wrist = latest["wrist_image"] is not None
        got_right = bool(latest["got_right_arm"])
        if got_img and got_wrist and got_right:
            break
        if rospy.Time.now().to_sec() - start_time > timeout:
            rospy.logwarn(f"Observation timeout. got_img={got_img}, got_wrist={got_wrist}, got_right_arm={got_right}")
            break
        rospy.sleep(0.02)
    joint_state_sub.unregister()
    image_state_sub.unregister()
    wrist_image_state_sub.unregister()

    def process_image(img, do_rotate_180: bool):
        import cv2
        if img is None:
            img = np.zeros((resize_hw, resize_hw, 3), dtype=np.uint8)
        elif img.shape[0] != resize_hw or img.shape[1] != resize_hw:
            img = cv2.resize(img, (resize_hw, resize_hw))
        if do_rotate_180:
            img = cv2.flip(img, -1)
        if convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    return {
        "observation/state": latest["right_state_8"].astype(np.float32),
        "observation/image": process_image(latest["image"], do_rotate_180=False),
        "observation/wrist_image": process_image(latest["wrist_image"], do_rotate_180=rotate_wrist_180),
        "prompt": prompt,
    }


def _get_arms_joint_names(arms_group) -> list:
    joints = arms_group.get_active_joints()
    filtered = [jn for jn in joints if ("arm_l" in jn or "arm_r" in jn)]
    if len(filtered) < 14:
        rospy.logwarn(f"Less than 14 arm joints found by name. Fallback to joints[:14]. joints={joints}")
        return joints[:14]
    return filtered[:14]


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
    for name, value in zip(ARM_L_NAMES, left_fixed_7):
        value_map[name] = float(value)
    for name, value in zip(ARM_R_NAMES, right_target_7):
        value_map[name] = float(value)
    if all(joint_name in value_map for joint_name in joint_names):
        return [value_map[joint_name] for joint_name in joint_names]
    target14 = np.concatenate([left_fixed_7, right_target_7], axis=0)
    return [float(x) for x in target14]


def _concat_trajs(trajs, joint_names):
    full = RobotTrajectory()
    full.joint_trajectory.joint_names = list(joint_names)
    t_offset = 0.0
    last_pos = None
    for seg_idx, traj in enumerate(trajs):
        for point_idx, point in enumerate(traj.joint_trajectory.points):
            if seg_idx > 0 and point_idx == 0:
                continue
            pos = np.array(point.positions, dtype=np.float64)
            if last_pos is not None and np.max(np.abs(pos - last_pos)) < 1e-9:
                continue
            new_point = JointTrajectoryPoint()
            new_point.positions = list(point.positions)
            new_point.velocities = list(point.velocities) if point.velocities else []
            new_point.accelerations = list(point.accelerations) if point.accelerations else []
            new_point.effort = []
            new_point.time_from_start = rospy.Duration.from_sec(t_offset + point.time_from_start.to_sec())
            full.joint_trajectory.points.append(new_point)
            last_pos = pos
        if full.joint_trajectory.points:
            t_offset = full.joint_trajectory.points[-1].time_from_start.to_sec()
    return full


def _retime_uniform_dt(traj: RobotTrajectory, dt: float) -> RobotTrajectory:
    out = RobotTrajectory()
    out.joint_trajectory.joint_names = traj.joint_trajectory.joint_names
    for i, point in enumerate(traj.joint_trajectory.points):
        new_point = JointTrajectoryPoint()
        new_point.positions = list(point.positions)
        new_point.velocities = []
        new_point.accelerations = []
        new_point.effort = []
        new_point.time_from_start = rospy.Duration.from_sec((i + 1) * float(dt))
        out.joint_trajectory.points.append(new_point)
    return out


def plan_and_execute_horizon_with_arms_group(robot: DualDianaHelper, left_fixed_7: np.ndarray, right_waypoints_7: np.ndarray, planning_time: float, waypoint_dt: float, vel_scale: float, acc_scale: float) -> bool:
    if not hasattr(robot, "arms_group") or robot.arms_group is None:
        rospy.logerr("robot.arms_group is None.")
        return False
    group = robot.arms_group
    group.set_planning_time(float(planning_time))
    group.set_max_velocity_scaling_factor(float(vel_scale))
    group.set_max_acceleration_scaling_factor(float(acc_scale))
    joint_names = _get_arms_joint_names(group)
    seg_trajs = []
    cur_start_14 = None
    for k in range(right_waypoints_7.shape[0]):
        target14 = _make_target14_by_name(joint_names, left_fixed_7, right_waypoints_7[k])
        if cur_start_14 is not None:
            group.set_start_state(_robot_state_from_joints(joint_names, cur_start_14))
        else:
            group.set_start_state_to_current_state()
        group.set_joint_value_target({joint_name: float(pos) for joint_name, pos in zip(joint_names, target14)})
        plan_ret = group.plan()
        if isinstance(plan_ret, tuple):
            success, plan_msg, _, _ = plan_ret
            if (not success) or plan_msg is None or len(plan_msg.joint_trajectory.points) == 0:
                rospy.logwarn(f"MoveIt plan failed at waypoint {k}")
                return False
            seg_trajs.append(plan_msg)
        else:
            if plan_ret is None or len(plan_ret.joint_trajectory.points) == 0:
                rospy.logwarn(f"MoveIt plan failed at waypoint {k}")
                return False
            seg_trajs.append(plan_ret)
        cur_start_14 = np.array(seg_trajs[-1].joint_trajectory.points[-1].positions, dtype=np.float64)
    full = _concat_trajs(seg_trajs, joint_names)
    if len(full.joint_trajectory.points) == 0:
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
    except Exception as exc:
        rospy.logwarn(f"retime_trajectory failed, executing uniform-dt trajectory: {exc}")
    ok = group.execute(full, wait=True)
    group.stop()
    group.clear_pose_targets()
    return bool(ok)


def apply_rlt_or_exploration(obs: dict, pi_actions: np.ndarray, rlt_enabled: bool, current_mode: str, adapter: RLTActionRuntime | None, args: Args) -> tuple[np.ndarray, np.ndarray, dict]:
    delta = np.zeros_like(pi_actions, dtype=np.float64)
    info = {"source": "none", "risk": 0.0, "value": 0.0, "stopped_by_risk": False}
    if not rlt_enabled:
        return pi_actions.copy(), delta, info
    if current_mode == "RETURN" and not args.rlt_on_return:
        info["source"] = "disabled_on_return"
        return pi_actions.copy(), delta, info
    if adapter is not None:
        corrected, rlt_info = adapter.correct_observation_chunk(obs, pi_actions)
        delta = np.asarray(corrected, dtype=np.float64) - np.asarray(pi_actions, dtype=np.float64)
        info.update(rlt_info)
        info["source"] = "rlt"
        if rlt_info.get("stopped_by_risk", False):
            return pi_actions.copy(), np.zeros_like(pi_actions, dtype=np.float64), info
        return corrected, delta, info
    if args.explore_delta_std > 0:
        delta = np.random.normal(0.0, args.explore_delta_std, size=pi_actions.shape)
        delta[:, 7] = 0.0
        delta = np.clip(delta, -args.explore_delta_max, args.explore_delta_max)
        info["source"] = "random_explore"
        return (pi_actions + delta).astype(np.float64), delta.astype(np.float64), info
    info["source"] = "label_only"
    return pi_actions.copy(), delta, info


def main(args: Args) -> None:
    rospy.init_node("openpi_policy_client_two_prompts_rlt_episode", anonymous=True)
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info(f"Server metadata: {policy.get_server_metadata()}")
    rospy.sleep(1.0)
    adapter = None
    if args.rlt_checkpoint:
        adapter = RLTActionRuntime(args.rlt_checkpoint, device=args.rlt_device, max_delta=args.rlt_max_delta, risk_stop_threshold=args.rlt_risk_stop_threshold)
        rospy.loginfo(f"Loaded RLT checkpoint: {args.rlt_checkpoint}")
    recorder = EpisodeReplayRecorder(args.replay_dir, gamma=args.episode_gamma)
    robot = DualDianaHelper(load_moveit=True, arm_l_active=True, arm_r_active=True, no_grippers=False, gripper_l=None, gripper_r=GripperType.Q_GRIPPER, fetch_joint_states=True, load_joint_trajectory_controller=True)
    robot.load_position_trajectory_controllers()
    joint_goal_r = np.deg2rad([-32, 5, -18, 95, -8, -85, -1])
    joint_goal_l = np.deg2rad([-32, -23, -49, 97, 11, -98, -1])
    joint_goal = np.concatenate([joint_goal_l, joint_goal_r, np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)])
    robot.move_to_joint_goal(joint_goal, use_moveit=True)
    left_fixed = np.array(joint_goal_l, dtype=np.float64)
    joint_names = _get_arms_joint_names(robot.arms_group)
    rospy.loginfo("MoveIt arm joint order:")
    for i, joint_name in enumerate(joint_names):
        rospy.loginfo(f"  [{i}] {joint_name}")
    cmd_watcher = CommandWatcher()
    cmd_watcher.start()
    current_mode = "FEED"
    current_prompt = args.prompt
    rlt_enabled = False
    rlt_toggle_on_next = False
    rlt_toggle_off_next = False
    rospy.loginfo("=== TWO-PROMPT RLT EPISODE VERSION ===")
    policy.infer(observation_ball(current_prompt, rotate_wrist_180=args.rotate_wrist_180, convert_to_rgb=args.convert_to_rgb, resize_hw=args.resize_hw))
    count = 0
    start_time = time.time()
    try:
        while not rospy.is_shutdown() and count < int(args.num_steps):
            cmd = cmd_watcher.pop()
            if cmd == "r":
                current_mode = "RETURN"
                current_prompt = args.return_prompt
                rospy.logwarn(f"[MODE] Switched to RETURN. prompt='{current_prompt}'")
            elif cmd == "c":
                current_mode = "FEED"
                current_prompt = args.prompt
                rospy.logwarn(f"[MODE] Switched to FEED. prompt='{current_prompt}'")
            elif cmd == "t":
                rlt_enabled = not rlt_enabled
                rlt_toggle_on_next = rlt_enabled
                rlt_toggle_off_next = not rlt_enabled
                rospy.logwarn(f"[RLT] enabled={rlt_enabled}")
            elif cmd in {"s", "f", "x", "e", "n"}:
                outcome = {"s": "success", "f": "failure", "x": "unsafe", "e": "env_reset", "n": "neutral"}[cmd]
                ep_dir = recorder.finish_episode(outcome=outcome, note=f"manual_{outcome}")
                rospy.logwarn(f"[EPISODE] finished outcome={outcome}, saved={ep_dir}")
                if cmd == "e":
                    rospy.logwarn("[EPISODE] Environment reset marked. Move object, then continue.")
                continue
            elif cmd == "q":
                rospy.logwarn("[RUN] Quit requested.")
                break
            obs = observation_ball(current_prompt, rotate_wrist_180=args.rotate_wrist_180, convert_to_rgb=args.convert_to_rgb, resize_hw=args.resize_hw)
            if args.print_action_debug:
                rospy.loginfo(f"[OBS] mode={current_mode}, rlt={rlt_enabled}, prompt='{current_prompt}', state={_fmt_arr(obs['observation/state'])}")
            t_infer0 = time.time()
            result = policy.infer(obs)
            t_infer1 = time.time()
            if "actions" not in result or len(result["actions"]) < int(args.horizon):
                rospy.logwarn(f"Policy returned fewer than {args.horizon} actions, skipping.")
                continue
            pi_actions = np.asarray(result["actions"][: int(args.horizon)], dtype=np.float64)
            exec_actions, delta_actions, rlt_info = apply_rlt_or_exploration(obs, pi_actions, rlt_enabled, current_mode, adapter, args)
            if rlt_info.get("stopped_by_risk", False):
                recorder.add_chunk(EpisodeChunk(obs, pi_actions, pi_actions, np.zeros_like(pi_actions), current_mode, current_prompt, rlt_enabled, rlt_toggle_on_next, rlt_toggle_off_next, False, False, False, t_infer1 - t_infer0, 0.0, time.time(), "rlt_risk_stop"))
                rospy.logwarn(f"[RLT] Risk stop. risk={rlt_info['risk']:.3f}")
                rlt_toggle_on_next = False
                rlt_toggle_off_next = False
                continue
            if args.print_action_debug:
                rospy.loginfo(f"[ACTION] mode={current_mode}, rlt_src={rlt_info['source']}, delta_max={float(np.max(np.abs(delta_actions))):.4f}, first={_fmt_arr(exec_actions[0])}, last={_fmt_arr(exec_actions[-1])}, infer_dt={t_infer1 - t_infer0:.3f}s")
            t_exec0 = time.time()
            ok = plan_and_execute_horizon_with_arms_group(robot, left_fixed, exec_actions[:, :7].copy(), args.planning_time, args.waypoint_dt, args.vel_scale, args.acc_scale)
            t_exec1 = time.time()
            if ok:
                rg = float(exec_actions[-1, 7])
                if robot.gripper_r is not None:
                    if rg < 0:
                        robot.gripper_r.open()
                    else:
                        robot.gripper_r.close()
                count += int(args.horizon)
            else:
                rospy.logwarn("Chunk planning/execution failed.")
            recorder.add_chunk(EpisodeChunk(obs, pi_actions, exec_actions, delta_actions, current_mode, current_prompt, rlt_enabled, rlt_toggle_on_next, rlt_toggle_off_next, bool(ok), not bool(ok), False, t_infer1 - t_infer0, t_exec1 - t_exec0, time.time(), str(rlt_info.get("source", ""))))
            rlt_toggle_on_next = False
            rlt_toggle_off_next = False
            rospy.loginfo(f"[{current_mode}] Chunk recorded. steps~{count}/{args.num_steps}, rlt={rlt_enabled}, infer_dt={t_infer1 - t_infer0:.3f}s, exec_dt={t_exec1 - t_exec0:.3f}s")
    finally:
        cmd_watcher.stop()
        saved = recorder.finish_episode(outcome="aborted", note="script_exit")
        if saved is not None:
            rospy.logwarn(f"[EPISODE] final partial episode saved as aborted: {saved}")
    rospy.loginfo(f"Finished. executed_steps={count}, total_time={time.time() - start_time:.2f}s, mode={current_mode}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
