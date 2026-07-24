#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fully automated demo client for pi0/pi0.5 feeding/returning policy servers.

No RLT, no replay recording, no wrench logging.

Main feature:
  An external video stage model, e.g. VideoMAE, can publish the current task
  stage to a ROS String topic. This script automatically switches from FEED to
  RETURN when the configured return label is detected consistently. The return
  schedule keeps the previous demo behavior: the first N return chunks use the
  return channel, then the script automatically uses the feed/main channel while
  keeping the RETURN prompt.

Controls:
  The script can start in continuous execution with --auto-run. During run,
  press r/c/s/q directly, without Enter.

  Enter       : execute one policy chunk while paused
  run         : execute continuously; during run press r/c/s/q directly
  run N       : execute N chunks continuously, then pause again
  r           : manually switch to RETURN prompt; use return channel for first N return chunks
  c           : manually switch to FEED prompt and feed channel
  s           : while running, stop/pause continuous execution
  q           : quit
  ch NAME     : switch current mode to a specific policy channel
  feedch NAME : set FEED mode channel
  retch NAME  : set RETURN mode channel
  ls          : list policy channels
  stage       : print current video stage state
  q           : quit
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
import select
import termios
import tty
from typing import Dict, List, Tuple

import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from moveit_msgs.msg import RobotState, RobotTrajectory
from openpi_client import websocket_client_policy as _websocket_client_policy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tams_diana7_tools.dual_diana_helper import DualDianaHelper
from tams_diana7_tools.gripper_interfaces import GripperType
from trajectory_msgs.msg import JointTrajectoryPoint


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
    # Default/main server. It is also registered as channel name "main".
    host: str = "134.100.39.19"
    port: int = 8000

    # Extra policy channels, format:
    #   name=host:port,name2=host2:port2
    # Example:
    #   --channels feed=134.100.39.19:8000,return=134.100.39.19:8001,wrench=134.100.39.19:8002
    channels: str = ""
    feed_channel: str = "main"
    return_channel: str = "main"
    # After pressing r, use return_channel for this many chunks, then automatically use feed_channel.
    # This is useful when the first few return steps need a specialized return policy,
    # but later return can continue with the main/feed policy server.
    return_channel_chunks: int = 3

    # External video-stage automation, e.g. from a VideoMAE stage detector.
    # The detector should publish either a plain label such as "returning",
    # or a JSON string such as {"stage": "returning", "prob": 0.91}.
    auto_stage_switch: bool = True
    stage_topic: str = "/feeding_stage"
    stage_return_labels: str = "return,returning"
    stage_feed_labels: str = "feed,feeding,grasping"
    stage_prob_threshold: float = 0.75
    stage_min_consecutive: int = 3
    stage_stale_timeout: float = 1.5
    # Safer default for demos: video can automatically trigger RETURN, but
    # FEED reset is manual by pressing c. Enable this only if the stage detector
    # is very stable and you want fully cyclic operation.
    stage_allow_auto_feed: bool = False
    # Once RETURN starts, ignore further auto-stage switches until manual c,
    # unless stage_allow_auto_feed=True. This prevents oscillation.
    stage_latch_return: bool = True

    prompt: str = "pick up the spoon and feed the person"
    return_prompt: str = "return the spoon to the start position"

    num_steps: int = 2000
    horizon: int = 12
    waypoint_dt: float = 0.10
    vel_scale: float = 0.05
    acc_scale: float = 0.05
    planning_time: float = 0.5
    execute_skip_first_k: int = 0

    # Gripper execution:
    #   final      : old behavior, command gripper only after the whole chunk finishes.
    #   transition : command gripper during the chunk when action[:, 7] crosses the threshold.
    gripper_execution_mode: str = "final"
    gripper_close_threshold: float = 0.0

    rotate_wrist_180: bool = False
    convert_to_rgb: bool = True
    resize_hw: int = 224

    # Initial robot pose used by your previous feeding scripts.
    move_to_initial_pose: bool = True

    print_action_debug: bool = True

    # If True, start continuous execution immediately after initialization.
    # During continuous execution: r=RETURN schedule, c=FEED, s=pause, q=quit.
    auto_run: bool = True


def _fmt_arr(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x), precision=4, suppress_small=True, separator=", ")


class SingleKeyPoller:
    """Non-blocking single-key reader for live demo control.

    Works on a normal Linux terminal. Keys are read without pressing Enter while
    this context manager is active. If stdin is not a TTY, it simply returns None.
    """

    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        try:
            if sys.stdin.isatty():
                self.fd = sys.stdin.fileno()
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                self.enabled = True
        except Exception as exc:
            rospy.logwarn(f"[KEY] Could not enable single-key mode: {exc}")
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.enabled and self.fd is not None and self.old_settings is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        except Exception:
            pass
        self.enabled = False

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            ch = sys.stdin.read(1)
            if ch in ("\n", "\r"):
                return None
            return ch.lower()
        except Exception:
            return None


def parse_channels(args: Args) -> Dict[str, Tuple[str, int]]:
    out: Dict[str, Tuple[str, int]] = {"main": (args.host, int(args.port))}
    text = str(args.channels).strip()
    if not text:
        return out

    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(
                f"Bad channel spec '{item}'. Use name=host:port, e.g. feed=134.100.39.19:8000"
            )
        name, addr = item.split("=", 1)
        host, port_s = addr.rsplit(":", 1)
        name = name.strip()
        host = host.strip()
        if not name or not host:
            raise ValueError(f"Bad channel spec '{item}'.")
        out[name] = (host, int(port_s))
    return out


def observation_ball(
    prompt: str,
    timeout: float = 2.0,
    rotate_wrist_180: bool = True,
    convert_to_rgb: bool = True,
    resize_hw: int = 224,
) -> dict:
    latest = {
        "right_state_8": np.zeros((8,), dtype=np.float32),
        "got_right_arm": False,
        "image": None,
        "wrist_image": None,
    }
    bridge = CvBridge()

    def joint_state_callback(msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name) if i < len(msg.position)}

        if all(n in name_to_idx for n in ARM_R_NAMES):
            latest["right_state_8"][:7] = np.array(
                [msg.position[name_to_idx[n]] for n in ARM_R_NAMES], dtype=np.float32
            )
            latest["got_right_arm"] = True
        elif len(msg.position) >= 14:
            arm_r_names = [n for n in msg.name if "arm_r" in n]
            if len(arm_r_names) >= 7:
                try:
                    latest["right_state_8"][:7] = np.array(
                        [msg.position[name_to_idx[n]] for n in arm_r_names[:7]], dtype=np.float32
                    )
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
            rospy.logwarn(
                f"Observation timeout. got_img={got_img}, got_wrist={got_wrist}, got_right_arm={got_right}"
            )
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
    rospy.logwarn_throttle(
        2.0,
        "MoveIt joint order does not fully match ARM_L_NAMES/ARM_R_NAMES. "
        "Fallback to concatenate [left_fixed, right_target].",
    )
    return [float(x) for x in np.concatenate([left_fixed_7, right_target_7], axis=0)]


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



def _gripper_cmd_from_action(value: float, close_threshold: float = 0.0) -> str:
    """Map action dim 7 to a discrete gripper command."""
    return "close" if float(value) >= float(close_threshold) else "open"


def _send_gripper_command(robot: DualDianaHelper, command: str) -> None:
    if robot.gripper_r is None:
        return
    try:
        if command == "close":
            robot.gripper_r.close()
        else:
            robot.gripper_r.open()
    except Exception as exc:
        rospy.logwarn(f"[GRIPPER] Failed to send {command}: {exc}")


def _build_gripper_transition_events(
    gripper_values: np.ndarray,
    duration: float,
    close_threshold: float = 0.0,
) -> List[Tuple[float, str, int, float]]:
    """
    Return transition events [(time_from_start, command, waypoint_index, action_value), ...].

    We only schedule commands when action[:, 7] changes side of the open/close threshold.
    This avoids repeatedly sending close/open at every waypoint.
    """
    values = np.asarray(gripper_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []

    prev_cmd = _gripper_cmd_from_action(values[0], close_threshold)
    events: List[Tuple[float, str, int, float]] = []
    n = int(values.size)
    duration = max(0.0, float(duration))
    for i in range(1, n):
        cmd = _gripper_cmd_from_action(values[i], close_threshold)
        if cmd != prev_cmd:
            # Map waypoint i to the same relative timing as the executed arm trajectory.
            t_event = duration * float(i + 1) / float(n)
            events.append((t_event, cmd, i, float(values[i])))
            prev_cmd = cmd
    return events

def plan_and_execute_horizon_with_arms_group(
    robot: DualDianaHelper,
    left_fixed_7: np.ndarray,
    right_waypoints_7: np.ndarray,
    planning_time: float,
    waypoint_dt: float,
    vel_scale: float,
    acc_scale: float,
    gripper_values: np.ndarray | None = None,
    gripper_execution_mode: str = "final",
    gripper_close_threshold: float = 0.0,
) -> bool:
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

    mode = str(gripper_execution_mode).lower().strip()
    if mode != "transition" or gripper_values is None:
        ok = group.execute(full, wait=True)
        group.stop()
        group.clear_pose_targets()
        return bool(ok)

    duration = 0.0
    if full.joint_trajectory.points:
        duration = float(full.joint_trajectory.points[-1].time_from_start.to_sec())
    events = _build_gripper_transition_events(
        gripper_values=np.asarray(gripper_values, dtype=np.float64),
        duration=duration,
        close_threshold=float(gripper_close_threshold),
    )

    rospy.loginfo(
        f"[GRIPPER] transition mode: {len(events)} event(s), "
        f"duration={duration:.3f}s, threshold={float(gripper_close_threshold):.3f}"
    )
    ok = group.execute(full, wait=False)
    if not ok:
        group.stop()
        group.clear_pose_targets()
        return False

    start = time.time()
    event_idx = 0
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        elapsed = time.time() - start
        while event_idx < len(events) and elapsed >= events[event_idx][0]:
            t_event, cmd, wp_idx, value = events[event_idx]
            rospy.loginfo(
                f"[GRIPPER] mid-chunk event t={t_event:.3f}s, "
                f"waypoint={wp_idx}, action={value:.4f}, command={cmd}"
            )
            _send_gripper_command(robot, cmd)
            event_idx += 1
        if elapsed >= duration + 0.20:
            break
        rate.sleep()

    group.stop()
    group.clear_pose_targets()
    return True


def trim_actions(actions: np.ndarray, skip_first_k: int) -> Tuple[np.ndarray, int]:
    actions = np.asarray(actions, dtype=np.float64)
    k = max(0, int(skip_first_k))
    n = int(actions.shape[0])
    if k <= 0:
        return actions.copy(), 0
    if k >= n:
        k = max(0, n - 1)
    return actions[k:].copy(), k


def connect_policies(channels: Dict[str, Tuple[str, int]]):
    policies = {}
    for name, (host, port) in channels.items():
        rospy.logwarn(f"[POLICY] Connecting channel '{name}' at {host}:{port}")
        policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
        try:
            logging.info(f"[POLICY] {name} metadata: {policy.get_server_metadata()}")
        except Exception as exc:
            rospy.logwarn(f"[POLICY] Could not fetch metadata for {name}: {exc}")
        policies[name] = policy
    return policies


def execute_one_chunk(
    *,
    policy,
    channel_name: str,
    current_mode: str,
    current_prompt: str,
    robot: DualDianaHelper,
    left_fixed: np.ndarray,
    args: Args,
    count: int,
) -> int:
    obs = observation_ball(
        current_prompt,
        rotate_wrist_180=args.rotate_wrist_180,
        convert_to_rgb=args.convert_to_rgb,
        resize_hw=args.resize_hw,
    )
    if args.print_action_debug:
        rospy.loginfo(
            f"[OBS] mode={current_mode}, channel={channel_name}, prompt='{current_prompt}', "
            f"state={_fmt_arr(obs['observation/state'])}"
        )

    t0 = time.time()
    result = policy.infer(obs)
    t1 = time.time()
    if "actions" not in result or len(result["actions"]) < int(args.horizon):
        rospy.logwarn(f"Policy returned fewer than {args.horizon} actions, skipping.")
        return count

    actions = np.asarray(result["actions"][: int(args.horizon)], dtype=np.float64)
    exec_actions, skipped = trim_actions(actions, args.execute_skip_first_k)
    if skipped > 0:
        rospy.logwarn(f"[CHUNK-TRIM] Skipping first {skipped} waypoint(s). execute_len={len(exec_actions)}/{len(actions)}")

    if args.print_action_debug:
        rospy.loginfo(
            f"[ACTION] mode={current_mode}, channel={channel_name}, len={len(exec_actions)}, "
            f"first={_fmt_arr(exec_actions[0])}, last={_fmt_arr(exec_actions[-1])}, infer_dt={t1-t0:.3f}s"
        )

    tex0 = time.time()
    ok = plan_and_execute_horizon_with_arms_group(
        robot=robot,
        left_fixed_7=left_fixed,
        right_waypoints_7=exec_actions[:, :7].copy(),
        planning_time=args.planning_time,
        waypoint_dt=args.waypoint_dt,
        vel_scale=args.vel_scale,
        acc_scale=args.acc_scale,
        gripper_values=exec_actions[:, 7].copy() if exec_actions.shape[1] >= 8 else None,
        gripper_execution_mode=args.gripper_execution_mode,
        gripper_close_threshold=args.gripper_close_threshold,
    )
    tex1 = time.time()

    if ok:
        rg = float(exec_actions[-1, 7])
        final_cmd = _gripper_cmd_from_action(rg, args.gripper_close_threshold)
        if str(args.gripper_execution_mode).lower().strip() == "transition":
            rospy.loginfo(
                f"[GRIPPER] final_action={rg:.4f}, final_cmd={final_cmd}, "
                "sending final safeguard after transition mode"
            )
        else:
            rospy.loginfo(f"[GRIPPER] final_action={rg:.4f}, command={final_cmd}")
        _send_gripper_command(robot, final_cmd)
        count += int(exec_actions.shape[0])
        rospy.loginfo(
            f"[{current_mode}] executed channel={channel_name}, steps~{count}/{args.num_steps}, "
            f"infer_dt={t1-t0:.3f}s, exec_dt={tex1-tex0:.3f}s"
        )
    else:
        rospy.logwarn(f"[{current_mode}] chunk planning/execution failed. channel={channel_name}")
    return count


def _parse_label_set(text: str) -> set[str]:
    return {x.strip().lower() for x in str(text).split(",") if x.strip()}


def _parse_stage_string(text: str) -> tuple[str, float | None, dict]:
    """Parse a stage detector String message.

    Supported payloads:
      - "returning"
      - "returning 0.91"
      - "stage=returning prob=0.91"
      - '{"stage": "returning", "prob": 0.91}'

    Returns (stage_label, probability_or_None, raw_dict).
    """
    raw = str(text).strip()
    if not raw:
        return "", None, {}

    # JSON is preferred for probability-aware switching.
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            stage = str(
                data.get("stage")
                or data.get("label")
                or data.get("class")
                or data.get("state")
                or ""
            ).strip().lower()
            prob = data.get("prob", data.get("probability", data.get("score", data.get("confidence"))))
            prob_f = None if prob is None else float(prob)
            return stage, prob_f, data
        except Exception:
            pass

    tokens = raw.replace(",", " ").split()
    stage = tokens[0].strip().lower() if tokens else raw.lower()
    prob_f = None
    kv = {}
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            kv[key.strip().lower()] = value.strip()
    if "stage" in kv:
        stage = kv["stage"].lower()
    elif "label" in kv:
        stage = kv["label"].lower()
    for key in ("prob", "probability", "score", "confidence"):
        if key in kv:
            try:
                prob_f = float(kv[key])
            except Exception:
                prob_f = None
            break
    if prob_f is None and len(tokens) >= 2:
        try:
            prob_f = float(tokens[1])
        except Exception:
            prob_f = None
    return stage, prob_f, kv


def main(args: Args) -> None:
    rospy.init_node("openpi_policy_client_demo_channels", anonymous=True)

    channels = parse_channels(args)
    if args.feed_channel not in channels:
        raise ValueError(f"feed_channel='{args.feed_channel}' not in channels={list(channels)}")
    if args.return_channel not in channels:
        raise ValueError(f"return_channel='{args.return_channel}' not in channels={list(channels)}")

    policies = connect_policies(channels)
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

    joint_goal_r = np.deg2rad([-32, 5, -18, 95, -8, -85, -1])
    joint_goal_l = np.deg2rad([-32, -23, -49, 97, 11, -98, -1])
    if bool(args.move_to_initial_pose):
        joint_goal = np.concatenate([
            joint_goal_l,
            joint_goal_r,
            np.array([0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ])
        robot.move_to_joint_goal(joint_goal, use_moveit=True)
    left_fixed = np.array(joint_goal_l, dtype=np.float64)

    arms_joint_names = _get_arms_joint_names(robot.arms_group)
    rospy.loginfo("MoveIt arms_group joint order:")
    for i, joint_name in enumerate(arms_joint_names):
        rospy.loginfo(f"  [{i}] {joint_name}")

    current_mode = "FEED"
    current_prompt = args.prompt
    current_channel = args.feed_channel
    return_channel_remaining = 0

    return_labels = _parse_label_set(args.stage_return_labels)
    feed_labels = _parse_label_set(args.stage_feed_labels)
    stage_state = {
        "label": "",
        "prob": None,
        "raw": "",
        "stamp": 0.0,
        "return_streak": 0,
        "feed_streak": 0,
        "auto_return_latched": False,
    }

    rospy.loginfo("=== FULLY AUTOMATED DEMO CHANNEL VERSION: no RLT, no replay, no wrench log ===")
    rospy.loginfo(f"Gripper mode={args.gripper_execution_mode}, close_threshold={args.gripper_close_threshold}")
    rospy.loginfo(
        "Controls: Enter=one chunk, run=continuous, run N=N chunks, "
        "while running: r=RETURN schedule, c=FEED, s=pause, q=quit. "
        "Paused controls: ch NAME=switch channel, feedch NAME=set FEED channel, "
        "retch NAME=set RETURN channel, ls=list channels, stage=print stage, q=quit."
    )
    rospy.loginfo(f"Channels: {channels}")
    rospy.loginfo(
        f"feed_channel={args.feed_channel}, return_channel={args.return_channel}, "
        f"return_channel_chunks={args.return_channel_chunks}"
    )
    rospy.loginfo(
        f"auto_stage_switch={args.auto_stage_switch}, stage_topic='{args.stage_topic}', "
        f"return_labels={sorted(return_labels)}, feed_labels={sorted(feed_labels)}, "
        f"prob_threshold={args.stage_prob_threshold}, min_consecutive={args.stage_min_consecutive}, "
        f"allow_auto_feed={args.stage_allow_auto_feed}, latch_return={args.stage_latch_return}"
    )

    # Warm-up current policy.
    policies[current_channel].infer(
        observation_ball(
            current_prompt,
            rotate_wrist_180=args.rotate_wrist_180,
            convert_to_rgb=args.convert_to_rgb,
            resize_hw=args.resize_hw,
        )
    )

    count = 0
    start_time = time.time()
    quit_requested = False

    def switch_to_return():
        nonlocal current_mode, current_prompt, current_channel, return_channel_remaining
        current_mode = "RETURN"
        current_prompt = args.return_prompt
        return_channel_remaining = max(0, int(args.return_channel_chunks))
        current_channel = args.return_channel if return_channel_remaining > 0 else args.feed_channel
        rospy.logwarn(
            f"[MODE] RETURN, prompt='{current_prompt}', channel='{current_channel}', "
            f"return_channel_remaining={return_channel_remaining}. "
            f"After that it will use feed_channel='{args.feed_channel}'."
        )

    def switch_to_feed():
        nonlocal current_mode, current_prompt, current_channel, return_channel_remaining
        current_mode = "FEED"
        current_prompt = args.prompt
        current_channel = args.feed_channel
        return_channel_remaining = 0
        stage_state["auto_return_latched"] = False
        stage_state["return_streak"] = 0
        stage_state["feed_streak"] = 0
        rospy.logwarn(f"[MODE] FEED, prompt='{current_prompt}', channel='{current_channel}'")

    def stage_cb(msg: String):
        label, prob, _ = _parse_stage_string(msg.data)
        now = time.time()
        stage_state["label"] = label
        stage_state["prob"] = prob
        stage_state["raw"] = str(msg.data)
        stage_state["stamp"] = now

        confident = True if prob is None else (float(prob) >= float(args.stage_prob_threshold))
        if confident and label in return_labels:
            stage_state["return_streak"] += 1
            stage_state["feed_streak"] = 0
        elif confident and label in feed_labels:
            stage_state["feed_streak"] += 1
            stage_state["return_streak"] = 0
        elif label:
            stage_state["return_streak"] = 0
            stage_state["feed_streak"] = 0

        rospy.loginfo_throttle(
            0.5,
            f"[STAGE] label={label}, prob={prob}, "
            f"return_streak={stage_state['return_streak']}, feed_streak={stage_state['feed_streak']}"
        )

    if bool(args.auto_stage_switch):
        rospy.Subscriber(args.stage_topic, String, stage_cb, queue_size=1)
        rospy.logwarn(f"[STAGE] Auto stage switch enabled. Listening on {args.stage_topic}")

    def maybe_auto_switch_from_stage():
        # Run only at chunk boundaries. This avoids interrupting MoveIt mid-trajectory.
        now = time.time()
        stamp = float(stage_state.get("stamp") or 0.0)
        if not bool(args.auto_stage_switch):
            return
        if stamp <= 0.0:
            return
        if float(args.stage_stale_timeout) > 0 and (now - stamp) > float(args.stage_stale_timeout):
            rospy.logwarn_throttle(2.0, f"[STAGE] Stage message stale: age={now - stamp:.2f}s")
            return

        min_k = max(1, int(args.stage_min_consecutive))
        if current_mode != "RETURN" and int(stage_state["return_streak"]) >= min_k:
            rospy.logwarn(
                f"[STAGE-AUTO] Trigger RETURN from video stage: raw='{stage_state['raw']}', "
                f"streak={stage_state['return_streak']}"
            )
            switch_to_return()
            stage_state["auto_return_latched"] = True
            stage_state["return_streak"] = 0
            stage_state["feed_streak"] = 0
            return

        if (
            bool(args.stage_allow_auto_feed)
            and current_mode != "FEED"
            and int(stage_state["feed_streak"]) >= min_k
            and (not bool(args.stage_latch_return) or not bool(stage_state["auto_return_latched"]))
        ):
            rospy.logwarn(
                f"[STAGE-AUTO] Trigger FEED from video stage: raw='{stage_state['raw']}', "
                f"streak={stage_state['feed_streak']}"
            )
            switch_to_feed()
            stage_state["return_streak"] = 0
            stage_state["feed_streak"] = 0

    def update_return_channel_schedule():
        nonlocal current_channel
        if current_mode == "RETURN":
            scheduled_channel = args.return_channel if return_channel_remaining > 0 else args.feed_channel
            if scheduled_channel != current_channel:
                rospy.logwarn(f"[CHANNEL-AUTO] RETURN switched channel {current_channel} -> {scheduled_channel}")
                current_channel = scheduled_channel

    def after_chunk_bookkeeping():
        nonlocal current_channel, return_channel_remaining
        if current_mode == "RETURN" and return_channel_remaining > 0:
            return_channel_remaining -= 1
            if return_channel_remaining == 0:
                current_channel = args.feed_channel
                rospy.logwarn(
                    f"[CHANNEL-AUTO] Finished first {args.return_channel_chunks} RETURN chunk(s) "
                    f"on return_channel. Now using feed_channel='{current_channel}' with RETURN prompt."
                )

    def run_continuous(max_chunks: int | None = None) -> None:
        nonlocal count, quit_requested
        executed_chunks = 0
        rospy.logwarn(
            "[RUN] Continuous execution started. Live keys: "
            "r=RETURN schedule, c=FEED, s=pause, q=quit. "
            "Keys are read directly; no Enter needed."
        )
        with SingleKeyPoller() as keys:
            while not rospy.is_shutdown() and count < int(args.num_steps):
                if max_chunks is not None and executed_chunks >= int(max_chunks):
                    rospy.logwarn(f"[RUN] Finished requested {max_chunks} chunk(s), pausing.")
                    return

                # Process any key pressed since the last chunk.
                ch = keys.poll()
                if ch == "r":
                    switch_to_return()
                elif ch == "c":
                    switch_to_feed()
                elif ch in ("s", " "):
                    rospy.logwarn("[RUN] Pause requested by key.")
                    return
                elif ch == "q":
                    rospy.logwarn("[RUN] Quit requested by key.")
                    quit_requested = True
                    return
                elif ch is not None:
                    rospy.logwarn("[KEY] Live keys: r=RETURN, c=FEED, s/space=pause, q=quit")

                maybe_auto_switch_from_stage()
                update_return_channel_schedule()
                count = execute_one_chunk(
                    policy=policies[current_channel],
                    channel_name=current_channel,
                    current_mode=current_mode,
                    current_prompt=current_prompt,
                    robot=robot,
                    left_fixed=left_fixed,
                    args=args,
                    count=count,
                )
                executed_chunks += 1
                after_chunk_bookkeeping()

    if bool(args.auto_run):
        try:
            run_continuous(max_chunks=None)
        except KeyboardInterrupt:
            rospy.logwarn("[RUN] Interrupted by Ctrl+C.")
        if quit_requested:
            rospy.logwarn("[RUN] Quit requested after auto-run.")

    while (not quit_requested) and (not rospy.is_shutdown()) and count < int(args.num_steps):
        cmd = input(
            f"\n[demo paused | mode={current_mode} | channel={current_channel} | "
            f"feedch={args.feed_channel} | retch={args.return_channel} | "
            f"ret_remaining={return_channel_remaining} | "
            f"steps={count}/{args.num_steps}]\n"
            "Enter=one chunk | run/run N=continuous | r=RETURN | c=FEED | "
            "ch NAME=switch model | ls=list | stage=stage status | q=quit > "
        ).strip()
        cmd_l = cmd.lower()

        chunks_to_run = 0
        run_forever = False

        if cmd_l == "":
            chunks_to_run = 1
        elif cmd_l == "q":
            rospy.logwarn("[RUN] Quit requested.")
            break
        elif cmd_l == "r":
            switch_to_return()
            continue
        elif cmd_l == "c":
            switch_to_feed()
            continue
        elif cmd_l == "ls":
            for name, (host, port) in channels.items():
                marker = "*" if name == current_channel else " "
                rospy.logwarn(f"{marker} {name}: {host}:{port}")
            continue
        elif cmd_l == "stage":
            age = time.time() - float(stage_state.get("stamp") or 0.0) if stage_state.get("stamp") else -1.0
            rospy.logwarn(
                f"[STAGE] label={stage_state['label']}, prob={stage_state['prob']}, "
                f"age={age:.2f}s, raw='{stage_state['raw']}', "
                f"return_streak={stage_state['return_streak']}, feed_streak={stage_state['feed_streak']}, "
                f"latched={stage_state['auto_return_latched']}"
            )
            continue
        elif cmd_l.startswith("ch "):
            name = cmd.split(maxsplit=1)[1].strip()
            if name not in policies:
                rospy.logwarn(f"[CHANNEL] Unknown channel '{name}'. Available={list(policies)}")
                continue
            current_channel = name
            rospy.logwarn(f"[CHANNEL] Current mode={current_mode} now uses channel='{current_channel}'")
            continue
        elif cmd_l.startswith("feedch "):
            name = cmd.split(maxsplit=1)[1].strip()
            if name not in policies:
                rospy.logwarn(f"[CHANNEL] Unknown channel '{name}'. Available={list(policies)}")
                continue
            args.feed_channel = name
            if current_mode == "FEED":
                current_channel = name
            rospy.logwarn(f"[CHANNEL] FEED channel set to '{name}'")
            continue
        elif cmd_l.startswith("retch ") or cmd_l.startswith("returnch "):
            name = cmd.split(maxsplit=1)[1].strip()
            if name not in policies:
                rospy.logwarn(f"[CHANNEL] Unknown channel '{name}'. Available={list(policies)}")
                continue
            args.return_channel = name
            if current_mode == "RETURN" and return_channel_remaining > 0:
                current_channel = name
            rospy.logwarn(f"[CHANNEL] RETURN channel set to '{name}'")
            continue
        elif cmd_l == "run":
            run_forever = True
        elif cmd_l.startswith("run "):
            try:
                chunks_to_run = max(1, int(cmd_l.split(maxsplit=1)[1]))
            except Exception:
                rospy.logwarn("[CMD] Use 'run' or 'run N', e.g. run 5")
                continue
        else:
            rospy.logwarn(f"[CMD] Unknown command '{cmd}'.")
            continue

        try:
            if run_forever:
                run_continuous(max_chunks=None)
                if quit_requested:
                    break
            else:
                # For Enter or run N, still execute the same scheduling logic.
                run_continuous(max_chunks=chunks_to_run)
                if quit_requested:
                    break
        except KeyboardInterrupt:
            rospy.logwarn("[RUN] Interrupted by Ctrl+C.")
            break

    rospy.loginfo(
        f"Finished. executed_steps={count}, total_time={time.time() - start_time:.2f}s, "
        f"mode={current_mode}, channel={current_channel}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
