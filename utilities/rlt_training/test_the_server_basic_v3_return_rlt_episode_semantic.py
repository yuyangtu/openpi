#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-prompt pi0.5 client + episode-level RLT replay recorder
with pause-before-each-chunk operation and manual end-effector
translation exploration.

This version fixes the Jacobian issue:
- Execution still uses the dual-arm `robot.arms_group`.
- Jacobian for EE translation exploration uses a right-arm chain MoveIt group,
  because the dual-arm group `arms` is not a kinematic chain.

Controls at pause prompt:
  Enter : execute one chunk
  r     : switch to RETURN prompt
  c     : switch to FEED prompt
  t     : toggle residual / EE offset on/off
  i/k   : EE x +/-
  j/l   : EE y +/-
  u/o   : EE z +/-
  z     : reset EE offset to zero
  p     : print current EE offset
  m     : start/finish a recording episode. First m starts a new episode;
          second m finishes it and stops recording; third m starts another one.
  g     : force gripper CLOSE for the next executed chunk only
  b     : force gripper OPEN for the next executed chunk only
  a     : clear gripper override
  s     : finish current episode as success
  f     : finish current episode as failure
  x     : finish current episode as unsafe/collision
  e     : finish current episode as env_reset
  n     : finish current episode as neutral
  q     : quit
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
from tams_diana7_tools.dual_diana_helper import DualDianaHelper
from tams_diana7_tools.gripper_interfaces import GripperType
from trajectory_msgs.msg import JointTrajectoryPoint

from rlt_training.episode_replay_buffer import EpisodeChunk, EpisodeReplayRecorder


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

    # Replay recording control. Default False is useful for feeding-phase-only
    # data collection: execute approach/grasp without saving, then press m to
    # start saving chunks from the feeding stage.
    # Press m again to finish the current recording episode and stop recording.
    # Press m again to start a new recording episode.
    record_from_start: bool = False
    record_stop_outcome: str = "neutral"  # outcome used when ending an episode with m

    # Optional trained RLT checkpoint. If empty, manual EE exploration is used when t is enabled.
    rlt_checkpoint: str = ""
    rlt_device: str = "cuda"
    rlt_max_delta: float = 0.03
    rlt_risk_stop_threshold: float = 0.95
    rlt_on_return: bool = False

    # Manual Cartesian EE offset exploration.
    use_manual_ee_offset: bool = True
    right_arm_group_name: str = ""    # set explicitly if auto-detection fails, e.g. "arm_r"
    ee_step: float = 0.003            # 3 mm per key press
    ee_max_offset: float = 0.020      # max absolute xyz offset, 2 cm
    ee_apply_last_k: int = 4          # apply offset only to last K waypoints of the chunk
    ee_joint_delta_max: float = 0.025 # max joint residual per waypoint, radians
    ee_damping: float = 1e-3          # damped least-squares regularization

    # Manual gripper override for human correction.
    # In your previous execution code: gripper action < 0 => open, >= 0 => close.
    gripper_close_value: float = 1.0
    gripper_open_value: float = -1.0

    # Execution mode for correction when residual/RLT is enabled.
    # residual_vla: old behavior, correction is applied on top of VLA chunk.
    # direct_ee:    use semantic ee_offset to generate a short EE primitive from current state.
    correction_execution_mode: str = "residual_vla"
    direct_correction_steps: int = 6
    direct_min_ee_norm: float = 1e-5
    direct_use_vla_gripper_when_no_override: bool = True

    # Skip the first K waypoints of non-direct VLA chunks. Useful if the first
    # few policy actions have a small backward/recovery artifact.
    execute_skip_first_k: int = 0

    # Gate monitor / optional auto application.
    # log_rlt_gate: print model's gate probability every chunk when checkpoint is loaded.
    # auto_rlt_gate: if gate_prob >= threshold, apply RLT/direct_ee even if manual t is off.
    log_rlt_gate: bool = True
    auto_rlt_gate: bool = False
    auto_rlt_threshold: float = 0.85

    # Old joint-noise fallback. Keep disabled unless explicitly needed.
    explore_delta_std: float = 0.0
    explore_delta_max: float = 0.02

    rotate_wrist_180: bool = False
    convert_to_rgb: bool = True
    resize_hw: int = 224

    horizon: int = 12
    waypoint_dt: float = 0.10
    vel_scale: float = 0.05
    acc_scale: float = 0.05
    planning_time: float = 0.5

    print_action_debug: bool = True


def _fmt_arr(x: np.ndarray) -> str:
    return np.array2string(
        np.asarray(x),
        precision=4,
        suppress_small=True,
        separator=", ",
    )


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
                [msg.position[name_to_idx[n]] for n in ARM_R_NAMES],
                dtype=np.float32,
            )
            latest["got_right_arm"] = True

        elif len(msg.position) >= 14:
            arm_r_names = [n for n in msg.name if "arm_r" in n]
            if len(arm_r_names) >= 7:
                try:
                    latest["right_state_8"][:7] = np.array(
                        [msg.position[name_to_idx[n]] for n in arm_r_names[:7]],
                        dtype=np.float32,
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
                f"Observation timeout. got_img={got_img}, "
                f"got_wrist={got_wrist}, got_right_arm={got_right}"
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


def _make_target14_by_name(
    joint_names: List[str],
    left_fixed_7: np.ndarray,
    right_target_7: np.ndarray,
) -> List[float]:
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
            new_point.time_from_start = rospy.Duration.from_sec(
                t_offset + point.time_from_start.to_sec()
            )
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

        group.set_joint_value_target({
            joint_name: float(pos)
            for joint_name, pos in zip(joint_names, target14)
        })

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

        cur_start_14 = np.array(
            seg_trajs[-1].joint_trajectory.points[-1].positions,
            dtype=np.float64,
        )

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


def get_right_arm_chain_group(robot: DualDianaHelper, right_arm_group_name: str = ""):
    """Find a single-chain MoveIt group for right-arm Jacobian.

    The dual-arm group `arms` is not a chain and cannot compute Jacobian.
    This function returns a right-arm chain group if available.
    """
    # 1) User-specified group name.
    if right_arm_group_name:
        try:
            import moveit_commander
            group = moveit_commander.MoveGroupCommander(right_arm_group_name)
            joints = group.get_active_joints()
            rospy.logwarn(f"[JACOBIAN] Using explicit right arm group '{right_arm_group_name}', joints={joints}")
            return group
        except Exception as exc:
            rospy.logwarn(f"[JACOBIAN] Explicit right arm group '{right_arm_group_name}' failed: {exc}")

    # 2) Attributes exposed by DualDianaHelper.
    candidate_attrs = [
        "arm_r_group",
        "right_arm_group",
        "r_arm_group",
        "arm_r_move_group",
        "right_group",
        "move_group_r",
        "group_r",
    ]
    for attr in candidate_attrs:
        if hasattr(robot, attr):
            group = getattr(robot, attr)
            if group is not None:
                try:
                    joints = group.get_active_joints()
                    rospy.logwarn(f"[JACOBIAN] Using robot.{attr} for right-arm Jacobian, joints={joints}")
                    return group
                except Exception:
                    pass

    # 3) Try common MoveIt group names.
    try:
        import moveit_commander

        try:
            all_names = robot.robot.get_group_names()
            rospy.logwarn(f"[JACOBIAN] Available MoveIt groups: {all_names}")
        except Exception as exc:
            all_names = []
            rospy.logwarn(f"[JACOBIAN] Could not get MoveIt group names: {exc}")

        candidate_group_names = [
            "arm_r",
            "right_arm",
            "right_arm_group",
            "diana_r",
            "diana_R",
            "arm_right",
            "right",
            "manipulator_r",
            "r_arm",
        ]

        # Prefer actual group names that look right-arm-like.
        for name in all_names:
            low = name.lower()
            if ("right" in low) or ("arm_r" in low) or low.endswith("_r") or low == "r_arm":
                if name not in candidate_group_names:
                    candidate_group_names.insert(0, name)

        tried = set()
        for name in candidate_group_names:
            if name in tried:
                continue
            tried.add(name)
            try:
                group = moveit_commander.MoveGroupCommander(name)
                joints = group.get_active_joints()
                right_count = len([j for j in joints if j in ARM_R_NAMES or "arm_r" in j])
                if len(joints) == 7 or right_count >= 7:
                    rospy.logwarn(f"[JACOBIAN] Using MoveGroupCommander('{name}') for right-arm Jacobian, joints={joints}")
                    return group
                rospy.logwarn(f"[JACOBIAN] Candidate group '{name}' rejected, joints={joints}")
            except Exception:
                continue

    except Exception as exc:
        rospy.logwarn(f"[JACOBIAN] Could not import/use moveit_commander: {exc}")

    rospy.logerr(
        "[JACOBIAN] Could not find a right-arm chain MoveIt group. "
        "Manual EE offset will be disabled. Run with --right-arm-group-name <NAME> if you know it."
    )
    return None


def map_group_dq_to_action_order(group_joint_names: list[str], dq_group_order: np.ndarray) -> np.ndarray:
    """Map dq from right-arm MoveIt group joint order into policy ARM_R_NAMES action order."""
    dq_group_order = np.asarray(dq_group_order, dtype=np.float64).reshape(-1)
    out = np.zeros((7,), dtype=np.float64)

    if len(group_joint_names) != len(dq_group_order):
        rospy.logwarn(f"[JACOBIAN] group_joint_names/dq length mismatch: {len(group_joint_names)} vs {len(dq_group_order)}")
        n = min(7, len(dq_group_order))
        out[:n] = dq_group_order[:n]
        return out

    if all(name in ARM_R_NAMES for name in group_joint_names):
        for name, value in zip(group_joint_names, dq_group_order):
            out[ARM_R_NAMES.index(name)] = float(value)
        return out

    # Fallback: if exactly 7 joints but names are different, assume same order as policy.
    if len(dq_group_order) == 7:
        rospy.logwarn_throttle(
            2.0,
            f"[JACOBIAN] Right group joint names do not match ARM_R_NAMES. "
            f"Assuming same order. group_joint_names={group_joint_names}",
        )
        return dq_group_order.copy()

    n = min(7, len(dq_group_order))
    out[:n] = dq_group_order[:n]
    return out


def apply_manual_ee_translation_offset(
    jacobian_group,
    right_waypoints_7: np.ndarray,
    ee_offset_xyz: np.ndarray,
    args: Args,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a Cartesian translation offset to the last K waypoints via Jacobian DLS.

    The offset is interpreted in the planning/base frame of the right-arm MoveIt group.
    Only translation is corrected; orientation is not directly constrained.
    """
    corrected_right = np.asarray(right_waypoints_7, dtype=np.float64).copy()
    delta_actions = np.zeros((right_waypoints_7.shape[0], 8), dtype=np.float64)

    offset = np.asarray(ee_offset_xyz, dtype=np.float64).reshape(3)
    if np.max(np.abs(offset)) < 1e-9:
        return corrected_right, delta_actions, {
            "source": "manual_ee_offset_zero",
            "ee_offset": offset.copy(),
            "max_joint_delta": 0.0,
        }

    if jacobian_group is None:
        rospy.logwarn_throttle(1.0, "[JACOBIAN] No right-arm chain group. Manual EE offset disabled for this chunk.")
        return corrected_right, delta_actions, {
            "source": "manual_ee_offset_failed_no_right_chain_group",
            "ee_offset": offset.copy(),
            "max_joint_delta": 0.0,
        }

    if not hasattr(jacobian_group, "get_jacobian_matrix"):
        rospy.logwarn("Right-arm group has no get_jacobian_matrix(). Manual EE offset disabled.")
        return corrected_right, delta_actions, {
            "source": "manual_ee_offset_failed_no_jacobian",
            "ee_offset": offset.copy(),
            "max_joint_delta": 0.0,
        }

    group_joint_names = list(jacobian_group.get_active_joints())
    rospy.logwarn_throttle(2.0, f"[JACOBIAN] right-arm chain joint_names={group_joint_names}")

    T = int(corrected_right.shape[0])
    last_k = max(1, min(int(args.ee_apply_last_k), T))
    start = T - last_k
    damping = float(args.ee_damping)
    max_dq = float(args.ee_joint_delta_max)

    max_abs_dq = 0.0
    any_success = False

    for t in range(start, T):
        # Smooth ramp: earlier waypoints receive smaller correction.
        alpha = float(t - start + 1) / float(last_k)
        dx = alpha * offset

        # Jacobian group is single right-arm chain, so it expects 7 joint values.
        right_target_7_action_order = np.asarray(corrected_right[t], dtype=np.float64).reshape(7)

        # Map policy action order into jacobian group order if possible.
        if all(name in ARM_R_NAMES for name in group_joint_names) and len(group_joint_names) == 7:
            right_target_group_order = [
                float(right_target_7_action_order[ARM_R_NAMES.index(name)])
                for name in group_joint_names
            ]
        else:
            # Fallback: assume group order equals action order.
            right_target_group_order = [float(x) for x in right_target_7_action_order]

        try:
            J = np.asarray(jacobian_group.get_jacobian_matrix(right_target_group_order), dtype=np.float64)
        except Exception as exc:
            rospy.logwarn(f"get_jacobian_matrix failed at waypoint {t}: {exc}")
            continue

        if J.ndim != 2 or J.shape[0] < 3 or J.shape[1] < 7:
            rospy.logwarn(f"Unexpected Jacobian shape {J.shape}; expected at least (3, 7).")
            continue

        J_pos = J[:3, :7]

        # Damped least squares:
        # dq = J^T (J J^T + λI)^-1 dx
        A = J_pos @ J_pos.T + damping * np.eye(3)

        try:
            dq_group_order = J_pos.T @ np.linalg.solve(A, dx)
        except np.linalg.LinAlgError:
            dq_group_order = J_pos.T @ np.linalg.pinv(A) @ dx

        dq_group_order = np.clip(dq_group_order, -max_dq, max_dq)
        dq_action_order = map_group_dq_to_action_order(group_joint_names, dq_group_order)

        corrected_right[t, :7] += dq_action_order
        delta_actions[t, :7] = dq_action_order
        max_abs_dq = max(max_abs_dq, float(np.max(np.abs(dq_action_order))))
        any_success = True

    if not any_success:
        return corrected_right, delta_actions, {
            "source": "manual_ee_offset_failed_all_jacobians",
            "ee_offset": offset.copy(),
            "max_joint_delta": 0.0,
        }

    return corrected_right, delta_actions, {
        "source": "manual_ee_offset",
        "ee_offset": offset.copy(),
        "max_joint_delta": max_abs_dq,
    }



def apply_gripper_override(
    pi_actions: np.ndarray,
    exec_actions: np.ndarray,
    delta_actions: np.ndarray,
    gripper_override: str,
    args: Args,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Force the gripper channel for the next chunk.

    Convention used by the existing executor:
        action[:, 7] < 0  -> open gripper
        action[:, 7] >= 0 -> close gripper

    We set the whole chunk gripper channel, not only the final waypoint, so that
    the saved residual contains a clear human correction signal.
    """
    info = {
        "applied": False,
        "mode": gripper_override,
        "value": 0.0,
        "delta_max": 0.0,
    }

    if gripper_override not in {"close", "open"}:
        return exec_actions, delta_actions, info

    forced_value = float(args.gripper_close_value if gripper_override == "close" else args.gripper_open_value)

    exec_actions = np.asarray(exec_actions, dtype=np.float64).copy()
    delta_actions = np.asarray(delta_actions, dtype=np.float64).copy()

    exec_actions[:, 7] = forced_value
    delta_actions[:, 7] = forced_value - np.asarray(pi_actions[:, 7], dtype=np.float64)

    info.update({
        "applied": True,
        "mode": gripper_override,
        "value": forced_value,
        "delta_max": float(np.max(np.abs(delta_actions[:, 7]))),
    })
    return exec_actions, delta_actions, info



def trim_chunk_for_execution(
    pi_actions: np.ndarray,
    exec_actions: np.ndarray,
    delta_actions: np.ndarray,
    skip_first_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop the first K waypoints for non-direct execution.

    This is deliberately NOT used for direct_ee primitives, because direct_ee
    starts from the current measured state and has only a short correction path.
    """
    k = max(0, int(skip_first_k))
    n = int(exec_actions.shape[0])
    if k <= 0:
        return pi_actions, exec_actions, delta_actions, 0
    if k >= n:
        k = max(0, n - 1)
    return pi_actions[k:].copy(), exec_actions[k:].copy(), delta_actions[k:].copy(), k


def build_direct_ee_correction_chunk(
    jacobian_group,
    obs: dict,
    pi_actions: np.ndarray,
    ee_offset_xyz: np.ndarray,
    gripper_override: str,
    args: Args,
    source_prefix: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a direct EE correction primitive from the current robot state.

    This is used only when correction_execution_mode == 'direct_ee' and RLT is
    enabled manually or by auto gate. It ignores VLA arm waypoints as an executed
    path, but can optionally inherit the VLA final gripper command.
    """
    current_state8 = np.asarray(obs["observation/state"], dtype=np.float64).reshape(-1)
    if current_state8.shape[0] < 8:
        raise ValueError(f"Expected observation/state with >=8 dims, got {current_state8.shape}")

    steps = max(1, int(args.direct_correction_steps))
    reference_actions = np.tile(current_state8[:8][None, :], (steps, 1)).astype(np.float64)
    exec_actions = reference_actions.copy()
    delta_actions = np.zeros_like(exec_actions, dtype=np.float64)

    ee_offset = np.asarray(ee_offset_xyz, dtype=np.float64).reshape(3)
    ee_norm = float(np.linalg.norm(ee_offset))

    info = {
        "source": source_prefix,
        "execution_mode": "direct_ee",
        "ee_offset": ee_offset.copy(),
        "gripper_override": gripper_override,
        "skip_arm_motion": False,
        "direct_steps": steps,
        "max_joint_delta": 0.0,
    }

    if ee_norm >= float(args.direct_min_ee_norm):
        direct_args = dataclasses.replace(args, ee_apply_last_k=steps)
        corrected_right, delta_ee, ee_info = apply_manual_ee_translation_offset(
            jacobian_group=jacobian_group,
            right_waypoints_7=reference_actions[:, :7].copy(),
            ee_offset_xyz=ee_offset,
            args=direct_args,
        )
        exec_actions[:, :7] = corrected_right
        delta_actions[:, :7] = delta_ee[:, :7]
        info.update(ee_info)
        info["source"] = f"{source_prefix}+direct_ee_offset"
        info["execution_mode"] = "direct_ee"
        info["ee_offset"] = ee_offset.copy()
        info["max_joint_delta"] = float(np.max(np.abs(delta_actions[:, :7])))
    else:
        info["source"] = f"{source_prefix}+direct_ee_zero"

    # Manual gripper override has highest priority.
    exec_actions, delta_actions, ginfo = apply_gripper_override(
        reference_actions,
        exec_actions,
        delta_actions,
        gripper_override,
        args,
    )

    # If no manual/model override, keep the VLA final gripper value.
    # No gripper hold/latch is used.
    if (not ginfo["applied"]) and bool(args.direct_use_vla_gripper_when_no_override) and pi_actions.size:
        vla_g = float(np.asarray(pi_actions, dtype=np.float64)[-1, 7])
        exec_actions[:, 7] = vla_g
        delta_actions[:, 7] = vla_g - reference_actions[:, 7]
        ginfo = {
            "applied": True,
            "mode": "vla_final",
            "value": vla_g,
            "delta_max": float(np.max(np.abs(delta_actions[:, 7]))),
        }
        info["source"] = f"{info['source']}+vla_gripper_final"

    info["gripper_info"] = ginfo

    if ee_norm < float(args.direct_min_ee_norm):
        info["skip_arm_motion"] = True
        exec_actions = exec_actions[:1].copy()
        delta_actions = delta_actions[:1].copy()

    return exec_actions.astype(np.float64), delta_actions.astype(np.float64), info


def apply_rlt_or_exploration(
    jacobian_group,
    obs: dict,
    pi_actions: np.ndarray,
    rlt_enabled: bool,
    current_mode: str,
    adapter,
    args: Args,
    manual_ee_offset: np.ndarray,
    gripper_override: str,
    precomputed_rlt: tuple[np.ndarray, dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    delta = np.zeros_like(pi_actions, dtype=np.float64)
    info = {
        "source": "none",
        "risk": 0.0,
        "value": 0.0,
        "stopped_by_risk": False,
        "ee_offset": np.asarray(manual_ee_offset, dtype=np.float64).copy(),
        "gripper_override": gripper_override,
    }

    if not rlt_enabled:
        exec_actions = pi_actions.copy()
        exec_actions, delta, ginfo = apply_gripper_override(pi_actions, exec_actions, delta, gripper_override, args)
        info["gripper_info"] = ginfo
        if ginfo["applied"]:
            info["source"] = "manual_gripper_only"
        return exec_actions, delta, info

    if current_mode == "RETURN" and not args.rlt_on_return:
        info["source"] = "disabled_on_return"
        exec_actions = pi_actions.copy()
        exec_actions, delta, ginfo = apply_gripper_override(pi_actions, exec_actions, delta, gripper_override, args)
        info["gripper_info"] = ginfo
        if ginfo["applied"]:
            info["source"] = "manual_gripper_only_return"
        return exec_actions, delta, info

    if str(args.correction_execution_mode).lower() == "direct_ee":
        # Direct semantic correction path.
        direct_ee_offset = np.asarray(manual_ee_offset, dtype=np.float64).reshape(3)
        effective_gripper = gripper_override
        source_prefix = "direct_manual"
        rlt_info = {}

        if adapter is not None:
            if precomputed_rlt is None:
                corrected_unused, rlt_info = adapter.correct_observation_chunk(obs, pi_actions)
            else:
                corrected_unused, rlt_info = precomputed_rlt
            info.update(rlt_info)

            if rlt_info.get("stopped_by_risk", False):
                info["source"] = "direct_rlt_risk_stop"
                exec_actions = pi_actions.copy()
                zero_delta = np.zeros_like(pi_actions, dtype=np.float64)
                info["gripper_info"] = {"applied": False, "mode": "none", "value": 0.0, "delta_max": 0.0}
                return exec_actions, zero_delta, info

            if rlt_info.get("kind") == "rlt_actor_critic_semantic" or rlt_info.get("source") == "rlt_checkpoint_semantic":
                direct_ee_offset = np.asarray(
                    rlt_info.get("ee_offset", np.zeros((3,), dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(3)
                model_gripper = str(rlt_info.get("gripper_override", "none"))
                effective_gripper = gripper_override if gripper_override in {"close", "open"} else model_gripper
                source_prefix = "direct_rlt_semantic"
            else:
                # Legacy checkpoint has joint residuals, not semantic EE offset.
                source_prefix = "direct_legacy_checkpoint_ignored"

        exec_actions, direct_delta, direct_info = build_direct_ee_correction_chunk(
            jacobian_group=jacobian_group,
            obs=obs,
            pi_actions=pi_actions,
            ee_offset_xyz=direct_ee_offset,
            gripper_override=effective_gripper,
            args=args,
            source_prefix=source_prefix,
        )
        direct_info.update({k: v for k, v in info.items() if k not in direct_info})
        return exec_actions, direct_delta, direct_info

    # Residual-on-VLA path. Kept for backward compatibility.
    if adapter is not None:
        if precomputed_rlt is None:
            corrected, rlt_info = adapter.correct_observation_chunk(obs, pi_actions)
        else:
            corrected, rlt_info = precomputed_rlt
        delta = np.asarray(corrected, dtype=np.float64) - np.asarray(pi_actions, dtype=np.float64)
        info.update(rlt_info)
        info["source"] = rlt_info.get("source", "rlt_checkpoint")
        if rlt_info.get("stopped_by_risk", False):
            exec_actions = pi_actions.copy()
            zero_delta = np.zeros_like(pi_actions, dtype=np.float64)
            exec_actions, zero_delta, ginfo = apply_gripper_override(pi_actions, exec_actions, zero_delta, gripper_override, args)
            info["gripper_info"] = ginfo
            return exec_actions, zero_delta, info
        corrected, delta, ginfo = apply_gripper_override(pi_actions, corrected, delta, gripper_override, args)
        info["gripper_info"] = ginfo
        if ginfo["applied"]:
            info["source"] = f"{info['source']}+manual_gripper"
        return corrected, delta, info

    # Manual EE translation exploration.
    if args.use_manual_ee_offset:
        corrected_right, delta_ee, ee_info = apply_manual_ee_translation_offset(
            jacobian_group=jacobian_group,
            right_waypoints_7=pi_actions[:, :7].copy(),
            ee_offset_xyz=manual_ee_offset,
            args=args,
        )
        exec_actions = pi_actions.copy()
        exec_actions[:, :7] = corrected_right
        info.update(ee_info)
        exec_actions, delta_ee, ginfo = apply_gripper_override(pi_actions, exec_actions, delta_ee, gripper_override, args)
        info["gripper_info"] = ginfo
        if ginfo["applied"]:
            info["source"] = f"{info['source']}+manual_gripper"
        return exec_actions.astype(np.float64), delta_ee.astype(np.float64), info

    # Old random joint-noise fallback. Avoid using this for real grasping unless very small.
    if args.explore_delta_std > 0:
        delta = np.random.normal(0.0, args.explore_delta_std, size=pi_actions.shape)
        delta[:, 7] = 0.0
        delta = np.clip(delta, -args.explore_delta_max, args.explore_delta_max)
        info["source"] = "random_joint_explore"
        exec_actions = (pi_actions + delta).astype(np.float64)
        exec_actions, delta, ginfo = apply_gripper_override(pi_actions, exec_actions, delta, gripper_override, args)
        info["gripper_info"] = ginfo
        if ginfo["applied"]:
            info["source"] = f"{info['source']}+manual_gripper"
        return exec_actions.astype(np.float64), delta.astype(np.float64), info

    info["source"] = "label_only"
    exec_actions = pi_actions.copy()
    exec_actions, delta, ginfo = apply_gripper_override(pi_actions, exec_actions, delta, gripper_override, args)
    info["gripper_info"] = ginfo
    if ginfo["applied"]:
        info["source"] = "manual_gripper_only"
    return exec_actions, delta, info


def process_pause_command(
    cmd: str,
    current_mode: str,
    current_prompt: str,
    rlt_enabled: bool,
    ee_offset: np.ndarray,
    gripper_override: str,
    record_enabled: bool,
    args: Args,
    recorder: EpisodeReplayRecorder,
) -> tuple[str, str, bool, np.ndarray, str, bool, bool, bool]:
    """Process non-motion commands at the pause prompt.

    Returns:
        current_mode, current_prompt, rlt_enabled, ee_offset, gripper_override,
        record_enabled, execute_chunk, quit_requested
    """
    execute_chunk = False
    quit_requested = False

    if cmd == "":
        execute_chunk = True

    elif cmd == "r":
        current_mode = "RETURN"
        current_prompt = args.return_prompt
        rospy.logwarn(f"[MODE] Switched to RETURN. prompt='{current_prompt}'")

    elif cmd == "c":
        current_mode = "FEED"
        current_prompt = args.prompt
        rospy.logwarn(f"[MODE] Switched to FEED. prompt='{current_prompt}'")

    elif cmd == "t":
        rlt_enabled = not rlt_enabled
        rospy.logwarn(f"[RLT/EE-OFFSET] enabled={rlt_enabled}")

    elif cmd == "i":
        ee_offset[0] += args.ee_step
    elif cmd == "k":
        ee_offset[0] -= args.ee_step
    elif cmd == "j":
        ee_offset[1] += args.ee_step
    elif cmd == "l":
        ee_offset[1] -= args.ee_step
    elif cmd == "u":
        ee_offset[2] += args.ee_step
    elif cmd == "o":
        ee_offset[2] -= args.ee_step
    elif cmd == "z":
        ee_offset[:] = 0.0
    elif cmd == "p":
        rospy.logwarn(f"[EE-OFFSET] current xyz base-frame offset = {_fmt_arr(ee_offset)} m")

    elif cmd == "m":
        if not record_enabled:
            # Start a fresh recording episode. Chunks executed before this point
            # are intentionally not saved.
            ep_id = recorder.start_new_episode()
            record_enabled = True
            rospy.logwarn(
                f"[RECORD] START episode={ep_id}. Subsequent executed chunks WILL be saved."
            )
        else:
            # Finish the current recording episode and stop recording. The next
            # m starts a new episode.
            outcome = str(args.record_stop_outcome).lower()
            ep_dir = recorder.finish_episode(outcome=outcome, note=f"m_stop_{outcome}")
            record_enabled = False
            rospy.logwarn(
                f"[RECORD] FINISH episode outcome={outcome}, saved={ep_dir}. "
                "Subsequent chunks will NOT be saved until m is pressed again."
            )

    elif cmd == "g":
        gripper_override = "close"
        rospy.logwarn("[GRIPPER] Next executed chunk will force CLOSE, independent of VLA gripper output.")
    elif cmd == "b":
        gripper_override = "open"
        rospy.logwarn("[GRIPPER] Next executed chunk will force OPEN, independent of VLA gripper output.")
    elif cmd == "a":
        gripper_override = "none"
        rospy.logwarn("[GRIPPER] Cleared gripper override.")

    elif cmd in {"s", "f", "x", "e", "n"}:
        outcome = {
            "s": "success",
            "f": "failure",
            "x": "unsafe",
            "e": "env_reset",
            "n": "neutral",
        }[cmd]
        ep_dir = recorder.finish_episode(outcome=outcome, note=f"manual_{outcome}")
        record_enabled = False
        rospy.logwarn(
            f"[EPISODE] finished outcome={outcome}, saved={ep_dir}. "
            "Recording is now OFF; press m to start a new episode."
        )
        if cmd == "e":
            rospy.logwarn("[EPISODE] Environment reset marked. Move object, then continue.")

    elif cmd == "q":
        rospy.logwarn("[RUN] Quit requested.")
        quit_requested = True

    elif cmd == "h":
        rospy.logwarn(
            "Help: Enter=execute, r=RETURN, c=FEED, t=toggle residual, "
            "i/k=x+/x-, j/l=y+/y-, u/o=z+/z-, z=zero offset, "
            "m=start/finish recording episode, "
            "g=force close next chunk, b=force open next chunk, a=clear gripper, "
            "s/f/x/e/n=finish episode, q=quit."
        )

    else:
        rospy.logwarn(f"[CMD] Unknown command '{cmd}'. Press h for help.")

    ee_offset = np.clip(ee_offset, -float(args.ee_max_offset), float(args.ee_max_offset))

    if cmd in {"i", "k", "j", "l", "u", "o", "z"}:
        rospy.logwarn(f"[EE-OFFSET] xyz base-frame offset = {_fmt_arr(ee_offset)} m")

    return current_mode, current_prompt, rlt_enabled, ee_offset, gripper_override, record_enabled, execute_chunk, quit_requested


def main(args: Args) -> None:
    rospy.init_node("openpi_policy_client_two_prompts_rlt_episode_ee_offset", anonymous=True)

    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info(f"Server metadata: {policy.get_server_metadata()}")
    rospy.sleep(1.0)

    adapter = None
    if args.rlt_checkpoint:
        from rlt_training.runtime_adapter import RLTActionRuntime
        adapter = RLTActionRuntime(
            args.rlt_checkpoint,
            device=args.rlt_device,
            max_delta=args.rlt_max_delta,
            risk_stop_threshold=args.rlt_risk_stop_threshold,
        )
        rospy.loginfo(f"Loaded RLT checkpoint: {args.rlt_checkpoint}")

    recorder = EpisodeReplayRecorder(args.replay_dir, gamma=args.episode_gamma)

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

    jacobian_group = get_right_arm_chain_group(robot, args.right_arm_group_name)

    current_mode = "FEED"
    current_prompt = args.prompt
    rlt_enabled = False
    rlt_toggle_on_next = False
    rlt_toggle_off_next = False
    manual_ee_offset = np.zeros((3,), dtype=np.float64)
    gripper_override = "none"  # one of: none, close, open. Applied to next executed chunk, then reset.
    record_enabled = bool(args.record_from_start)
    if record_enabled:
        ep_id = recorder.start_new_episode()
        rospy.logwarn(f"[RECORD] record_from_start=True, START episode={ep_id}.")
    recorded_chunks = 0
    skipped_record_chunks = 0

    rospy.loginfo("=== TWO-PROMPT RLT EPISODE VERSION: SEMANTIC DIRECT_EE + GATE MONITOR ===")
    rospy.loginfo(
        "Controls: Enter=one chunk, r=RETURN, c=FEED, t=toggle residual, "
        "i/k=x+/x-, j/l=y+/y-, u/o=z+/z-, z=zero offset, "
        "m=start/finish recording episode, "
        "g=force CLOSE next chunk, b=force OPEN next chunk, a=clear gripper, "
        "s/f/x/e/n=finish episode, q=quit."
    )
    rospy.loginfo(
        f"EE offset step={args.ee_step:.4f} m, max={args.ee_max_offset:.4f} m, "
        f"apply_last_k={args.ee_apply_last_k}, joint_delta_max={args.ee_joint_delta_max:.4f} rad, "
        f"correction_execution_mode={args.correction_execution_mode}, "
        f"execute_skip_first_k={args.execute_skip_first_k}, "
        f"log_rlt_gate={args.log_rlt_gate}, auto_rlt_gate={args.auto_rlt_gate}, "
        f"auto_rlt_threshold={args.auto_rlt_threshold:.3f}, "
        f"record_from_start={args.record_from_start}, "
        f"record_stop_outcome={args.record_stop_outcome}."
    )

    # Warm-up inference.
    policy.infer(
        observation_ball(
            current_prompt,
            rotate_wrist_180=args.rotate_wrist_180,
            convert_to_rgb=args.convert_to_rgb,
            resize_hw=args.resize_hw,
        )
    )

    count = 0
    start_time = time.time()

    try:
        while not rospy.is_shutdown() and count < int(args.num_steps):
            cmd = input(
                f"\n[paused before next chunk | mode={current_mode} | "
                f"residual={rlt_enabled} | record={record_enabled} | "
                f"recorded_chunks={recorded_chunks} | skipped_record_chunks={skipped_record_chunks} | "
                f"ee_offset={_fmt_arr(manual_ee_offset)} m | "
                f"gripper_override={gripper_override} | steps={count}/{args.num_steps}]\n"
                "Enter=execute ONE chunk | r=RETURN | c=FEED | t=toggle residual | "
                "m=start/finish RECORD EPISODE | "
                "i/k=x+/x- | j/l=y+/y- | u/o=z+/z- | z=zero | "
                "g=force CLOSE next chunk | b=force OPEN next chunk | a=clear gripper | "
                "s=success | f=failure | x=unsafe | e=env_reset | n=neutral | q=quit | h=help > "
            ).strip().lower()

            prev_rlt_enabled = rlt_enabled
            (
                current_mode,
                current_prompt,
                rlt_enabled,
                manual_ee_offset,
                gripper_override,
                record_enabled,
                execute_chunk,
                quit_requested,
            ) = process_pause_command(
                cmd=cmd,
                current_mode=current_mode,
                current_prompt=current_prompt,
                rlt_enabled=rlt_enabled,
                ee_offset=manual_ee_offset,
                gripper_override=gripper_override,
                record_enabled=record_enabled,
                args=args,
                recorder=recorder,
            )

            if prev_rlt_enabled != rlt_enabled:
                rlt_toggle_on_next = rlt_enabled
                rlt_toggle_off_next = not rlt_enabled

            if quit_requested:
                break

            if not execute_chunk:
                continue

            obs = observation_ball(
                current_prompt,
                rotate_wrist_180=args.rotate_wrist_180,
                convert_to_rgb=args.convert_to_rgb,
                resize_hw=args.resize_hw,
            )

            if args.print_action_debug:
                rospy.loginfo(
                    f"[OBS] mode={current_mode}, residual={rlt_enabled}, "
                    f"ee_offset={_fmt_arr(manual_ee_offset)} m, "
                    f"gripper_override={gripper_override}, "
                    f"prompt='{current_prompt}', state={_fmt_arr(obs['observation/state'])}"
                )

            t_infer0 = time.time()
            result = policy.infer(obs)
            t_infer1 = time.time()

            if "actions" not in result or len(result["actions"]) < int(args.horizon):
                rospy.logwarn(f"Policy returned fewer than {args.horizon} actions, skipping.")
                continue

            pi_actions = np.asarray(result["actions"][: int(args.horizon)], dtype=np.float64)

            # Optional gate preview. This lets you see whether the model thinks
            # the current state should open/apply RLT correction. It does not
            # modify actions unless --auto-rlt-gate is used or manual t is ON.
            precomputed_rlt = None
            gate_prob = -1.0
            gate_auto_enabled = False
            if adapter is not None and (bool(args.log_rlt_gate) or bool(args.auto_rlt_gate)):
                try:
                    preview_corrected, preview_info = adapter.correct_observation_chunk(obs, pi_actions)
                    precomputed_rlt = (preview_corrected, preview_info)
                    gate_prob = float(preview_info.get("gate_prob", -1.0))
                    gate_auto_enabled = bool(args.auto_rlt_gate) and (gate_prob >= float(args.auto_rlt_threshold))
                    rospy.logwarn(
                        f"[RLT-GATE] gate_prob={gate_prob:.3f}, "
                        f"manual_residual={rlt_enabled}, "
                        f"auto_enabled={gate_auto_enabled}, "
                        f"threshold={float(args.auto_rlt_threshold):.3f}, "
                        f"ee_offset={_fmt_arr(np.asarray(preview_info.get('ee_offset', np.zeros(3))))}, "
                        f"gripper_override={preview_info.get('gripper_override', 'none')}, "
                        f"gripper_probs={_fmt_arr(np.asarray(preview_info.get('gripper_probs', np.zeros(3))))}"
                    )
                except Exception as exc:
                    rospy.logwarn(f"[RLT-GATE] preview failed: {exc}")
                    precomputed_rlt = None

            effective_rlt_enabled = bool(rlt_enabled or gate_auto_enabled)

            exec_actions, delta_actions, rlt_info = apply_rlt_or_exploration(
                jacobian_group=jacobian_group,
                obs=obs,
                pi_actions=pi_actions,
                rlt_enabled=effective_rlt_enabled,
                current_mode=current_mode,
                adapter=adapter,
                args=args,
                manual_ee_offset=manual_ee_offset,
                gripper_override=gripper_override,
                precomputed_rlt=precomputed_rlt,
            )
            rlt_info["manual_rlt_enabled"] = bool(rlt_enabled)
            rlt_info["auto_rlt_enabled"] = bool(gate_auto_enabled)
            rlt_info["effective_rlt_enabled"] = bool(effective_rlt_enabled)
            if gate_prob >= 0:
                rlt_info["gate_prob"] = float(gate_prob)

            if rlt_info.get("stopped_by_risk", False):
                if record_enabled:
                    recorder.add_chunk(EpisodeChunk(
                        obs,
                        pi_actions,
                        pi_actions,
                        np.zeros_like(pi_actions),
                        current_mode,
                        current_prompt,
                        rlt_enabled,
                        rlt_toggle_on_next,
                        rlt_toggle_off_next,
                        False,
                        False,
                        False,
                        t_infer1 - t_infer0,
                        0.0,
                        time.time(),
                        "rlt_risk_stop",
                    ))
                    recorded_chunks += 1
                else:
                    skipped_record_chunks += 1
                    rospy.logwarn("[RECORD] Not recording this risk-stop chunk because record=False.")
                rospy.logwarn(f"[RLT] Risk stop. risk={rlt_info['risk']:.3f}")
                rlt_toggle_on_next = False
                rlt_toggle_off_next = False
                continue

            if args.print_action_debug:
                rospy.loginfo(
                    f"[ACTION] mode={current_mode}, src={rlt_info.get('source', '')}, "
                    f"gate_prob={float(rlt_info.get('gate_prob', -1.0)):.3f}, "
                    f"manual_rlt={rlt_info.get('manual_rlt_enabled', rlt_enabled)}, "
                    f"auto_rlt={rlt_info.get('auto_rlt_enabled', False)}, "
                    f"effective_rlt={rlt_info.get('effective_rlt_enabled', rlt_enabled)}, "
                    f"ee_offset={_fmt_arr(np.asarray(rlt_info.get('ee_offset', manual_ee_offset)))} m, "
                    f"gripper_override={rlt_info.get('gripper_override', gripper_override)}, "
                    f"gripper_delta_max={float(np.max(np.abs(delta_actions[:, 7]))):.4f}, "
                    f"delta_max={float(np.max(np.abs(delta_actions))):.4f}, "
                    f"first={_fmt_arr(exec_actions[0])}, last={_fmt_arr(exec_actions[-1])}, "
                    f"infer_dt={t_infer1 - t_infer0:.3f}s"
                )

            # Skip first K waypoints for VLA/residual chunks, not for direct_ee.
            is_direct_execution = rlt_info.get("execution_mode", "") == "direct_ee"
            if is_direct_execution:
                pi_actions_record = pi_actions
                exec_actions_record = exec_actions
                delta_actions_record = delta_actions
                actual_skip = 0
                rospy.logwarn(
                    f"[DIRECT-EE] len={exec_actions_record.shape[0]}, "
                    f"skip_arm_motion={rlt_info.get('skip_arm_motion', False)}, "
                    f"source={rlt_info.get('source', '')}"
                )
            else:
                pi_actions_record, exec_actions_record, delta_actions_record, actual_skip = trim_chunk_for_execution(
                    pi_actions=pi_actions,
                    exec_actions=exec_actions,
                    delta_actions=delta_actions,
                    skip_first_k=args.execute_skip_first_k,
                )
                if actual_skip > 0:
                    rospy.logwarn(
                        f"[CHUNK-TRIM] Skipping first {actual_skip} waypoint(s). "
                        f"execute_len={exec_actions_record.shape[0]}/{exec_actions.shape[0]}"
                    )

            t_exec0 = time.time()
            if rlt_info.get("skip_arm_motion", False):
                ok = True
                rospy.logwarn("[DIRECT-EE] skip_arm_motion=True, only applying this chunk's gripper/no-op.")
            else:
                ok = plan_and_execute_horizon_with_arms_group(
                    robot=robot,
                    left_fixed_7=left_fixed,
                    right_waypoints_7=exec_actions_record[:, :7].copy(),
                    planning_time=args.planning_time,
                    waypoint_dt=args.waypoint_dt,
                    vel_scale=args.vel_scale,
                    acc_scale=args.acc_scale,
                )
            t_exec1 = time.time()

            if ok:
                rg = float(exec_actions_record[-1, 7])
                rospy.loginfo(
                    f"[GRIPPER] final_action={rg:.4f}, command={'close' if rg >= 0 else 'open'}, "
                    f"source={rlt_info.get('source', '')}"
                )
                if robot.gripper_r is not None:
                    if rg < 0:
                        robot.gripper_r.open()
                    else:
                        robot.gripper_r.close()
                count += int(exec_actions_record.shape[0])
            else:
                rospy.logwarn("Chunk planning/execution failed.")

            note = (
                f"src={rlt_info.get('source', '')};"
                f"ee_offset={_fmt_arr(manual_ee_offset)};"
                f"gate_prob={float(rlt_info.get('gate_prob', -1.0)):.5f};"
                f"manual_rlt={bool(rlt_info.get('manual_rlt_enabled', rlt_enabled))};"
                f"auto_rlt={bool(rlt_info.get('auto_rlt_enabled', False))};"
                f"record_enabled={record_enabled};"
                f"recorded_chunk_index={recorded_chunks};"
                f"max_joint_delta={float(np.max(np.abs(delta_actions_record[:, :7]))):.5f};"
                f"gripper_override={rlt_info.get('gripper_override', gripper_override)};"
                f"max_gripper_delta={float(np.max(np.abs(delta_actions_record[:, 7]))):.5f}"
            )

            if record_enabled:
                # IMPORTANT BUG FIX:
                # EpisodeChunk has explicit fields manual_ee_offset_xyz and gripper_override.
                # If we do not pass them here, EpisodeReplayRecorder saves default
                # manual_ee_offset_xyz=[0,0,0] and gripper_override=none, even though
                # the text note contains the correct values.
                saved_ee_offset = np.asarray(manual_ee_offset, dtype=np.float32).reshape(3)
                saved_gripper_override = gripper_override if gripper_override in {"close", "open"} else "none"

                recorder.add_chunk(EpisodeChunk(
                    obs,
                    pi_actions_record,
                    exec_actions_record,
                    delta_actions_record,
                    current_mode,
                    current_prompt,
                    bool(rlt_info.get("effective_rlt_enabled", rlt_enabled)),
                    rlt_toggle_on_next,
                    rlt_toggle_off_next,
                    bool(ok),
                    not bool(ok),
                    False,
                    t_infer1 - t_infer0,
                    t_exec1 - t_exec0,
                    time.time(),
                    note,
                    manual_ee_offset_xyz=saved_ee_offset,
                    gripper_override=saved_gripper_override,
                ))
                rospy.logwarn(
                    f"[RECORD] saved chunk label: manual_ee_offset_xyz={_fmt_arr(saved_ee_offset)}, "
                    f"gripper_override={saved_gripper_override}"
                )
                recorded_chunks += 1
            else:
                skipped_record_chunks += 1
                rospy.logwarn(
                    f"[RECORD] record=False, executed chunk NOT saved. "
                    f"Press m before the chunk where feeding data should start. skipped={skipped_record_chunks}"
                )

            rlt_toggle_on_next = False
            rlt_toggle_off_next = False
            # Gripper override is intentionally one-shot: it only affects the just-executed chunk.
            gripper_override = "none"

            rospy.loginfo(
                f"[{current_mode}] Chunk {'recorded' if record_enabled else 'executed-not-recorded'}. "
                f"steps~{count}/{args.num_steps}, recorded_chunks={recorded_chunks}, "
                f"residual={rlt_enabled}, src={rlt_info.get('source', '')}, "
                f"ee_offset={_fmt_arr(manual_ee_offset)} m, "
                f"gripper_override_applied={rlt_info.get('gripper_info', {}).get('mode', 'none')}, "
                f"infer_dt={t_infer1 - t_infer0:.3f}s, exec_dt={t_exec1 - t_exec0:.3f}s"
            )

    finally:
        saved = recorder.finish_episode(outcome="aborted", note="script_exit")
        if saved is not None:
            rospy.logwarn(f"[EPISODE] final partial episode saved as aborted: {saved}")

    rospy.loginfo(
        f"Finished. executed_steps={count}, recorded_chunks={recorded_chunks}, "
        f"skipped_record_chunks={skipped_record_chunks}, "
        f"total_time={time.time() - start_time:.2f}s, mode={current_mode}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
