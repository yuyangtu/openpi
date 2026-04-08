#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
import pathlib
import shutil
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME


# ===================== 配置区 =====================
TASK_DIRS = {
    "/media/tu/US202/bags/feeding_task_headmove": (
        "pick up the spoon and feed the person",
        "return the spoon to the start position",
    ),
    "/media/tu/US202/bags/feeding_task_headmove2": (
        "pick up the spoon and feed the person",
        "return the spoon to the start position",
    ),
}

REPO_ID = "pick_and_feed_headmove220_demo_goodcut_wrench6"

TARGET_FPS = 10
IMG_SIZE = (224, 224)

SKIP_EPISODE_INDICES = {
    67, 203, 204, 189, 177, 167, 136, 126, 116, 103, 101, 96, 77, 15, 6, 5, 3,
    18, 20, 26, 65, 81, 86, 94, 106, 117, 118, 141, 143, 145, 146, 147, 149,
    152, 156, 157, 158, 159, 183, 186, 189, 198, 110, 111, 112, 113, 104,
}

# ===== pick_end: gripper规则（保持不变）=====
GRIPPER_KEY = "q_gripper_r_FJ"
HOLD_THRESH = 0.7
HOLD_TIME_SEC = 2.0
ALLOW_DROP_RATIO = 0.1  # 允许10%掉帧抗抖

# ===== 新截断规则：峰值 + 回撤百分比 =====
CART_TOPIC = "/arm_r/cartesian_state_controller/cartesian_state_controller/cartesian_state"

X_SMOOTH_SEC = 0.3          # x 平滑窗口（秒），避免尖峰
PEAK_SEARCH_MIN_SEC = 0.5   # pick_end之后至少再看这么久才开始找峰值（避免pick_end附近抖动）
RETRACT_RATIO = 0.05        # 从峰值回撤 5% 就截断
RETRACT_HOLD_SEC = 0.3      # 回撤阈值需要持续多久才算有效（秒）
FEED_BUFFER_SEC = 0.0       # 可选：截断点再往后多留一点（秒），默认0

# 最短片段（秒）
MIN_EP_SEC = 1.0
# =================================================


def init_dataset(repo_id: str, state_dim: int = 14, action_dim: int = 8, fps: int = 10) -> LeRobotDataset:
    out_path = pathlib.Path(LEROBOT_HOME) / repo_id
    if out_path.exists():
        shutil.rmtree(out_path)

    features = {
        "image": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "wrist_image": {
            "dtype": "video",
            "shape": (IMG_SIZE[1], IMG_SIZE[0], 3),
            "names": ["height", "width", "channels"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state_dim"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action_dim"],
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="diana7",
        fps=int(fps),
        features=features,
        use_videos=True,
        video_backend="opencv",
        image_writer_processes=0,
        image_writer_threads=0,
    )


def read_images_256x256(bag_file: str, desired_fps: float = 10.0):
    interval = 1.0 / desired_fps
    last_time = {"image": -1e18, "wrist_image": -1e18}
    bridge = CvBridge()
    image_data = {k: [] for k in last_time}
    image_times = {k: [] for k in last_time}

    topic_map = {
        "/top_view/color/image_raw": "image",
        "/diana_R_view/color/image_raw": "wrist_image",
    }

    with rosbag.Bag(bag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=list(topic_map.keys())):
            cam_key = topic_map[topic]
            cur_time = t.to_sec()
            if cur_time - last_time[cam_key] < interval:
                continue
            last_time[cam_key] = cur_time

            np_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            resized = cv2.resize(np_img, IMG_SIZE)
            image_data[cam_key].append(resized)
            image_times[cam_key].append(cur_time)

    return image_times, image_data


def read_joint_states_fill_forward(bag_file: str, joint_names: list[str], target_fps: float = 10.0):
    all_times = []
    all_states = []
    joint_cache = {nm: None for nm in joint_names}

    with rosbag.Bag(bag_file, "r") as bag:
        for _, msg, t in bag.read_messages(topics=["/joint_states"]):
            cur_t = t.to_sec()
            cur_state = joint_cache.copy()

            for i, nm in enumerate(msg.name):
                if nm in cur_state:
                    cur_state[nm] = float(msg.position[i])

            for nm in joint_names:
                if cur_state[nm] is None:
                    cur_state[nm] = joint_cache[nm] if joint_cache[nm] is not None else 0.0
                else:
                    joint_cache[nm] = cur_state[nm]

            all_times.append(cur_t)
            all_states.append(cur_state.copy())

    if len(all_times) == 0:
        return [], [], []

    all_times = np.array(all_times, dtype=np.float64)
    sampled_times = np.arange(all_times[0], all_times[-1], step=1.0 / target_fps)
    if len(sampled_times) == 0:
        sampled_times = np.array([all_times[0]], dtype=np.float64)

    resampled_states = []
    for nm in joint_names:
        joint_values = np.array([s[nm] for s in all_states], dtype=np.float64)
        interpolated = np.interp(sampled_times, all_times, joint_values)
        resampled_states.append(interpolated)

    final_states = [dict(zip(joint_names, v)) for v in zip(*resampled_states)]
    final_actions = final_states[1:] + [final_states[-1]]
    return sampled_times.tolist(), final_states, final_actions


def _extract_x_from_msg(msg) -> float:
    if hasattr(msg, "pose") and hasattr(msg.pose, "position") and hasattr(msg.pose.position, "x"):
        return float(msg.pose.position.x)
    if hasattr(msg, "position") and hasattr(msg.position, "x"):
        return float(msg.position.x)
    if hasattr(msg, "x"):
        return float(msg.x)
    raise ValueError("Cannot extract x from cartesian_state message.")


def read_cartesian_x_resampled(bag_file: str, target_times: list[float], topic: str, default_x: float = 0.0):
    ts, xs = [], []
    with rosbag.Bag(bag_file, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            ts.append(t.to_sec())
            xs.append(_extract_x_from_msg(msg))

    if len(ts) == 0:
        return [default_x] * len(target_times)

    ts = np.array(ts, dtype=np.float64)
    xs = np.array(xs, dtype=np.float64)
    target_times = np.array(target_times, dtype=np.float64)
    return np.interp(target_times, ts, xs).tolist()


def read_wrench_resampled(bag_file: str, target_times: list[float], topic: str):
    """
    从 diana7_msgs/CartesianState 中读取 6维 wrench:
    [fx, fy, fz, tx, ty, tz]
    并插值到 target_times
    """
    ts = []
    wrs = []

    with rosbag.Bag(bag_file, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            ts.append(t.to_sec())
            wrs.append([
                float(msg.wrench.linear.x),
                float(msg.wrench.linear.y),
                float(msg.wrench.linear.z),
                float(msg.wrench.angular.x),
                float(msg.wrench.angular.y),
                float(msg.wrench.angular.z),
            ])

    if len(ts) == 0:
        return [np.zeros(6, dtype=np.float32) for _ in target_times]

    ts = np.array(ts, dtype=np.float64)
    wrs = np.array(wrs, dtype=np.float64)  # [N, 6]
    target_times = np.array(target_times, dtype=np.float64)

    out = []
    for d in range(6):
        out.append(np.interp(target_times, ts, wrs[:, d]))
    out = np.stack(out, axis=1).astype(np.float32)  # [T, 6]

    return [out[i] for i in range(len(out))]


def find_nearest_time_index(target_time: float, tlist: list[float]) -> int:
    if not tlist:
        return 0
    arr = np.array(tlist)
    return int(np.argmin(np.abs(arr - target_time)))


def find_pick_end_index_from_gripper(jstates, fps, hold_thresh, hold_time_sec, gripper_key, allow_drop_ratio):
    hold_frames = max(1, int(round(hold_time_sec * fps)))
    grip = np.array([float(s.get(gripper_key, 0.0)) for s in jstates], dtype=np.float64)
    above = (grip > hold_thresh).astype(np.int32)

    window_sum = np.convolve(above, np.ones(hold_frames, dtype=np.int32), mode="valid")
    min_ok = int(np.ceil((1.0 - allow_drop_ratio) * hold_frames))

    candidates = np.where(window_sum >= min_ok)[0]
    return int(candidates[0]) if len(candidates) else None


def find_first_sustained_index(values, start_idx, thresh, fps, hold_sec, mode="ge", allow_drop_ratio=0.1):
    hold_frames = max(1, int(round(hold_sec * fps)))
    v = np.array(values[start_idx:], dtype=np.float64)
    if len(v) < hold_frames:
        return None

    ok = (v >= thresh).astype(np.int32) if mode == "ge" else (v <= thresh).astype(np.int32)
    window_sum = np.convolve(ok, np.ones(hold_frames, dtype=np.int32), mode="valid")
    min_ok = int(np.ceil((1.0 - allow_drop_ratio) * hold_frames))

    candidates = np.where(window_sum >= min_ok)[0]
    return int(start_idx + candidates[0]) if len(candidates) else None


def smooth_series_moving_average(values: list[float], fps: float, smooth_sec: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    win = max(1, int(round(smooth_sec * fps)))
    if win <= 1:
        return v
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(v, kernel, mode="same")


def find_peak_and_retract_cut(vs: np.ndarray, start_idx: int, fps: float,
                              retract_ratio: float, retract_hold_sec: float,
                              allow_drop_ratio: float) -> tuple[int | None, int | None, float | None]:
    """
    在 vs[start_idx:] 中找峰值 peak_idx，然后从 peak_idx 起找首次
    vs <= peak_val*(1-retract_ratio) 且持续 retract_hold_sec 的 index 作为 retract_idx。
    返回 (peak_idx, retract_idx, peak_val)
    """
    if start_idx >= len(vs):
        return None, None, None

    seg = vs[start_idx:]
    if len(seg) == 0:
        return None, None, None

    peak_rel = int(np.argmax(seg))
    peak_idx = start_idx + peak_rel
    peak_val = float(vs[peak_idx])

    retract_thresh = peak_val * (1.0 - float(retract_ratio))

    retract_idx = find_first_sustained_index(
        vs.tolist(),
        start_idx=peak_idx,
        thresh=retract_thresh,
        fps=fps,
        hold_sec=retract_hold_sec,
        mode="le",
        allow_drop_ratio=allow_drop_ratio,
    )

    return peak_idx, retract_idx, peak_val


def add_episode(
    dataset: LeRobotDataset,
    episode_name: str,
    jtimes,
    jstates,
    jactions,
    wrench_series,
    itimes,
    idata,
    task_name: str,
):
    ep_buffer = dataset.create_episode_buffer()
    dataset.episode_buffer = ep_buffer
    total_frames = len(jtimes)

    for i in tqdm.tqdm(range(total_frames), desc=f"写入 {episode_name}"):
        c_h_idx = find_nearest_time_index(jtimes[i], itimes["image"])
        c_w_idx = find_nearest_time_index(jtimes[i], itimes["wrist_image"])

        joint_arr = np.array(list(jstates[i].values()), dtype=np.float32)   # 8维
        wrench_arr = np.array(wrench_series[i], dtype=np.float32)           # 6维
        st_arr = np.concatenate([joint_arr, wrench_arr], axis=0)            # 14维

        ac_arr = np.array(list(jactions[i].values()), dtype=np.float32)     # 8维

        frame = {
            "image": idata["image"][c_h_idx] if idata["image"] else np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8),
            "wrist_image": idata["wrist_image"][c_w_idx] if idata["wrist_image"] else np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8),
            "state": st_arr,
            "actions": ac_arr,
        }
        dataset.add_frame(frame)

    ep_buffer["size"] = total_frames
    ep_buffer["task"] = [task_name] * total_frames
    dataset.save_episode()


def process_all_tasks(task_dirs: dict, fps: float = 10.0):
    joint_names = [
        "arm_r_joint_1", "arm_r_joint_2", "arm_r_joint_3", "arm_r_joint_4",
        "arm_r_joint_5", "arm_r_joint_6", "arm_r_joint_7",
        "q_gripper_r_FJ",
    ]

    state_dim = len(joint_names) + 6   # 8 + 6 = 14
    action_dim = len(joint_names)      # 8

    ds = init_dataset(
        repo_id=REPO_ID,
        state_dim=state_dim,
        action_dim=action_dim,
        fps=int(fps),
    )

    written_count = 0
    bag_ep_idx = 0

    mapping_path = pathlib.Path("episode_bag_mapping.txt")
    if mapping_path.exists():
        mapping_path.unlink()

    for bag_dir, (task_prompt_feed, task_prompt_return) in task_dirs.items():
        bag_files = sorted(pathlib.Path(bag_dir).glob("*.bag"), key=lambda p: p.name)
        print(f"\n📂 输入目录: {bag_dir} -> {len(bag_files)} 个bag文件")

        for i, bag_path in enumerate(bag_files):
            bag_file = str(bag_path)
            print(f"\n🚀 处理: #{i:03d} {bag_path.name}")

            cur_ep_idx = bag_ep_idx
            bag_ep_idx += 1

            with mapping_path.open("a", encoding="utf-8") as f:
                f.write(f"{cur_ep_idx}\t{bag_path}\n")

            if cur_ep_idx in SKIP_EPISODE_INDICES:
                print(f"🗑️ 跳过整个bag episode_index={cur_ep_idx}: {bag_path.name}")
                continue

            try:
                itimes, idata = read_images_256x256(bag_file, desired_fps=fps)
                jtimes, jstates, jactions = read_joint_states_fill_forward(
                    bag_file, joint_names, target_fps=fps
                )

                if len(jtimes) == 0:
                    print(f"⚠️ 跳过: {bag_path.name} (joint_states 无帧)")
                    continue

                # 读取6维wrench，并对齐到jtimes
                wrench_series = read_wrench_resampled(
                    bag_file=bag_file,
                    target_times=jtimes,
                    topic=CART_TOPIC,
                )

                x_series = read_cartesian_x_resampled(
                    bag_file, jtimes, topic=CART_TOPIC, default_x=0.0
                )
                x_min, x_max = float(np.min(x_series)), float(np.max(x_series))

                # ===== pick_end（gripper不变）=====
                pick_end = find_pick_end_index_from_gripper(
                    jstates=jstates,
                    fps=fps,
                    hold_thresh=HOLD_THRESH,
                    hold_time_sec=HOLD_TIME_SEC,
                    gripper_key=GRIPPER_KEY,
                    allow_drop_ratio=ALLOW_DROP_RATIO,
                )
                if pick_end is None:
                    print(f"⚠️ {bag_path.name}: 没找到 pick_end => 跳过；x[min,max]=({x_min:.3f},{x_max:.3f})")
                    continue

                # ===== 新逻辑：从 pick_end 后找峰值，再从峰值回撤 5% 截断 =====
                vs = smooth_series_moving_average(x_series, fps=fps, smooth_sec=X_SMOOTH_SEC)

                peak_search_start = min(len(vs) - 1, pick_end + int(round(PEAK_SEARCH_MIN_SEC * fps)))

                peak_idx, retract_idx, peak_val = find_peak_and_retract_cut(
                    vs,
                    start_idx=peak_search_start,
                    fps=fps,
                    retract_ratio=RETRACT_RATIO,
                    retract_hold_sec=RETRACT_HOLD_SEC,
                    allow_drop_ratio=ALLOW_DROP_RATIO,
                )
                if peak_idx is None or peak_val is None:
                    print(f"⚠️ {bag_path.name}: 没找到 peak => 跳过；x[min,max]=({x_min:.3f},{x_max:.3f})")
                    continue

                if retract_idx is None:
                    feed_end = len(jtimes)
                else:
                    feed_end = retract_idx

                if FEED_BUFFER_SEC > 0:
                    feed_end = min(len(jtimes), int(feed_end + round(FEED_BUFFER_SEC * fps)))

                min_len = int(MIN_EP_SEC * fps)
                if feed_end < min_len:
                    print(f"⚠️ {bag_path.name}: feed段太短 ({feed_end/fps:.2f}s) => 跳过")
                    continue

                # 1) feed
                add_episode(
                    ds,
                    bag_path.name + "_feed",
                    jtimes[:feed_end],
                    jstates[:feed_end],
                    jactions[:feed_end],
                    wrench_series[:feed_end],
                    itimes,
                    idata,
                    task_name=task_prompt_feed,
                )
                written_count += 1

                # 2) return
                return_len = len(jtimes) - feed_end
                if return_len >= min_len:
                    add_episode(
                        ds,
                        bag_path.name + "_return",
                        jtimes[feed_end:],
                        jstates[feed_end:],
                        jactions[feed_end:],
                        wrench_series[feed_end:],
                        itimes,
                        idata,
                        task_name=task_prompt_return,
                    )
                    written_count += 1
                else:
                    print(f"⚠️ {bag_path.name}: return段太短 ({return_len/fps:.2f}s) => 跳过return段")

                retract_str = "None" if retract_idx is None else f"{retract_idx}({retract_idx/fps:.2f}s)"
                retract_thresh = peak_val * (1.0 - RETRACT_RATIO)

                wr_np = np.stack(wrench_series, axis=0) if len(wrench_series) > 0 else np.zeros((1, 6), dtype=np.float32)
                wr_mean = wr_np.mean(axis=0)
                wr_std = wr_np.std(axis=0)

                print(
                    f"✅ {bag_path.name}: original_episode_index={cur_ep_idx}, written_episode_count={written_count}, "
                    f"pick_end={pick_end}({pick_end/fps:.2f}s), peak_search_start={peak_search_start}({peak_search_start/fps:.2f}s), "
                    f"peak_idx={peak_idx}({peak_idx/fps:.2f}s, x_peak~{peak_val:.3f}), "
                    f"retract_thresh~{retract_thresh:.3f} (ratio={RETRACT_RATIO:.3f}), retract_idx={retract_str}, "
                    f"feed_end={feed_end}({feed_end/fps:.2f}s), return_len={return_len/fps:.2f}s, "
                    f"x[min,max]=({x_min:.3f},{x_max:.3f}), "
                    f"wrench_mean={np.round(wr_mean, 3)}, wrench_std={np.round(wr_std, 3)}"
                )

            except Exception as e:
                print(f"❌ 错误: {bag_path.name} => {e}")

    ds.stop_image_writer()

    print("\n✅ 全部完成！")
    print(f"Dataset: {LEROBOT_HOME}/{REPO_ID}")
    print(f"Written episodes (feed+return): {written_count}")
    print("Episode-bag mapping written to: episode_bag_mapping.txt")


if __name__ == "__main__":
    process_all_tasks(TASK_DIRS, fps=TARGET_FPS)










