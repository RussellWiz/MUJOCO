import os
import threading
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

# 修改为指向 psm_control.xml 的绝对路径，使用相对于本脚本的路径拼接即可
import pathlib
PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
XML_PATH = str(PROJECT_DIR / "Assest" / "psm_official" / "psm_control.xml")

# ---------- 键盘状态 ----------
_keys = set()
_lock = threading.Lock()


def _on_press(key):
    with _lock:
        try:
            _keys.add(key.char.lower())
        except AttributeError:
            _keys.add(key)


def _on_release(key):
    with _lock:
        try:
            _keys.discard(key.char.lower())
        except AttributeError:
            _keys.discard(key)


# ---------- 参数 ----------
# 远心运动（RCM）下不建议直接“控 tool_tip 的世界位置”，而是控制与机构几何对应的点。
# 对 Classic PSM 官方模型来说，insertion（滑动）关节的参考点在 outer_pitch 坐标系下
# 有一个固定偏移：xyz = (0, 0.4318, 0)（见 `Assest/psm_official/urdf/Classic/psm_base.urdf.xacro`）。
# 该点是远心机构的关键参考点（可理解为器械轴线穿过的“枢轴/穿刺口参考点”）。
PIVOT_BODY = "outer_pitch"
PIVOT_OFFSET_LOCAL = np.array([0.0, 0.4318, 0.0])  # (m) in `outer_pitch` frame

POS_STEP = 0.0008  # 每步笛卡尔位移 (m)
GRIP_STEP = 0.05  # 每步夹爪控制量（相对于 PSM 夹爪的角度）
JOINT7_STEP = 0.005  # 每步绕竖直方向的旋转步长 (rad)
DAMPING = 1e-2  # 阻尼系数
ORI_GAIN = 0.3  # 姿态纠偏增益
MAX_DQ = 0.01  # 每步单关节最大变化量

# 在占位惯性/约束模型里，持续闭环会把数值噪声放大成关节“抽动”。
# 加一个死区：误差很小就不再更新 ctrl，避免 yaw/pitch 在零附近来回抖。
POS_DEADBAND = 2e-4  # (m)
ROT_DEADBAND = 2e-3  # (rad) approximately

# 目标与关节更新平滑（抑制移动时抖动）
TARGET_SMOOTH = 0.25  # (0~1) 越大越“跟手”，越小越平滑
DQ_SMOOTH = 0.35      # (0~1) 越大越“跟手”，越小越平滑
MAX_POS_CORR = 0.002  # (m) 每步最多修正的笛卡尔误差（避免小步来回抖）

# 占位惯性 + equality 约束下，姿态闭环很容易造成高频发散。
# 临时模型默认只做“位置”闭环（更稳），需要姿态再打开。
TRACK_ORIENTATION = False
ORI_GAIN_POS_ONLY = 0.0


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if max_norm > 0.0 and n > max_norm:
        return v * (max_norm / n)
    return v

CTRL_HINT = """
┌───────────────────────────────────────┐
│          dVRK PSM 键盘控制             │
├─────────┬─────────────────────────────┤
│  W / S  │  末端 +X / -X (前/后)       │
│  A / D  │  末端 +Y / -Y (左/右)       │
│  PgUp/Dn│  末端 +Z / -Z (上/下)       │
│  O / P  │  夹爪 开 / 闭               │
│  Z / X  │  末端绕竖直轴 旋转          │
│  Esc    │  退出                       │
│                                       │
│  夹爪始终自动保持垂直向下              │
└─────────┴─────────────────────────────┘
"""


# ---------- 指定 body 上某点的世界坐标 ----------
def point_world_pos(data, body_id, offset_local: np.ndarray) -> np.ndarray:
    R = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + R @ offset_local


# ---------- 计算姿态纠偏误差 ----------
def get_orientation_error(R_cur, R_des):
    """
    计算当前旋转矩阵到期望旋转矩阵的误差向量
    """
    e_rot = 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )
    return e_rot


# ---------- 读取键盘 ----------
def get_cmd():
    dpos = np.zeros(3)
    dgripper = 0.0
    dRotZ = 0.0
    with _lock:
        if "w" in _keys:
            dpos[0] += POS_STEP
        if "s" in _keys:
            dpos[0] -= POS_STEP
        if "a" in _keys:
            dpos[1] += POS_STEP
        if "d" in _keys:
            dpos[1] -= POS_STEP
        if keyboard.Key.page_up in _keys:
            dpos[2] += POS_STEP
        if keyboard.Key.page_down in _keys:
            dpos[2] -= POS_STEP
        if "o" in _keys:
            dgripper = GRIP_STEP
        if "p" in _keys:
            dgripper = -GRIP_STEP
        if "x" in _keys:
            dRotZ = JOINT7_STEP
        if "z" in _keys:
            dRotZ = -JOINT7_STEP
    return dpos, dgripper, dRotZ


# ---------- 主循环 ----------
def main():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"找不到模型文件 {XML_PATH}。请先运行 Src/loadpsm_control.py 生成。")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    pivot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PIVOT_BODY)
    assert pivot_body_id != -1, f"找不到 body '{PIVOT_BODY}'，请检查 XML"

    # 与末端运动相关的6个执行器关节
    actuator_joints = ["yaw", "pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw"]
    qvel_indices = []
    for j_name in actuator_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        assert jid != -1, f"Missing joint {j_name}"
        qvel_indices.append(model.jnt_dofadr[jid])

    jaw_actuator_idx = 6  # ctrl_jaw 是第7个执行器
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]

    mujoco.mj_resetData(model, data)

    # 默认让机械臂达到初始位姿
    data.ctrl[0] = 0.0
    data.ctrl[1] = 0.0
    data.ctrl[2] = 0.12  # 插入一定深度以免干涉
    home_ctrl = data.ctrl.copy()

    # 将模型稍微多步长使初始位姿稳定
    for _ in range(500):
        mujoco.mj_step(model, data)

    # 记录初始的“对应点”（远心参考点）位置，以及参考姿态（使用该 body 的姿态打底）
    cmd_target_pos = point_world_pos(data, pivot_body_id, PIVOT_OFFSET_LOCAL).copy()
    cmd_target_R = data.xmat[pivot_body_id].reshape(3, 3).copy()
    target_pos = cmd_target_pos.copy()
    target_R = cmd_target_R.copy()
    dq_filt = np.zeros(6, dtype=float)

    print(CTRL_HINT)

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    step_cnt = 0
    prev_time = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Reset 检测
            if data.time < prev_time:
                data.ctrl[:] = home_ctrl
                cmd_target_pos = point_world_pos(data, pivot_body_id, PIVOT_OFFSET_LOCAL).copy()
                cmd_target_R = data.xmat[pivot_body_id].reshape(3, 3).copy()
                target_pos = cmd_target_pos.copy()
                target_R = cmd_target_R.copy()
                dq_filt[:] = 0.0
                print("[Reset] ctrl 已恢复")
            prev_time = data.time

            dpos, dgripper, dRotZ = get_cmd()

            # 1. 增量更新全局目标
            cmd_target_pos += dpos
            if dRotZ:
                # 计算绕世界 Z 轴的旋转矩阵
                c, s = np.cos(dRotZ), np.sin(dRotZ)
                Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                cmd_target_R = Rz @ cmd_target_R

            # 对位置目标做低通，避免每帧跳变导致的关节来回修正
            target_pos = target_pos + TARGET_SMOOTH * (cmd_target_pos - target_pos)
            # 姿态目标保持为严格旋转矩阵（不要对旋转矩阵做线性插值）
            target_R = cmd_target_R

            # 2. 与当前物理状态比较（闭环 IK）
            current_pos = point_world_pos(data, pivot_body_id, PIVOT_OFFSET_LOCAL)
            current_R = data.xmat[pivot_body_id].reshape(3, 3)

            e_pos = clip_norm(target_pos - current_pos, MAX_POS_CORR)
            e_rot = get_orientation_error(current_R, target_R) if TRACK_ORIENTATION else np.zeros(3)

            # 误差足够小则不更新，抑制抽动
            if np.linalg.norm(e_pos) < POS_DEADBAND and np.linalg.norm(e_rot) < ROT_DEADBAND:
                mujoco.mj_step(model, data)
                step_cnt += 1
                if step_cnt % 500 == 0:
                    p = point_world_pos(data, pivot_body_id, PIVOT_OFFSET_LOCAL)
                    print(
                        f"[PIVOT] x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}  "
                        f"gripper={data.ctrl[jaw_actuator_idx]:.2f}"
                    )
                viewer.sync()
                continue

            # 3. 计算雅可比（仅针对6个自由度）
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            point = current_pos
            mujoco.mj_jac(model, data, jacp, jacr, point, pivot_body_id)

            Jp = jacp[:, qvel_indices]
            Jr = jacr[:, qvel_indices]
            if TRACK_ORIENTATION:
                J = np.vstack([Jp, Jr])
            else:
                J = Jp

            # 使用增益构成追踪误差
            K_pos = 0.5  # 位置追踪比例

            if TRACK_ORIENTATION:
                # 误差拼接: [位置误差, 姿态误差]
                e = np.concatenate([K_pos * e_pos, ORI_GAIN * e_rot])
            else:
                e = K_pos * e_pos

            # 求解关节增量（位置-only 时为 3x6）
            dim = 6 if TRACK_ORIENTATION else 3
            JJT = J @ J.T + DAMPING * np.eye(dim)
            dq = J.T @ np.linalg.solve(JJT, e)

            # 对关节增量做低通，抑制移动时高频“抽动”
            dq_filt = dq_filt + DQ_SMOOTH * (dq - dq_filt)

            # 限幅
            dq_max = np.max(np.abs(dq_filt))
            if dq_max > MAX_DQ:
                dq_filt *= MAX_DQ / dq_max

            # 更新位姿控制量
            data.ctrl[:6] = np.clip(data.ctrl[:6] + dq_filt, ctrl_lo[:6], ctrl_hi[:6])

            # 更新夹爪控制量
            if dgripper:
                data.ctrl[jaw_actuator_idx] = float(
                    np.clip(
                        data.ctrl[jaw_actuator_idx] + dgripper,
                        ctrl_lo[jaw_actuator_idx],
                        ctrl_hi[jaw_actuator_idx],
                    )
                )

            mujoco.mj_step(model, data)

            step_cnt += 1
            if step_cnt % 500 == 0:
                p = point_world_pos(data, pivot_body_id, PIVOT_OFFSET_LOCAL)
                print(f"[PIVOT] x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}  " f"gripper={data.ctrl[jaw_actuator_idx]:.2f}")

            viewer.sync()

    listener.stop()


if __name__ == "__main__":
    main()

