"""
VHACD 圆环 + Panda 机械臂场景测试
键控操作与 load.py 完全一致。
"""
import threading
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

XML_PATH = r"d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene_vhacd_fine.xml"

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
EE_BODY     = "hand"
TCP_OFFSET  = np.array([0.0, 0.0, 0.1034])

POS_STEP    = 0.0006
GRIP_STEP   = 3
JOINT7_STEP = 0.005
DAMPING     = 1e-2
ORI_GAIN    = 0.3
MAX_DQ      = 0.002

CTRL_HINT = """
┌───────────────────────────────────────┐
│  Panda + 精细分解圆环 键盘控制         │
├─────────┬─────────────────────────────┤
│  W / S  │  末端 +X / -X (前/后)       │
│  A / D  │  末端 +Y / -Y (左/右)       │
│  PgUp/Dn│  末端 +Z / -Z (上/下)       │
│  O / P  │  夹爪 张开 / 闭合           │
│  Z / X  │  末端绕竖直轴 旋转          │
│  Esc    │  退出                       │
│                                       │
│  夹爪始终自动保持垂直向下              │
└─────────┴─────────────────────────────┘
"""

# ---------- TCP 世界坐标 ----------
def tcp_world_pos(model, data, ee_id):
    R = data.xmat[ee_id].reshape(3, 3)
    return data.xpos[ee_id] + R @ TCP_OFFSET

# ---------- 计算姿态纠偏误差 ----------
def vertical_orientation_error(data, ee_id):
    R_cur = data.xmat[ee_id].reshape(3, 3)
    z_des = np.array([0.0, 0.0, -1.0])
    x_proj = R_cur[:, 0].copy()
    x_proj[2] = 0.0
    norm = np.linalg.norm(x_proj)
    if norm < 1e-6:
        x_proj = np.array([1.0, 0.0, 0.0])
    else:
        x_proj /= norm
    y_des = np.cross(z_des, x_proj)
    R_des = np.column_stack([x_proj, y_des, z_des])
    e_rot = 0.5 * (np.cross(R_cur[:, 0], R_des[:, 0]) +
                   np.cross(R_cur[:, 1], R_des[:, 1]) +
                   np.cross(R_cur[:, 2], R_des[:, 2]))
    return e_rot

# ---------- 微分逆运动学 ----------
def diff_ik(model, data, ee_id, dpos):
    e_rot = vertical_orientation_error(data, ee_id)
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    point = tcp_world_pos(model, data, ee_id)
    mujoco.mj_jac(model, data, jacp, jacr, point, ee_id)
    Jp = jacp[:, :7]
    Jr = jacr[:, :7]
    J  = np.vstack([Jp, Jr])
    e  = np.concatenate([dpos, ORI_GAIN * e_rot])
    JJT = J @ J.T + DAMPING * np.eye(6)
    dq  = J.T @ np.linalg.solve(JJT, e)
    dq_max = np.max(np.abs(dq))
    if dq_max > MAX_DQ:
        dq *= MAX_DQ / dq_max
    return dq

# ---------- 读取键盘 ----------
def get_cmd():
    dpos = np.zeros(3)
    dgripper = 0.0
    djoint7  = 0.0
    with _lock:
        if 'w' in _keys: dpos[0] += POS_STEP
        if 's' in _keys: dpos[0] -= POS_STEP
        if 'a' in _keys: dpos[1] += POS_STEP
        if 'd' in _keys: dpos[1] -= POS_STEP
        if keyboard.Key.page_up   in _keys: dpos[2] += POS_STEP
        if keyboard.Key.page_down in _keys: dpos[2] -= POS_STEP
        if 'o' in _keys: dgripper  =  GRIP_STEP
        if 'p' in _keys: dgripper  = -GRIP_STEP
        if 'z' in _keys: djoint7   =  JOINT7_STEP
        if 'x' in _keys: djoint7   = -JOINT7_STEP
    return dpos, dgripper, djoint7

# ---------- 主循环 ----------
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    assert ee_id != -1, f"找不到 body '{EE_BODY}'，请检查 XML"

    torus_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torus")
    torus_jnt_id  = model.body_jntadr[torus_body_id]
    torus_qadr    = model.jnt_qposadr[torus_jnt_id]
    torus_init_qpos = model.qpos0[torus_qadr:torus_qadr + 7].copy()

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[torus_qadr:torus_qadr + 7] = torus_init_qpos

    ctrl_lo = model.actuator_ctrlrange[:7, 0]
    ctrl_hi = model.actuator_ctrlrange[:7, 1]
    home_ctrl = data.ctrl.copy()

    print(CTRL_HINT)

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    step_cnt = 0
    prev_time = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if data.time < prev_time:
                data.ctrl[:] = home_ctrl
                data.qpos[torus_qadr:torus_qadr + 7] = torus_init_qpos
                print("[Reset] ctrl + 圆环位置 已恢复")
            prev_time = data.time

            dpos, dgripper, djoint7 = get_cmd()

            dq = diff_ik(model, data, ee_id, dpos)
            data.ctrl[:7] = np.clip(data.ctrl[:7] + dq, ctrl_lo, ctrl_hi)

            if djoint7:
                data.ctrl[6] = float(np.clip(
                    data.ctrl[6] + djoint7, ctrl_lo[6], ctrl_hi[6]))

            if dgripper:
                data.ctrl[7] = float(np.clip(data.ctrl[7] + dgripper, 0, 255))

            mujoco.mj_step(model, data)

            step_cnt += 1
            if step_cnt % 500 == 0:
                tcp = tcp_world_pos(model, data, ee_id)
                print(f"[TCP] x={tcp[0]:+.3f}  y={tcp[1]:+.3f}  z={tcp[2]:+.3f}  "
                      f"gripper={data.ctrl[7]:.0f}")

            viewer.sync()

    listener.stop()

if __name__ == "__main__":
    main()
