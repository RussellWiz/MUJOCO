# sigma.7 → MuJoCo PSM 遥操作调参记录（稳定化过程）

本文总结 `Sigma/sigma7_psm_teleop.py` 的稳定化调参过程：从“能跟随但 yaw/pitch 抽动、insertion 抖动”到“稳定可控”。目标是给后续复现一个**症状→原因→改法→推荐参数**的闭环。

> 背景：PSM 模型在占位惯性/强约束/位置伺服（position actuator）场景下，对高频噪声非常敏感。sigma.7 读取到的位置/姿态存在细微抖动，若直接做 6D IK 并把解直接写到 `data.ctrl`，会把噪声放大成关节抽动。

---

## 1. 初始现象与定位

### 现象 A：`ctrl_yaw` / `ctrl_pitch` 剧烈抽动

- **表现**：关节在小范围高频抖动，视觉上像“打颤”，但末端总体能跟随。
- **常见诱因**：
  - sigma.7 输入噪声（位置/姿态的细小抖动）；
  - 位置伺服 kp 偏硬；
  - equality 约束/并联结构导致耦合更强；
  - IK 每帧都在做“微小误差修正”，形成高频来回更新。

### 现象 B：`insertion` 轻微/剧烈抖动

- **表现**：插入关节出现小幅高频上下跳变；有时甚至比 yaw/pitch 更显著。
- **关键原因**：
  - insertion 的单位是“米”，但与旋转关节一起共用同一个 `max-ctrl-step` 时，限速/滤波会出现**量纲不匹配**；
  - insertion 通常 actuator kp 更大、对噪声更敏感。

### 现象 C：一组参数“效果很好但完全动不了”

典型命令：

```bash
python Sigma/sigma7_psm_teleop.py --no-orientation --max-ctrl-step 0.0008 --ctrl-smooth 0.15 --deadband-pos 5e-4
```

原因是“三重抑制叠加”：

- `deadband-pos` 太大（小动作被死区吞掉）
- `ctrl-smooth` 太小（目标变化跟不上）
- `max-ctrl-step` 太小（每帧允许的 ctrl 变化太少）

---

## 2. 关键稳定化策略（从键控经验迁移）

从 `Src/load_keyboard_tempmodel_dvrk.py` 里总结出来并迁移到 sigma.7 的核心套路：

### 2.1 位置-only 优先（先稳再加姿态）

新增参数：

- `--no-orientation`：只跟踪位置，不跟踪 sigma.7 姿态（最稳的 baseline）

原因：

- 姿态闭环对“旋转误差 + 角速度雅可比”非常敏感，在占位惯性/约束场景下更容易引发高频振荡。

### 2.2 死区（deadband）

新增参数：

- `--deadband-pos`（m）
- `--deadband-rot`（rad，姿态误差范数阈值）

作用：
误差足够小时跳过控制更新，避免在零附近“来回修正”导致抽动。

### 2.3 目标低通（target smoothing）

新增参数：

- `--target-smooth`（0..1）

作用：
sigma.7 的微小位置噪声不会直接变成目标跳变，从源头减少抖动激励。

### 2.4 IK 解低通（dq smoothing）

新增参数：

- `--dq-smooth`（0..1）

作用：
平滑 IK 求解出来的关节增量，减少高频来回变化。

### 2.5 actuator 目标低通 + 限速（ctrl smoothing + rate limit）

新增参数：

- `--ctrl-smooth`（0..1）
- `--max-ctrl-step-rot`（rad/帧）
- `--max-ctrl-step-ins`（m/帧）

作用：
即使 dq 已经平滑，`data.ctrl`（位置伺服目标）如果每帧仍快速摆动，yaw/pitch 仍会抽动。对 `ctrl` 本身做低通 + 限速，是压抖的最后一道闸门。

---

## 3. insertion 抖动的专门处理（量纲与刚度问题）

为了解决 insertion 抽动，单独引入：

- `--insertion-dq-scale`：把 IK 解里的 insertion 分量缩小（0..1）
- `--dq-smooth-ins`：对 insertion 分量用更强平滑（0..1，越小越平滑）

为什么有效：

- insertion 是唯一的平移关节（m），通常对误差更敏感；
- 约束/刚度导致插入方向更容易形成高频来回纠偏；
- 单独“降速 + 降幅 + 强滤波”能显著减少 insertion 抖动，但不影响整体可控性。

---

## 4. 典型“症状 → 对策”速查

### 4.1 yaw/pitch 抖但还能动

- **先做**：增大 `--deadband-pos`（例如 `2e-4` → `5e-4`）
- **再做**：减小 `--max-ctrl-step-rot`（例如 `0.004` → `0.003` → `0.002`）
- **再做**：增大 `--ctrl-smooth`（例如 `0.22` → `0.28`）

### 4.2 insertion 抖动明显

- **先做**：减小 `--max-ctrl-step-ins`（例如 `0.0006` → `0.0004` → `0.0003`）
- **再做**：减小 `--insertion-dq-scale`（例如 `0.35` → `0.25` → `0.2`）
- **再做**：减小 `--dq-smooth-ins`（例如 `0.18` → `0.12` → `0.08`）

### 4.3 “几乎动不了”

- **优先检查**：`--deadband-pos` 是否过大（例如 `5e-4` 对小幅动作会很致命）
- **其次检查**：`--max-ctrl-step-rot/ins` 是否太小
- **最后检查**：`--ctrl-smooth` 是否过小（过小会让目标跟随非常慢）

---

## 5. 推荐运行方式（稳定 baseline）

### 5.1 最稳 baseline（建议先从这里开始）

只做位置跟踪：

```bash
python Sigma/sigma7_psm_teleop.py --no-orientation
```

> 说明：这是“你反馈满意”的调参思路固化版。具体数值可根据你设备噪声、仿真刚度再微调。

---

## 6. 代码层面最终落地的改动点（便于回溯）

文件：`Sigma/sigma7_psm_teleop.py`

- **IK 支持 position-only**：`compute_ik_step(..., track_orientation=...)`，位置-only 时使用 \(3\times6\) Jacobian。
- **死区**：小误差直接跳过控制更新（防 chatter）。
- **目标位置低通**：`target_position_cmd → target_position_filt`。
- **dq 低通**：`dq → dq_filt`，并对 insertion 分量单独处理。
- **ctrl 低通 + 限速**：`desired_ctrl → ctrl_filt → rate-limited delta → data.ctrl`。
- **分关节限速**：`max-ctrl-step-rot` 与 `max-ctrl-step-ins` 分开，解决量纲不一致引发的 insertion 抖动。

---

## 7. 后续增强建议（可选）

- **在 viewer 中实时显示/记录**：`e_pos`、`dq_filt`、`ctrl` 的 RMS，用数据指导调参。
- **姿态恢复策略**：稳定后再逐步加回姿态（去掉 `--no-orientation`，并从很小的 `--orientation-gain` 开始）。
- **模型侧稳定化**：如果你最终使用 `psm_control_stable.xml` 或进一步降低 actuator kp、提高 damping/armature，能显著减轻控制侧的负担。
