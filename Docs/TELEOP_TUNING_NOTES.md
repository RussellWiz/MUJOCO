# Sigma.7 - dVRK PSM Teleop 调参记录与踩坑总结

本文记录 `Sigma/sigma7_psm_teleop.py` 从位置跟随、姿态映射、full IK、分离控制到抗抖 profile 的调参过程。重点是保存已经验证过的结论和踩过的坑，避免后续重复回到泛泛分析。

## 当前推荐运行方式

当前较可用的参数已经保存到 `Sigma/config.py`：

```powershell
D:\anaconda3\envs\dvrk\python.exe Sigma\sigma7_psm_teleop.py --config-profile split_antijitter_keep_rot
```

如果只想换日志文件：

```powershell
D:\anaconda3\envs\dvrk\python.exe Sigma\sigma7_psm_teleop.py --config-profile split_antijitter_keep_rot --log-csv Logs\test01.csv
```

查看可用 profile：

```powershell
D:\anaconda3\envs\dvrk\python.exe Sigma\sigma7_psm_teleop.py --list-config-profiles
```

当前 profile 的核心策略：

- `orientation_mode = split`
- 位置控制点：`track_body = tool_wrist`
- 姿态控制点：`orientation_body = tool_tip`
- 夹爪锁死：`jaw_mode = locked`
- 主链 `yaw / pitch / insertion` 只负责 xyz 位移
- distal 链 `roll / wrist_pitch / wrist_yaw` 只负责姿态

## 基础结论

1. PSM 必须保留 RCM 约束，不能当普通自由 6D 末端机械臂调。
2. `yaw / pitch / insertion` 是 RCM 主链，天然主要决定空间位置。
3. `roll / wrist_pitch / wrist_yaw` 更适合承担 distal 姿态变化。
4. 如果强行让同一组 IK 同时自由解 xyz 和 YRP，主链会和 wrist 互相抢运动，表现为关节抽动、姿态没反应或位置失真。
5. `tool_tip` 做位置控制更严格，但会强烈限制姿态；`tool_wrist` 做位置控制更宽松，姿态更容易转起来。
6. 当前工程目标不是严格复现官方 full orientation teleop，而是在可控性上折中：xyz 可移动，末端可转，RCM 主链不过度乱跳。

## 阶段一：rpy 轴映射尝试

早期使用 `orientation-mode rpy`，思路是：

- position IK 先让 `tool_tip` 跟随 xyz
- Sigma 姿态拆成 `yaw / pitch / roll`
- yaw/pitch 用 `blend/direct/bias` 影响主链
- roll 直接跟随
- wrist_pitch / wrist_yaw 锁在 calibration home

这个阶段观察到：

- `tool_tip` 平移跟随稳定，常见误差约几毫米。
- roll 有明显响应。
- pitch 有响应但受 position IK 抵消。
- yaw 最容易被 position IK 抵消，因为 yaw 本身就是 tool_tip x/y 的关键自由度。
- `direct` 模式能让 yaw/pitch/roll 明显转，但会牺牲位置，误差可到厘米级。
- `blend` 模式位置更稳，但 yaw 常被抵消。

### 坑：yaw 和 roll 读到了同一个 Sigma 轴

日志里出现过：

```text
rpy_in=(yaw, pitch, roll)
yaw == roll
```

尤其在 `master_yaw_axis=z`、`master_roll_axis=z` 时，yaw 和 roll 完全同源。后续测试 `x/y/z` 映射后，发现不是单纯换轴能解决。

### 坑：单轴角提取策略本身不稳

原始方法 `master_local_axis_angle()` 是从相对旋转矩阵里对每个局部轴分别 `atan2` 抽角。这在组合旋转下会串轴、跳变，尤其 Sigma.7 手柄实际动作不是理想单轴旋转时。

尝试过新增：

```text
--rpy-input-method rotvec
```

它用 rotation vector 分量替代单轴 `atan2`。但实际仍不能从根本上解决，因为问题不只在姿态输入拆解，也在 PSM 自由度分配。

结论：不要继续把主要精力放在 `--master-yaw-axis x/y/z` 上，rpy 拆轴不是最终方案。

## 阶段二：参考 v0 full orientation IK

参考文件：

```text
Sigma/v0_请勿参考/sigma7_psm_teleop_v0.py
```

v0 能明显实现 y/r/p 三方向旋转，原因不是它轴映射更准，而是控制策略不同：

1. v0 不把 Sigma 姿态拆成 yaw/pitch/roll 标量。
2. v0 使用完整相对旋转矩阵：

```python
master_rotation_delta = master_rotation @ calibration.master_rotation.T
target_rotation = calibration.target_rotation_home @ master_rotation_delta
```

3. v0 用 full orientation IK，把位置和姿态放进同一个加权 IK。
4. v0 允许 `roll / wrist_pitch / wrist_yaw` 参与姿态求解。
5. v0 的主控制点是 `tool_wrist`，不是 `tool_tip`。

### 坑：直接复刻 full IK 到主脚本会冲突

加入 `orientation-mode hybrid` 后，现象是：

- `tool_wrist` 有旋转，但 insertion 不稳定。
- xyz 位移一开始不明显，后来修正 actuator 更新方式后 xyz 出现。
- `tool_tip` 模式下容易“锁死”。
- `ctrl_yaw / ctrl_pitch / ctrl_roll` 抽动明显。
- YRP 旋转反而不一定有明显响应。

原因：

- full IK 把 `yaw / pitch / insertion / roll / wrist_pitch / wrist_yaw` 放在同一个求解器。
- 在 RCM 约束下，主链既想满足位置，又被姿态项拉扯。
- 到极限位置或 Jacobian 条件变差时，IK 会在多个关节之间来回找解，造成抽动。

另一个坑是 actuator 更新方式：

- position-only 稳定策略里用 `measured_qpos + dq`
- v0 风格更接近 `data.ctrl + dq`

full-pose IK 下如果继续用 `measured_qpos + dq` 并配合 insertion lag guard，控制量会被吃掉，尤其 insertion 会被反复重置。

## 阶段三：分离运动量 split 模式

最终采用的主要策略是：

```text
--orientation-mode split
```

核心思想：

- 位置任务：只用 `yaw / pitch / insertion`
- 姿态任务：只用 `roll / wrist_pitch / wrist_yaw`

具体实现：

- 对 `track_body` 计算 position Jacobian。
- 用 `POSITION_JOINTS = ("yaw", "pitch", "insertion")` 解 xyz。
- 对 `orientation_body` 计算 rotational Jacobian。
- 用 `DISTAL_JOINTS = ("roll", "wrist_pitch", "wrist_yaw")` 解姿态。
- 最后把两个 dq 合并到 6 关节命令。

当前推荐：

```text
track_body = tool_wrist
orientation_body = tool_tip
```

原因：

- `tool_wrist` 作为位置点，RCM 主链更容易稳定移动。
- `tool_tip` 作为姿态点，末端视觉旋转更符合预期。
- 如果 `tool_tip` 同时作为位置和姿态点，约束太强，姿态更容易变弱或锁住。

## 阶段四：夹爪锁死

当前先锁死夹爪：

```text
--jaw-mode locked
```

原因：

- 调姿态和位置时，夹爪开合会引入额外视觉变化。
- jaw mimic 关节会让末端视觉更复杂。
- 先锁死可以减少变量，方便判断 `roll / wrist_pitch / wrist_yaw` 是否真的在承担姿态。

后续如果要恢复夹爪跟随，可以用：

```text
--jaw-mode follow
```

## 阶段五：抽动问题与抗抖

split 模式下已经能同时有位移和旋转，但早期存在抽动，尤其在极限位置附近。

### 抽动来源

1. 输入噪声导致 IK 每帧产生很小但方向变化的 dq。
2. 目标靠近关节限位时，Jacobian 条件变差，解会来回摆。
3. distal wrist 接近限位，尤其 `wrist_yaw` 接近边界时，旋转任务容易被放大。
4. position 和 orientation 虽然分离，但最终仍在同一个机械链上，极限位姿附近不可避免会耦合。
5. insertion actuator 有几毫米滞后，不能把所有 `ins(q/c)` 差值都当成 Sigma 输入问题。

### 已加入的抗抖机制

1. `--dq-deadband`

小于阈值的关节增量直接清零：

```text
dq_deadband = 0.00008
```

2. `--split-distal-smooth`

distal wrist 单独低通，避免姿态链过度敏感：

```text
split_distal_smooth = 0.18
```

3. `--limit-slowdown-margin`

关节接近 actuator ctrlrange 限位时提前减速：

```text
limit_slowdown_margin = 0.14
limit_slowdown_min_scale = 0.12
```

4. insertion 单独更保守：

```text
hybrid_insertion_dq_scale = 0.16
hybrid_dq_smooth_ins = 0.07
max_ctrl_step_ins = 0.00032
```

虽然参数名里还有 `hybrid`，当前 split 模式也复用了这套 insertion 平滑逻辑。

### 坑：过度抗抖会把姿态锁死

曾尝试更保守参数：

- 降低 `orientation_gain`
- 增大 `damping_full`
- 降低 `max_orientation_error`
- 降低 `max_joint_step`
- 降低 `split_distal_step_scale`
- 降低 `split_distal_smooth`
- 增大 `limit_slowdown_margin`

结果是末端旋转几乎被锁死。

结论：不能同时削弱所有姿态通道。当前较好的原则是：

- 保持姿态通道可动：`orientation_gain=0.28`、`split_distal_step_scale=0.7`、`split_distal_smooth=0.18`
- 主要稳主链位移和 insertion：轻微增加 `damping_pos`，轻微降低 `max_ctrl_step_pos/ins`
- 限位减速不要过大，否则一接近边界 wrist 会像被锁住

## 当前 profile 参数

当前 profile 名称：

```text
split_antijitter_keep_rot
```

关键参数：

```text
orientation_mode = split
track_body = tool_wrist
orientation_body = tool_tip
scale_x/y/z = 0.55 / 0.55 / 0.25
orientation_gain = 0.28
damping_pos = 0.0006
damping_full = 0.002
max_position_error = 0.006
max_orientation_error = 0.18
max_joint_step = 0.004
dq_deadband = 0.00008
target_smooth = 0.22
dq_smooth = 0.25
split_distal_smooth = 0.18
ctrl_smooth = 0.18
max_ctrl_step_pos = 0.0023
max_ctrl_step_ins = 0.00032
max_ctrl_step_roll = 0.0025
hybrid_insertion_dq_scale = 0.16
hybrid_dq_smooth_ins = 0.07
split_distal_step_scale = 0.7
limit_slowdown_margin = 0.14
limit_slowdown_min_scale = 0.12
jaw_mode = locked
```

## 日志分析重点

CSV 里重点看这些列：

- `target_x/y/z` vs body position：判断目标是否在移动
- `err_m`：判断 xyz 跟随误差
- `ori_err_norm`：判断姿态任务是否还没跟上
- `ins_q / ins_ctrl / ins_lag`：判断 insertion actuator 滞后
- `ctrl_yaw / ctrl_pitch / ctrl_insertion`：主链是否平稳
- `ctrl_roll / ctrl_wrist_pitch / ctrl_wrist_yaw`：distal 姿态是否响应
- `jaw_ctrl`：锁夹爪时应保持固定

已有日志顺序大致对应调参阶段：

```text
rpy_test_01.csv
rpy_axis_zyx.csv
rpy_rotvec_zyx.csv
hybrid_tool_wrist_01.csv
hybrid_tool_tip_01.csv
hybrid_tool_wrist_fix01.csv
split_wristpos_tipori_01.csv
split_tippos_tipori_01.csv
split_antijitter_01.csv
split_antijitter_keep_rot_01.csv
split_config_run.csv
```

## 后续建议

1. 不要再优先调 rpy 轴映射，除非只是做输入诊断。
2. 若极限位置仍轻微抖动，优先微调主链相关参数：

```text
damping_pos
max_ctrl_step_pos
max_ctrl_step_ins
hybrid_insertion_dq_scale
limit_slowdown_margin
```

3. 不要轻易降低这些姿态通道参数，否则末端会再次“锁死”：

```text
orientation_gain
split_distal_step_scale
split_distal_smooth
max_orientation_error
```

4. 如果要更严格控制 `tool_tip` 位置，可以尝试 `track_body=tool_tip`，但要预期姿态响应会变弱。
5. 如果要更大姿态范围，先检查 `ctrl_wrist_yaw` 和 `ctrl_wrist_pitch` 是否接近限位；限位附近继续加姿态 gain 只会带来抖动。
6. 夹爪恢复跟随后再单独调，不要和姿态/位移稳定性一起调。


# Sigma.7 手感调试笔记
## 核心调试原则
当 Sigma.7 设备自身手感**发沉、黏滞卡顿**，或是呈现**无动力空载感**时，**优先调试主手力反馈环路**，再改动分离式遥操作增益参数。

- 该故障现象，**有别于** MuJoCo 中 PSM 从手跟随运动迟缓的问题。
- 分离式遥操作参数主要影响**从手**运动表现，**并不会改变** Sigma.7 主手自由拖拽的手感。
- 若要贴合官方 SDK 示例标准，Sigma.7 的 `read_state` 状态读取与 `set_zero_force()` 零力设置，需运行在**独立高频循环**中。
- 避免将 Sigma.7 零力更新直接挂载到 MuJoCo 可视化窗口循环；
  原因是 `viewer.sync()` 与 `mj_step()` 会降低力反馈有效更新帧率，容易造成主手手感阻尼过大、发沉发涩。

## 推荐架构方案 V1
- 保留原有分离式遥操作映射逻辑，不做改动。
- 把 Sigma.7 数据采样 + 零力输出逻辑，放到**独立后台线程**高频运行。
- MuJoCo 遥操作主循环只消费**已缓存的最新主手采样数据**。

## 实操调试顺序
1. 检查 Sigma.7 后台循环是否稳定高频运行。
2. 先判定并调好 Sigma.7 自由空载拖拽手感。
3. 等待主手手感恢复正常后，再进行分离式运动参数的调校。

我可以帮你把这份 markdown 再精简成工程速查版，要不要我顺手优化一版？