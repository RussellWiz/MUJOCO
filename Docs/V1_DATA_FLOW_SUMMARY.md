# Sigma v1 数据流总结

这份笔记总结 `Sigma/v1` 是如何读取 sigma.7 数据、如何处理数据，以及如何把数据传进 MuJoCo 的。

## 1. 端到端流程

整条遥操作链路可以看成：

`sigma.7 硬件 -> SDK ctypes 封装 -> ForceDimensionState -> SigmaMaster 缓存 -> TeleopApp 主循环 -> MotionController/JawController -> MuJoCo data.ctrl -> mj_step() -> viewer/CSV`

## 2. 数据读取路径

最底层的读取发生在 `Sigma/force_dimension_sdk.py`。

- `ForceDimensionSDK.open()` 负责打开设备，并启用力输出 / 重力补偿。
- `read_state()` 会先尝试 DHD，必要时再回退到 DRD。
- `_read_state_dhd()` 读取：
  - 位置
  - 旋转
  - 夹爪角度
  - 线速度
  - 角速度
- 读取结果会被封装成 `ForceDimensionState`，作为 `v1` 后续模块共用的 Python 层数据结构。

## 3. 内部传输

`Sigma/v1/master.py` 负责 master 侧生命周期。

- `SigmaMaster.open()` 打开 SDK 设备，并启动一个后台轮询线程。
- 后台线程循环执行：
  - `read_state()`
  - `set_zero_force()`
- 最新样本会保存在 `_latest_state`。
- 主线程通过 `SigmaMaster.read_state()` 读取最近一次缓存的样本。

这是一种共享内存式的交接方式：

- 一个生产者线程
- 一个消费者线程
- 使用“最新值缓存”，不是队列

## 4. 主循环处理

`Sigma/v1/app.py` 负责消费 master 样本并驱动机器人。

每一轮循环大致会做：

1. `master_state = self.master.read_state()`
2. `orientation_error, insertion_lag, measured_arm_qpos, target_position_filt = self.motion.update_arm(master_state)`
3. `jaw_target = self.jaw.update(master_state)`
4. 调用 `mj_step()` 推进 MuJoCo
5. 调用 `viewer.sync()` 刷新界面

关键点是：MuJoCo 主循环只消费最新 master 样本，但不再直接决定 sigma.7 的 haptics 节拍。

## 5. 运动处理

`Sigma/v1/motion.py` 负责把 master 状态转成机器人控制量。

### 目标映射

`TargetMapper.update(master_state)` 会：

- 计算 sigma 相对标定点的平移量
- 乘上 `scale_x/y/z`
- 重新表达成任务空间坐标
- 得到目标位置
- 对目标位置做低通滤波

如果启用了姿态模式，它还会构造目标旋转并做滤波。

### IK 求解

`IkSolver.solve()` 会：

- 读取 MuJoCo Jacobian
- 计算位置 / 姿态误差
- 根据 `orientation_mode` 选择求解路径
  - `none`
  - `roll`
  - `rpy`
  - `full`
  - `hybrid`
  - `split`
- 返回关节增量 `dq`

### 执行器滤波

`MotionController.update_arm()` 会：

- 应用 deadband
- 限制每步变化量
- 在接近关节极限时减速
- 对 `dq` 做平滑
- 必要时加插入量滞后保护
- 把结果写入 `robot.data.ctrl`

## 6. 机器人侧数据

`Sigma/v1/robot.py` 是 MuJoCo 侧适配层。

它提供：

- 关节和执行器 ID
- 当前 body 位置
- 当前 body 旋转
- Jacobian
- 执行器控制范围

机器人是通过 `data.ctrl` 驱动的，不是直接改 `qpos`。

## 7. Jaw 数据流

jaw 控制和主臂是分开的。

- 输入：`master_state.gripper_deg`
- 映射：`gripper_to_jaw()`
- 滤波：deadband + smooth + max step
- 输出：`robot.data.ctrl[jaw_actuator]`

## 8. 标定

`make_calibration()` 负责保存 master 和机器人之间的初始对齐关系：

- sigma.7 的初始位置和旋转
- 机器人 home 位置和旋转
- 初始 `ctrl` 值

后续所有相对运动，都是基于这个标定来计算的。

## 9. 日志

如果开启 CSV logging，`TeleopApp.log_row()` 记录的是已经处理过的遥操作变量，而不是原始 SDK 缓冲。

常见记录项包括：

- master 位置
- target 位置
- tip 位置
- 位置误差和姿态误差
- 实际 qpos
- 执行器控制值
- jaw target
- 通信频率

## 10. 实际结论

从数据传输角度看，`v1` 是一个三层系统：

1. SDK 采样层
2. master 状态缓存传输层
3. MuJoCo 控制层

最重要的设计点是：sigma.7 的采样和零力输出现在运行在独立的高频后台循环里，所以主手手感被解耦出了 MuJoCo viewer 的节拍。
