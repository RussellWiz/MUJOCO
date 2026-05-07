# v1 为什么会乱抽：从数据读取和控制策略对比 v2

## 范围

这份说明聚焦分析 `Sigma/v1` 为什么会表现出不稳定、抖动、甚至“乱抽”，重点关注：

- `v1` 是怎么读取和使用 sigma.7 数据的
- 这些路径和 `v2` 相比有什么不同
- `v1` 控制链中哪些部分最容易放大噪声或引入振荡

本次主要查看的代码路径包括：

- [Sigma/v1/master.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/master.py)
- [Sigma/v1/motion.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/motion.py)
- [Sigma/v1/app.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/app.py)
- [Sigma/v2/master.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v2/master.py)
- [Sigma/v2/motion.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v2/motion.py)
- [Sigma/config.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/config.py)

## 简短结论

`v1` 不稳定的主要原因，不是它读取了“完全不一样”或者“明显错误”的 sigma.7 数据，而是同样的输入数据被送进了一个比 `v2` 更耦合、更敏感、层次更多的控制链。

`v1` 和 `v2` 的线程化 sigma 读取逻辑其实非常接近：

- 都调用 `device.read_state()`
- 都在后台线程中以大约 1 kHz 的频率轮询
- 每次读取后都调用 `device.set_zero_force()`
- 对外都通过 `read_state()` 返回最近一次缓存样本

真正拉开差距的，不是“怎么读”，而是“读完以后怎么用”：

- `v1` 使用相对于标定姿态的绝对位移
- `v1` 会构造绝对姿态目标矩阵，并进一步求解耦合的位置/姿态 IK
- `v1` 在目标、误差、关节步长、执行控制之间叠了多层滤波和限幅
- `v1` 某些模式下是基于 `data.ctrl` 继续积分，而不总是从测得的关节状态出发
- `v2` 则使用增量式输入，只解 3 自由度位置，并把腕部姿态控制完全拆开

这就是为什么 `v2` 通常会明显比 `v1` 稳。

## 1. 数据读取链路：v1 和 v2 基本一致

### 共同点

`v1` 和 `v2` 共享几乎相同的 sigma 输入结构：

- 打开 Force Dimension 设备
- 启动后台线程
- 持续读取最新状态
- 每个循环都输出 zero-force
- 控制主循环每次只拿最近一帧缓存值

相关代码：

- `v1`：`SigmaMaster._stream_loop()`，见 [Sigma/v1/master.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/master.py)
- `v2`：`SigmaMaster._stream_loop()`，见 [Sigma/v2/master.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v2/master.py)

### 小差别

`v2` 多了一个 `button_mask()` 接口，但这本身不会影响控制稳定性。

### 关于“是不是读数有问题”的结论

如果同一台 sigma.7、同一套环境下，`v1` 抖得厉害而 `v2` 平稳，那么第一怀疑对象不应该是“`v1` 读到了错误数据”。更应该怀疑的是：

- `v1` 在输入侧缺少足够的简化和抑制
- `v1` 后续控制链会把本来正常的 sigma 微小抖动放大出来

## 2. 第一个关键差异：v1 是绝对映射，v2 是增量映射

### v1

在 `v1` 中，目标位置是围绕标定姿态构造的：

- 先计算 `master_state.position - calibration.master_position`
- 再乘平移比例和坐标变换
- 最后加回 `target_position_home`

逻辑在 [Sigma/v1/motion.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/motion.py) 的 `TargetMapper.update()`。

这种做法的效果是：

- sigma 的当前姿态被当成“相对于 home 的绝对偏移”
- 任何轻微漂移、手抖、或者标定不一致，都会持续作为目标偏置存在
- 操作者如果手重新回到了一个“主观中位”，但和初始标定关系不一致，机器人仍然会持续追那个旧的绝对关系

### v2

在 `v2` 中，位置控制采用逐帧增量：

- `sigma_delta = master_state.position - prev_sigma_position`
- 很小的增量会先经过 `sigma_deadband` 被清零
- 世界系增量还会被 `max_world_increment` 限幅
- 之后才更新 command target

逻辑在 [Sigma/v2/motion.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v2/motion.py) 的 `SplitTeleopController.update()`。

这种做法的效果是：

- 输入噪声会在最前面就被抑制掉
- 不会因为标定和当前持握关系不完全一致而长期积累绝对偏差
- 每一步目标只能缓慢、有限地变化

### 为什么这会直接影响稳定性

这是 `v2` 手感更稳的核心原因之一。`v1` 允许 sigma 的微小位姿变化直接扰动绝对目标；`v2` 则只允许经过 deadband 和限幅后的微小增量进入控制器。

## 3. v1 在目标生成前没有对 sigma 平移做输入级 deadband

`v2` 明确在输入端处理了小扰动：

- `sigma_deadband`
- `max_world_increment`

而 `v1` 并没有在 `target_position_cmd` 生成前，对 sigma 平移做 deadband。

`v1` 有的是后级处理：

- target smoothing
- task error clipping
- joint-step clipping
- 可选 `dq_deadband`
- ctrl smoothing

但这些都发生在目标已经被生成之后。

这意味着在 `v1` 中：

- sigma 的微小噪声依然会持续推动 target
- IK 每一帧都能看到一个非零任务误差
- 控制器很难真正安静下来，尤其是在自由空间接近平衡点的时候

这正是“持续小抽动”的典型来源。

## 4. 第二个关键差异：即使是 split 模式，v1 的耦合仍然比 v2 更重

### v2 的做法

`v2` 的策略是刻意收缩过的：

- 位置只由 `yaw/pitch/insertion` 三个关节负责
- 腕部只由 `roll/wrist_pitch/wrist_yaw` 负责
- 腕部控制使用姿态增量直接更新
- 没有把末端 6DOF 姿态全部塞进一个统一的目标

也就是说，它把问题拆成了两个更简单的子问题。

### v1 的做法

`v1` 支持很多模式：

- `none`
- `roll`
- `rpy`
- `full`
- `hybrid`
- `split`

即使在 `split` 模式下，`v1` 仍然会维护：

- 一个滤波后的绝对目标位置
- 一个滤波后的绝对目标旋转
- 一个从当前姿态到目标姿态的 orientation error
- 在位置 IK 之外，再叠一层 distal orientation solve

对应逻辑位于 [Sigma/v1/motion.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/motion.py) 的 `MotionController.compute_errors()` 和 `IkSolver.solve()`。

和 `v2` 相比，`v1` 的耦合更重，因为它仍然在做这些事情：

- 构造绝对姿态目标矩阵
- 根据当前姿态和目标姿态计算误差
- 用一个反馈式姿态任务去驱动 distal joints，而不是简单地跟随有界增量

这会显著增加以下问题出现的概率：

- 坐标系不一致
- 轴映射不完美
- 目标滤波带来的相位延迟
- 位置链和腕部链之间互相“拖拽”

这些问题都会以抖动或者乱抽的形式表现出来。

## 5. v1 内部叠了很多层滤波，容易引入相位滞后

`v1` 内部有不少滤波和限幅层：

- `target_position_filt`
- `target_rotation_filt`
- `dq_filt`
- 可选的 `split_distal_smooth`
- hybrid/split 下额外的 insertion smoothing
- `ctrl_filt`
- 每关节 `max_ctrl_step`
- joint limit slowdown

单独看，这些机制都合理；但叠在一起后，容易形成一个“响应慢、但又持续修正”的控制器。只要底层执行器本身还有动力学延迟，就很容易出现相位滞后。

表现出来通常就是：

- 指令到得慢
- 机器人刚追上，主手已经换方向了
- 目标和实际运动错相
- 最后出现追赶、回弹、抽动、来回 hunt

而 `v2` 的链路短很多：

- 输入增量 deadband
- 目标增量限幅
- target smoothing
- 3 关节 DLS 求解
- dq smoothing
- actuator step 限幅

这个链条更短，更容易调，也更不容易形成“滤波器打架”。

## 6. v1 使用绝对旋转目标，还对旋转矩阵做滤波

在 `v1` 的 `full`、`hybrid`、`split` 模式下：

- sigma 姿态会先转成相对旋转
- 再构造一个 target rotation matrix
- 然后对这个目标矩阵做线性低通
- 再通过 SVD 正交化

这段逻辑在 [Sigma/v1/motion.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/v1/motion.py) 的 `TargetMapper.update()`。

这里的风险在于：

- 在线性矩阵空间里做插值，本身不是最干净的姿态插值方式
- 正交化可以修正矩阵，但不能完全消除滤波过程中的瞬态问题
- 一旦 master 姿态系和工具姿态系对得不够准，target rotation 就会以一种“自己也在漂”的方式变化，腕部会一直追着它跑

`v2` 直接绕开了这类问题，它只做：

- 计算相邻两帧 sigma 的姿态增量
- 转成 rotation vector
- 把这个增量直接、有限地加到腕部关节目标上

从数学上看它不如完整姿态目标“优雅”，但从远操作工程上看，它反而更稳、更抗噪。

## 7. v1 某些模式下是基于命令值继续积分，而不是始终基于测量值

在 `v1.update_arm()` 里：

- 对于 `hybrid` 和 `split`，`desired_ctrl` 是从 `self.robot.data.ctrl[...]` 开始的
- 对于其他模式，`desired_ctrl` 是从测得的 `arm_qpos()` 开始的

这意味着一部分模式下，新的命令是在“上一拍命令值”的基础上继续推，而不是每次都从实际关节测量值重新出发。

这会带来一种开环倾向：

- 如果执行器有滞后
- 如果 insertion 响应偏慢
- 如果仿真 plant 对命令的跟踪不完美

那么“命令状态”和“真实状态”就会逐渐分开。

一旦分开，后续修正又继续叠加在旧命令上，就容易出现：

- 命令堆积
- 追赶过头
- 控制器以为自己已经走到了某处，但机械体实际还没到

`v2` 虽然也会增量更新命令，但它的风险小得多，因为：

- 只解 3 个位置关节
- 输入增量在最前面就被限住
- 没有一个耦合的 full-pose 目标在不断推着系统跑

## 8. insertion 是 v1 里一个非常明显的风险点

从代码里能直接看出，`v1` 已经意识到 insertion 很容易出问题，因为它专门加入了：

- `max_insertion_servo_lag`
- `hybrid_insertion_dq_scale`
- `hybrid_dq_smooth_ins`
- 可选的 `hybrid_use_servo_lag_guard`

这本身就是一个很强的信号：insertion 轴已经被识别成 chatter/lag 的主要来源之一。

还有一个很关键的细节：

- insertion lag guard 在 `hybrid` 和 `split` 下不是默认强启的
- 只有显式打开 `hybrid_use_servo_lag_guard` 时，它才会在这些模式里生效

对应代码条件是：

- `if self.args.orientation_mode not in ("hybrid", "split") or self.args.hybrid_use_servo_lag_guard:`

这意味着在常见的 split 风格配置下，insertion 很可能依然暴露在“滞后后继续积分”的风险中。

这非常像实际看到的“抽一下”的来源：

- insertion 跟不上
- 控制器还在继续累计目标
- 同时位置误差和腕部误差还在变化
- 最后 arm 晚一点追上时，视觉上就像突然冲、突然回、突然抽

## 9. v1 的默认配置虽然叫 anti-jitter，但本质上仍然很复杂

`Sigma/v1/configs.py` 会从 [Sigma/config.py](/abs/path/d:/DVRK/MUJOCO-test/Sigma/config.py) 读取 `DEFAULT_PROFILE`，当前默认是：

- `split_antijitter_keep_rot`

这个 profile 的确已经在努力压抖动，但它依然包含：

- `orientation_mode = "split"`
- `track_body = "tool_wrist"`
- `orientation_body = "tool_tip"`
- 经过滤波的 distal orientation control
- 经过缩放和平滑的 insertion 控制
- rate-limited control

也就是说，即使是名字上最“稳”的那个 `v1` profile，本质上也还是一个被认真修补过的复杂控制器，而不是一个天然简单的控制器。

这决定了它的脆弱性天然会高于 `v2`。

## 10. “完全不稳、还乱抽”最可能的根因排序

如果 `v1` 比 `v2` 差很多，最可能的原因大致是：

1. sigma 的微小平移噪声在 `v1` 中直接进入目标生成，因为没有输入级平移 deadband。
2. 绝对位姿映射依赖初始标定，一旦标定关系和当前持握状态略有偏差，目标就会持续漂。
3. `split/hybrid/full` 下的绝对姿态目标追踪，让腕部一直在追一个经过滤波、可能带系误差的 target。
4. 多层滤波叠加后形成明显相位滞后，修正总是慢半拍。
5. insertion 跟踪落后，但控制器还在继续累计命令。
6. 某些模式下从 `data.ctrl` 继续推目标，而不是从真实测量值回到闭环，导致命令比实体恢复得更快。

这些都属于控制架构问题，不需要假设 sigma reader 本身有明显错误。

## 11. 为什么 v2 会明显更稳

`v2` 更稳，是因为它直接删掉了几整类风险：

- 不再绑定绝对 sigma-to-target 位置关系
- 不做 full-pose IK
- 不追踪绝对姿态目标矩阵
- 不做耦合的腕部反馈姿态求解
- 不再让 6 个关节同时去满足一个连续变化的位姿目标

取而代之的是：

- 有界的平移增量
- 只做 3 自由度位置求解
- 在一个更小的 Jacobian 上做 adaptive SVD-DLS
- 腕部直接跟随有限姿态增量

它不如 `v1`“全能”，但正因为收缩了问题，反而更适合稳定部署。

## 12. v1 的建议排查顺序

如果要在实验上验证上面的判断，建议按这个顺序排：

1. 先让 `v1` 尽量关闭 orientation 相关影响，观察主抖动是否明显下降。
2. 记录每帧 raw sigma position delta，和 `target_position_cmd` 的变化做对照。
3. 重点看 insertion：比较 `arm_qpos[2]` 和 `data.ctrl[insertion]` 是否长期分离。
4. 观察抖动是否主要在腕部姿态参与时出现。
5. 在 `TargetMapper` 输入端加 sigma 平移 deadband，再和原版做对比。
6. 在 split 模式下尝试让 `desired_ctrl` 从 measured arm state 出发，而不是从上一次 `data.ctrl` 继续累积。

如果第 1、3、5 步很快就能改善行为，那么基本就能确认：问题主要来自控制链本身，而不是 SDK 读数本身。

## 13. 最终结论

`v1` 的主要问题，不在于它和 `v2` 读到了完全不同的 sigma.7 数据。

真正的问题在于：`v1` 用一个更敏感、更容易积累滞后、更容易互相耦合的控制结构，去完成一个比 `v2` 更难的任务：

- 绝对目标跟踪，而不是有界增量跟踪
- 绝对姿态目标追踪，而不是腕部直接跟随增量
- 更多内部状态
- 更多滤波层
- 更多机会让漂移、滞后、过冲互相叠加

`v2` 更稳，不是因为它功能更多，而是因为它主动做了收缩，把问题控制在了更容易稳定的范围内。
