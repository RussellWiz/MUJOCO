# MuJoCo dVRK PSM 远心控制与抖动治理：踩坑总结

本文记录本项目在 MuJoCo 中对 dVRK PSM 做键盘/力反馈（sigma.7）遥操作时遇到的核心问题与对应修复，便于后续复现与排错。

## 1. 远心运动（RCM）下“控末端位置”不合适

- **现象**：直接控制 `tool_tip` 的世界坐标时，轨迹与“远心约束”不一致，容易出现不自然的运动。
- **根因**：PSM 的关键约束是器械轴线围绕 RCM/穿刺口枢轴点运动；`tool_tip` 位置并不能直接表达该约束。
- **处理**：
  - 键盘控制脚本将被控点改为“对应点”（插入机构参考点），并在该点上计算位置误差与雅可比。
  - 参考来源：Classic PSM 官方 xacro 中 `insertion` 关节 `origin xyz="0 0.4318 0"`（`Assest/psm_official/urdf/Classic/psm_base.urdf.xacro`）。

## 2. “控制点挂错 body”会导致平移不可达

- **现象**：sigma.7 遥操作里旋转 3 方向正常，但平移时机械臂几乎不动。
- **根因**：把控制点挂在 `outer_pitch` 上时，该点主要只受 `yaw/pitch` 影响，缺少 `insertion` 对 3D 平移的贡献，导致平移任务不可达/病态，DLS IK 结果趋近于 0。
- **处理**：将 sigma.7 的控制点改为 `tool_main`（`insertion` 的子 body），这样 `yaw/pitch/insertion` 都能参与实现平移。

## 3. 动力学参数“看起来真实”但在 MuJoCo 里可能非法

- **现象**：加载真实动力学参数 XML 时报错：
  - `inertia must satisfy A + B >= C; use 'balanceinertia' to fix`
- **根因**：
  - 论文/辨识参数通常是在“某个输出坐标系”下给出的惯性（含非对角项）。
  - 直接把惯性张量填到 MuJoCo body 坐标系下，如果坐标系不一致（缺少 \(RIR^T\) 旋转变换），可能导致惯性矩阵不满足正定/三角不等式约束，从而被 MuJoCo 判为非法。
- **处理**：
  - 在 `psm_control_dynamics.xml` 中开启 `balanceinertia="true"`，让 MuJoCo 自动平衡到可行惯性，先保证能跑。
  - 后续若要严格对齐，需要明确论文坐标系与 URDF/link frame 的相对姿态并做张量旋转变换。

## 4. “占位惯性 + 约束 + 高刚度伺服”会触发高频抖动

- **现象**：
  - 模型加载后 `ctrl_yaw/ctrl_pitch` 在零附近持续抽动；
  - 稍微移动后可能出现“失控式疯狂抖动”。
- **典型根因组合**：
  - **占位惯性**（大量 body 共享同一组很小的质量/惯性）使系统数值条件很差；
  - **equality 约束**（并联/联动关节）引入更强耦合；
  - **位置执行器 kp 过高**（例如 insertion kp 很大）在离散时间下等价于“硬弹簧”，容易激发高频；
  - **持续闭环 IK** 在误差很小也不断修正，会把数值噪声放大为关节抖动。

## 5. 旋转矩阵“线性插值”会让姿态控制发散

- **现象**：看似加入了平滑，但移动时反而出现明显发散/疯狂抖动。
- **根因**：对旋转矩阵做线性低通（矩阵直接相加插值）会破坏正交性，使其不再是合法旋转矩阵；姿态误差计算会变得不稳定，IK 输出乱跳。
- **处理**：
  - 不对旋转矩阵做线性插值；如需平滑姿态，使用四元数 slerp 或在 SO(3) 上插值。
  - 在临时占位模型中，优先关闭姿态闭环，仅做位置闭环以保证稳定。

## 6. 让“临时模型”先稳住的实用策略

在 `load_keyboard_tempmodel_dvrk.py` 的稳定化过程中，证明有效的策略包括：

- **死区（deadband）**：误差很小时跳过 `dq` 求解与 `ctrl` 更新，避免零附近抽动。
- **位置-only IK**：占位惯性 + 约束场景下，先只跟踪位置（3D 任务），不要强行叠加 6D 姿态任务。
- **每步位置修正限幅**：限制每仿真步最多纠正的位移误差，防止“过度修正 → 反向修正”循环。
- **（可选）降低执行器刚度与增加阻尼**：对于纯 XML 方案，可通过降低 `kp`、增大 `joint damping/armature/frictionloss`、适当增大 `iterations`、减小 `timestep` 来压高频。

## 7. 真实模型 / 临时模型 / 稳定模型文件组织

- **占位控制模型**：`Assest/psm_official/psm_control.xml`
- **导入辨识参数的真实动力学模型**：`Assest/psm_official/psm_control_dynamics.xml`
  - 已开启 `balanceinertia="true"` 以提高可加载性
- **占位惯性但更稳的参数版本**：`Assest/psm_official/psm_control_stable.xml`
- **键盘遥操作（临时模型，默认用 stable 版）**：`Src/load_keyboard_tempmodel_dvrk.py`
- **sigma.7 遥操作（控制点与键控同步）**：`Sigma/sigma7_psm_teleop.py`
- **sigma.7 遥操作（默认加载真实动力学 XML）**：`Sigma/teleop_realmodel.py`
  - 通过 `runpy.run_path` 按文件路径启动，避免 `Sigma` 不是 Python 包导致导入失败。

## 8. 后续建议（更“物理真实”且稳定）

- **严谨坐标系对齐**：将论文惯性从“输出坐标系”旋转到 MuJoCo body frame（\(I_{body} = R I_{paper} R^T\)），减少对 `balanceinertia` 的依赖。
- **控制与物理解耦**：真实惯性参数导入后，重新整定执行器 `kp` 与关节阻尼，避免“真实重 + 高刚度”导致的硬振。
- **姿态平滑用 SO(3) 方法**：需要平滑姿态时用四元数 slerp 或指数映射插值，避免线性混合矩阵。

