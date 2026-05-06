# dVRK Classic PSM（xacro/DAE）导入 MuJoCo 纪要：历经、踩雷点与当前方案

本文记录本项目将 **dVRK Classic PSM**（官方 ROS xacro + DAE mesh）在 **Windows** 环境下导入 **MuJoCo** 的过程、关键决策与踩雷点，目标是“**先能稳定加载/可视化/可控**，再谈精确动力学”。

---

## 背景与约束

- **官方资产形态**：ROS xacro + URDF 片段 + `package://` 引用 + DAE 网格
- **工作区**：`d:\DVRK\MUJOCO`
- **目标目录**：`Assest/psm_official`
- **核心约束**
  - Windows 环境下尽量不依赖 ROS 工作区（无 `xacro` 命令链）
  - 先可加载、结构不散、关节能动；惯量/接触后置
  - 视觉 mesh 不参与碰撞/质量：常用 `contype=0 conaffinity=0 density=0`
- **关键差异点**：URDF 中大量 **mimic**（并联结构、夹爪开合），MuJoCo 的 URDF 导入不可靠保留 mimic，需要 MJCF 的 equality 复刻。

---

## 一、最终形成的“两阶段”导入路线（现状）

### 阶段 A：官方 xacro → 可编译的“调试 URDF”

- **脚本**：`Src/load_dvrk_psm.py`
- **输入**：`dvrk_model-main/dvrk_model-main/urdf/Classic/*.xacro` + `meshes/Classic/PSM/*.dae`
- **输出**：
  - `Assest/psm_official/psm1_sca_mujoco.urdf`
  - `Assest/psm_official/converted_meshes/...`（DAE→OBJ + 拷贝其它网格）
  - `Assest/psm_official/README.md`
- **用途**：保证 MuJoCo 能直接编译、便于排查“链结构/坐标/网格引用/关节范围”等基础问题。

建议命令（仅编译、无 GUI）：

```bash
D:\anaconda3\envs\dvrk\python.exe d:\DVRK\MUJOCO\Src\load_dvrk_psm.py --no-gui
```

### 阶段 B：调试 URDF → 可控 MJCF（含 mimic equality）

- **脚本**：`Src/loadpsm_control.py`
- **输入**：`Assest/psm_official/psm1_sca_mujoco.urdf`
- **输出**：`Assest/psm_official/psm_control.xml`
- **控制方式**：MuJoCo GUI 的 Control 面板 slider（position actuator）
- **主控制关节（7 个）**：
  - `ctrl_yaw`
  - `ctrl_pitch`
  - `ctrl_insertion`
  - `ctrl_roll`
  - `ctrl_wrist_pitch`
  - `ctrl_wrist_yaw`
  - `ctrl_jaw`
- **mimic（7 条 equality）**：
  - `pitch_1 = +1 * pitch`
  - `pitch_2 = +1 * pitch`
  - `pitch_3 = -1 * pitch`
  - `pitch_4 = -1 * pitch`
  - `pitch_5 = +1 * pitch`
  - `jaw_mimic_1 = +0.5 * jaw`
  - `jaw_mimic_2 = -0.5 * jaw`

建议命令（仅编译、打印 actuator 列表）：

```bash
D:\anaconda3\envs\dvrk\python.exe d:\DVRK\MUJOCO\Src\loadpsm_control.py --no-gui
```

---

## 二、关键实现点（为什么这么做）

### 1）Windows 下不走 ROS xacro：改为 Python “有限展开”

官方 xacro 很复杂，本项目只需要 Classic PSM1 + SCA 工具链的子集，所以在 `load_dvrk_psm.py` 里做了：

- 抽取宏体（按宏名抓取 `<xacro:macro ...>...</xacro:macro>`）
- 展开常见 `${...}` 表达式（包含 `PI`、简单算术）
- 移除残留 xacro 标签
- 把 `package://dvrk_model/...` 转成本地相对路径，并把 `.dae` 转 `.obj`

### 2）给 URDF 补“稳定惯量”，只为让 MuJoCo 编译通过

MuJoCo 编译 URDF 时，运动 link 需要正的质量/惯量。官方 xacro/URDF 中有些 link 可能缺少 inertial 或 inertial 不适合直接用（尤其在我们抽取/裁剪宏后）。

因此 `load_dvrk_psm.py` 会给非 `world` link 注入一个简化 inertial（小质量 + 对角惯量），它的目的不是物理真实，而是：

- **保证 URDF 能被 MuJoCo 编译**
- **便于下一步定位“关节树/坐标/网格引用”问题**

### 3）mimic 在 MuJoCo URDF 导入中不可靠：必须改用 MJCF equality

PSM 里并联机构与夹爪手指大量依赖 URDF `<mimic>`。但 MuJoCo 的 URDF 导入不会稳定保留 mimic 语义，因此：

- 调试 URDF 阶段：可以把 mimic 标签移除/忽略，先让模型编译、可视化
- 可控 MJCF 阶段：用 `<equality><joint ... polycoef="0 k 0 0 0"/></equality>` 复刻 mimic

对应关系在 `Src/loadpsm_control.py` 的 `MIMIC_JOINTS` 字典里维护，避免散落在 XML 拼字符串里。

---

## 三、踩雷点汇总（高概率复现）

### 雷 1：URDF 的 RPY 不能直接当 MJCF 的 euler 用（会导致 link 错位“散架”）

**现象**

- 在 GUI 中能看到“主从关节”跟着动了（equality/actuator 生效）
- 但每个 link/visual 都不在正确位置，整体看起来不是一个刚性连续结构

**根因（本项目遇到的关键 bug）**

`loadpsm_control.py` 初版把 URDF joint/visual 的 `rpy` 字符串直接写入 MJCF 的 `euler="..."`。

但 **URDF 的 RPY（固定轴 roll-pitch-yaw）** 与 MuJoCo `euler` 的解释方式并非等价直接替换，导致坐标链变换错误，最典型表现是：

- 例如 `tool_main` 的零位位置从应在 `z≈0.93` 的位置跑到了 `x≈-0.68`，整条链“折过去”。

**修复方式（当前已落地）**

- 不再输出 `euler="r p y"`，改为：
  - 将 URDF `rpy` 显式转换成 **MuJoCo 四元数** `quat="w x y z"`
  - body 的姿态用 `quat`
  - visual geom 的局部姿态也用 `quat`

验证方式：对比同一零位下，官方 URDF 导入与生成 MJCF 的关键 body 位置应一致（例如 `tool_main`, `tool_wrist`, `outer_pitch_back` 等）。

### 雷 2：并联结构“看起来像多余的支链”，但不能随意删除

官方 PSM 里存在并联结构（例如 outer pitch 的多条视觉支链）。如果为了“简化”而删 link/joint：

- 可能导致模型视觉结构不完整
- 也可能导致后续你无法用 equality 复刻正确的从动关系

本项目当前策略是：**URDF 父子树尽量全保留**，主关节用 actuator 控制，从动关节用 equality 约束。

### 雷 3：DAE 直接喂给 MuJoCo/路径含 package:// 会卡住导入

**现象**

- 编译 URDF 报找不到 mesh，或加载失败

**解决**

- 把 `package://dvrk_model/...` 重写为本地相对路径
- 把 `.dae` 转换为 `.obj`（在 Windows 环境更易控、更统一）
  - 使用 `trimesh`（并依赖 `pycollada` 读取 DAE）

### 雷 4：visual mesh 参与质量/碰撞会让模型“难控/乱飞”

如果视觉 mesh 参与碰撞或参与质量（例如忘记 `density="0"`、或默认 contype/conaffinity 不为 0）：

- 可能引入大量三角面碰撞接触，稳定性变差
- 质量/惯量被 mesh 规模放大，控制表现异常（拖拽、抖动、下坠）

本项目当前默认：

- visual geom：`contype="0" conaffinity="0" density="0"`（只显示）
- 碰撞几何：后续再单独设计（可用 primitive 或简化凸包）

### 雷 5：初始 qpos 不满足 equality 会导致加载瞬间“拉扯/弹开”

当 equality 约束存在时，如果初始关节角不满足从动关系，加载后第一帧 solver 会强行满足约束，可能出现瞬间拉扯。

本项目目前主要靠“结构正确 + solver 参数相对温和”避免明显爆炸；如果仍出现弹开，优先考虑：

- 给关键关节设置初始 qpos（或 keyframe）使 equality 初始就满足
- 调整 equality 的 `solref/solimp`

---

## 四、当前可复现检查清单（推荐你之后每次改动都跑）

### 1）调试 URDF 能否编译

```bash
D:\anaconda3\envs\dvrk\python.exe d:\DVRK\MUJOCO\Src\load_dvrk_psm.py --no-gui
```

### 2）可控 MJCF 能否生成 + 编译

```bash
D:\anaconda3\envs\dvrk\python.exe d:\DVRK\MUJOCO\Src\loadpsm_control.py --no-gui
```

### 3）零位姿态对齐（URDF importer vs 生成 MJCF）

做法：用 `mujoco.mj_forward` 后对比关键 body 的 `xpos`（例如 `tool_main`, `tool_wrist`）。

---

## 五、后续建议（不破坏当前可用版本的前提下）

- **先做“视觉整体正确 + mimic 正确”**（当前已达成）
- 再做：
  - 更合理的惯量（仍可先用稳定假值，逐步替换）
  - 明确碰撞几何（建议先 primitive/简化凸包，避免全三角面碰撞）
  - 加入更友好的控制脚本（键盘/末端控制/限制器），但建议新建脚本，不覆盖当前稳定生成器

