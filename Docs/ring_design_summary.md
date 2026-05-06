# Panda 抓取圆环：`load.py` / `testscene.py` / `testscene2.py` / `testvhacd.py` 圆环设计总结

本项目围绕“Panda 夹爪抓取圆环（torus）”做了多种**碰撞几何**建模方案。各测试脚本的差异主要在于 **加载的场景 XML 不同**，从而对应不同的圆环碰撞设计。

---

## 运行入口一览

- **Capsule 32 段圆环（原始分段胶囊）**：`Src/load.py`
  - `XML_PATH = d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene32.xml`
- **Torus mesh 扇区凸分解（20 凸包）**：`Src/testscene.py`
  - `XML_PATH = d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene_vhacd_fine.xml`
- **“32 capsule → 生成 mesh → 再扇区凸分解（32 凸包）”**：`Src/testscene2.py`
  - `XML_PATH = d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene_capsule32_decomp.xml`
- **VHACD/凸包网格路线测试（当前 64 凸包）**：`Src/testvhacd.py`
  - `XML_PATH = d:\DVRK\MUJOCO\Assest\franka_emika_panda\scene_vhacd64.xml`

各测试脚本的键盘控制逻辑一致（世界系末端平移 + 夹爪开合 + 末端绕竖直轴旋转，且夹爪自动保持竖直向下）。

---

## 方案 A：`scene32.xml`（32 个 `capsule` 原生分段圆环）

### 使用脚本
- `Src/load.py`

### 圆环建模方式
- **碰撞几何**：`<geom type="capsule">` *32 个*，首尾相接构成空心圆环。
- **优点**
  - 胶囊体属于解析几何，接触稳定性通常优于三角网格。
  - “空心”结构天然存在，不会退化成凸包实心。
- **缺点**
  - 胶囊段数增加会显著增加约束数（`nefc`），段数太高曾触发栈溢出（因此使用 32 段并将 `condim` 降到 4）。

### 尺寸（来自 XML 注释）
- 主半径 \(R = 0.075\,m\)，管半径 \(r = 0.016\,m\)
- 外径约 \(18.2\,cm\)，内径约 \(11.8\,cm\)，管径约 \(3.2\,cm\)

### 典型物理参数（示例）
每段胶囊：`condim="4" friction="10 4 0.5" solref="0.015 1" margin="0.002" mass="0.001"`

---

## 方案 B：`scene_vhacd_fine.xml`（直接对 torus mesh 做扇区凸分解：20 个凸包）

### 使用脚本
- `Src/testscene.py`

### 圆环建模方式
- **视觉**：`torus_visual.obj`
- **碰撞几何**：将 `torus.obj` 按角度切成扇区，对每个扇区取凸包，生成 `torus_fine_convex_0..19.obj`
  - XML 中体现为 20 个 `<mesh name="torus_chX" ...>` + 20 个 `<geom name="torus_colX" type="mesh" ...>`
- **优点**
  - 相比“单 mesh 碰撞”不会变成整体凸包（空心更容易保留）。
  - 分解块数可调（例如 16/20/24），精度与性能可折中。
- **缺点**
  - 仍然是三角网格碰撞：更容易出现点接触、滚动、夹爪挤压时“滑走”等现象，需要更强的摩擦/抗滑参数配合。

### 姿态与放置
- 圆环 body：`quat="0.707 0.707 0 0"`（绕 X 轴 90°，让圆环平放）

### 备注
虽然文件名叫 `vhacd_fine`，但当前实现是**扇区切分 + 凸包**，并非 PyBullet 的 VHACD（VHACD 在当前环境曾崩溃，改用更可控的扇区法）。

---

## 方案 C：`scene_capsule32_decomp.xml`（由 32 capsule 生成 mesh，再扇区凸分解：32 个凸包）

### 使用脚本
- `Src/testscene2.py`（内部复用 `testscene.py` 的控制逻辑，只替换了 `XML_PATH`）

### 圆环建模方式
- **生成脚本**：`Src/decompose_capsule32_torus.py`
  - 先用 `trimesh.creation.capsule(...)` 构建 32 段胶囊环的 **三角 mesh**：`Assest/franka_emika_panda/assets/torus_capsule32_visual.obj`
  - 再把该 mesh 按角度切扇区取凸包，生成 32 个凸包文件：`torus_capsule32_convex_0..31.obj`
  - 输出场景：`Assest/franka_emika_panda/scene_capsule32_decomp.xml`

### 质量设计（关键修复点）
- 视觉 geom 设置为 `density="0"`：只显示、不参与质量/惯量（否则视觉 mesh 会把总质量抬到 kg 级，导致“根本抓不起来”）。
- 碰撞凸包均分总质量：
  - 当前总质量目标：`TOTAL_MASS = 0.012 kg`
  - 每块凸包：`mass="0.000375"`（32 块合计 0.012 kg）

### 优缺点
- **优点**
  - “形状来源”与 capsule 环一致，但碰撞体变成多凸包 mesh，便于对比“解析几何 vs 网格凸包”的差异。
  - 块数更高（32 块），理论上更贴合。
- **缺点**
  - 仍然是三角网格碰撞 + 凸包拼接，稳定抓取更依赖摩擦与求解器参数。

---

## 方案 D：`scene_vhacd64.xml`（VHACD/凸包网格路线测试：64 个凸包）

### 使用脚本
- `Src/testvhacd.py`

### 圆环建模方式
- **生成脚本**：`Src/decompose_vhacd32_torus.py`
  - 最早测试过真实 VHACD 输出的 8 凸包版本：`scene_vhacd.xml`
  - 后续为缓解穿模，又生成了 32 凸包版本：`scene_vhacd32.xml`
  - 当前 `testvhacd.py` 指向更细的 64 凸包版本：`scene_vhacd64.xml`
- **碰撞几何**：`torus_vhacd64_convex_0..63.obj`，共 64 个凸包。
- **当前参数**
  - 总质量：`0.008 kg`
  - `condim="6"`
  - `friction="24 10 2"`
  - `iterations="100"`，`noslip_iterations="50"`
  - 视觉 geom：`density="0"`，只显示、不参与质量/惯量。

### 已观察到的问题：穿模严重
- **8 凸包 VHACD**：穿模明显。凸包数量太少，圆环内孔和管壁被过粗地近似，夹爪接触时容易穿过局部空隙或错误接触面。
- **32 凸包版本**：穿模仍然严重。虽然环向分块变细，但每块仍是凸包，管截面的凹/圆形表面无法被稳定表达。
- **64 凸包版本**：穿模依旧明显。提高分块数、减轻质量、增大摩擦、提高 `noslip_iterations` 后仍未解决，说明主要矛盾不是摩擦不足，而是**凸包网格碰撞对该抓取任务的接触面表达不可靠**。

### 当前结论
VHACD/多凸包 mesh 路线不适合作为当前 Panda 夹爪抓取圆环的主方案。后续稳定抓取实验建议优先使用：

- `Src/testscene.py`：20 凸包扇区分解，已能稳定运行。
- `Src/testscene2.py`：32-capsule 生成 mesh 后再分解，已能稳定运行。

`Src/testvhacd.py` 建议保留为对比实验，用于说明“更细的凸包数量并不一定能解决穿模”，尤其是在夹爪这种小接触区域、强挤压接触的场景中。

---

## 常见现象与排查建议（针对“夹不起来 / 提不起来”）

- **视觉看起来一样**：视觉 mesh 通常不变（例如 `torus_visual.obj`），变化的是碰撞体；要确认碰撞体数量可在 XML 中看 `torus_col*` 数量。
- **质量异常偏大**：检查视觉 geom 是否意外参与质量（例如缺少 `density="0"` 或 `mass="0"`）。
- **“夹住但提不稳/易滑落”**：优先尝试提高
  - `condim`（从 4 到 6）
  - `noslip_iterations`
  - `friction`（尤其 torsional/rolling 分量）
  - 同时注意约束数过大可能导致性能或稳定性问题。
- **VHACD/凸包版本持续穿模**：如果 8/32/64 凸包都穿模，继续增加摩擦通常帮助有限。应优先回到解析几何（capsule）或经过验证的扇区分解方案，而不是继续堆高 VHACD 凸包数量。

