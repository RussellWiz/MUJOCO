# Sigma vp PyBullet PSM Teleoperation

This package mirrors the `Sigma/v2` split-control strategy, but uses PyBullet
instead of MuJoCo.

Run from the project root:

```powershell
python -m Sigma.vp
```

Default robot model:

```text
Assest/psm_official/psm1_sca_mujoco.urdf
```

Control strategy:

- sigma.7 translation increments become world-frame target increments
- `yaw`, `pitch`, and `insertion` solve `tool_tip` position with adaptive SVD-DLS
- PyBullet position Jacobian is computed by finite differences over the three position joints
- `roll`, `wrist_pitch`, and `wrist_yaw` are controlled independently from sigma.7 orientation increments
- `jaw` can follow the sigma.7 gripper or stay locked

Runtime dependencies:

- `pybullet`
- Force Dimension SDK DLLs
- a connected sigma.7 device
