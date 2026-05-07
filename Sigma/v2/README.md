# Sigma v2 PSM Teleoperation

This package controls the MuJoCo dVRK PSM with a Force Dimension sigma.7.

Run from the project root:

```powershell
python -m Sigma.v2
```

The v2 controller has no keyboard motion path. It maps sigma.7 motion into:

- world-frame target position increments for `tool_tip`
- adaptive SVD-DLS position IK using only `yaw`, `pitch`, and `insertion`
- decoupled wrist increments for `roll`, `wrist_pitch`, and `wrist_yaw`
- optional sigma.7 gripper following for `jaw`

Files:

- `app.py`: MuJoCo viewer loop and lifecycle
- `configs.py`: command-line parameters
- `master.py`: sigma.7 device polling
- `motion.py`: split control law and adaptive DLS solver
- `robot.py`: MuJoCo PSM body, joint, actuator, and Jacobian helpers
- `__main__.py`: package entry point
