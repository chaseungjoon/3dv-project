# Action conventions measured from held-out data

Fit `dpos(t+lag) = A a_xyz(t) + b` (tools/conventions.py). Scale = singular values of A (metres per action unit); frame = rotation angle of the polar factor of A.

| dataset | embodiment | Hz | best lag | R^2 | scale (m/unit) | frame rot (deg) | step mm (median / p99) | p99 speed mm/s | gripper corr |
|---|---|---|---|---|---|---|---|---|---|
| bridge_dataset | widowx | 5 | 0 | 1.0 | 1.0000, 1.0000, 1.0000 | 0.0 | 13.25 / 52.95 | 264.7 | 0.826 |
| fractal20220817_data | google_robot | 3 | 1 | 0.7722 | 0.2893, 0.2644, 0.1877 | 3.43 | 15.97 / 82.02 | 246.1 | -0.712 |
| taco_play | franka | 15 | 2 | 0.7523 | 0.0165, 0.0138, 0.0119 | 5.98 | 5.05 / 22.34 | 335.1 | -0.108 |
