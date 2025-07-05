# Copyright (c) 2025, Fabio Amadio (fabioamadio93@gmail.com),
# Lorenzo Uttini (uttini.lorenzo@gmail.com).

# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
import torch

from mujoco_kangaroo_sim2sim.policy_runner import KangarooPolicyRunner

# Initialize runner for the desired policy
policy_log_path = os.path.join(
    os.path.dirname(__file__), "policies/policy_A.pt"
)
runner = KangarooPolicyRunner(
    policy_log_path, "cuda" if torch.cuda.is_available() else "cpu"
)

# Define velocity target and run the policy
num_steps = 1000
x_vel_target = np.random.uniform(0.2, 0.8)
y_vel_target = np.random.uniform(-0.5, 0.5)
heading_target = np.random.uniform(-np.pi / 4, np.pi / 4)
runner.run_policy(num_steps, x_vel_target, y_vel_target, heading_target)
