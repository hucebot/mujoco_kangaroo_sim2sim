# Copyright (c) 2025, Fabio Amadio (fabioamadio93@gmail.com),
# Lorenzo Uttini (uttini.lorenzo@gmail.com).

# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import yaml
import torch
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_kangaroo_sim2sim.utils import (
    compute_cmd_vel,
    foh,
    get_ids,
    quat_rotate_inverse,
    set_tracking_viewer,
    squash_action,
)
import mujoco_kangaroo_sim2sim


class KangarooPolicyRunner:
    """
    A class to simulate a Kangaroo in MuJoCo and run a locomotion policy.
    """

    def __init__(self, policy_path: str, device: str) -> None:
        """
        Initialize the KangarooPolicyRunner.

        Args:
            policy_path (str): Path to the policy file (.pt or .jit).
            device (str): desired torch device ("cpu", "cuda")
        """
        self.pkg_path = os.path.dirname(mujoco_kangaroo_sim2sim.__file__)
        self.device = device

        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(
            f"{self.pkg_path}/kangaroo_mujoco/scene.xml"
        )
        self.sim_data = mujoco.MjData(self.mj_model)

        # Simulation and control parameters
        dt_sim = self.mj_model.opt.timestep
        dt_policy = 0.02
        self.hold_steps = int(dt_policy / dt_sim)

        # Load policy
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # Define offset for policy tanh-clamping
        self.clamp_offset = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, -0.54, -0.54, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )

        self.load_joint_ids()
        self.load_joint_cfg()

    def load_joint_cfg(self) -> None:
        """
        Load joint configuration parameters from a YAML file.

        Includes motor position limits and initial positions.
        """
        with open(f"{self.pkg_path}/config/joints.yaml", "r") as f:
            self.joint_config = yaml.safe_load(f)

        self.motor_max = torch.tensor(
            self.joint_config["motor_qpos_max"],
            dtype=torch.float32,
            device=self.device,
        )
        self.motor_min = torch.tensor(
            self.joint_config["motor_qpos_min"],
            dtype=torch.float32,
            device=self.device,
        )
        self.init_qpos = np.array(self.joint_config["init_qpos"])
        self.init_motor_cmd = np.array(self.joint_config["init_motor_cmd"])

    def load_joint_ids(self) -> None:
        """
        Load MuJoCo joint indices and define Isaac-to-MuJoCo actuator mappings.
        """
        motor_joint_names = [
            "leg_left_1_motor",
            "leg_right_1_motor",
            "leg_left_2_motor",
            "leg_left_3_motor",
            "leg_right_2_motor",
            "leg_right_3_motor",
            "leg_left_4_motor",
            "leg_left_5_motor",
            "leg_left_length_motor",
            "leg_right_length_motor",
            "leg_right_4_motor",
            "leg_right_5_motor",
        ]
        meas_joint_names = [
            "leg_left_1_joint",
            "leg_right_1_joint",
            "leg_left_2_joint",
            "leg_right_2_joint",
            "leg_left_3_joint",
            "leg_right_3_joint",
            "left_ankle_4_pendulum_joint",
            "left_ankle_5_pendulum_joint",
            "right_ankle_4_pendulum_joint",
            "right_ankle_5_pendulum_joint",
        ]
        _, self.motor_ids_qvel, self.motor_ids_qpos = get_ids(
            self.mj_model, motor_joint_names
        )
        _, self.meas_ids_qvel, self.meas_ids_qpos = get_ids(
            self.mj_model, meas_joint_names
        )
        # Map motor joint indices from IsaacSim to MuJoCo order
        self.map_motor_ids = {
            0: 0,
            1: 6,
            2: 1,
            3: 2,
            4: 7,
            5: 8,
            6: 4,
            7: 5,
            8: 3,
            9: 9,
            10: 10,
            11: 11,
        }

    def get_obs(
        self, last_action: torch.Tensor, cmd_vel: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct the observation vector for the policy input.

        Args:
            last_action (torch.Tensor): The last action applied by the policy.
            cmd_vel (torch.Tensor): Desired velocity [x, y, heading].

        Returns:
            torch.Tensor: Observation vector.
        """
        base_ang_vel = torch.tensor(
            self.sim_data.qvel[3:6], dtype=torch.float32, device=self.device
        )
        base_lin_vel_w = torch.tensor(
            self.sim_data.qvel[0:3], dtype=torch.float32, device=self.device
        )
        base_quat_w = torch.tensor(
            self.sim_data.qpos[3:7], dtype=torch.float32, device=self.device
        )
        base_lin_vel = quat_rotate_inverse(base_quat_w, base_lin_vel_w)

        gravity_tensor = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device
        )
        projected_gravity = quat_rotate_inverse(
            base_quat_w, gravity_tensor
        ).squeeze(0)
        qpos_motor = torch.tensor(
            self.sim_data.qpos[self.motor_ids_qpos],
            dtype=torch.float32,
            device=self.device,
        )
        qvel_motor = torch.tensor(
            self.sim_data.qvel[self.motor_ids_qvel],
            dtype=torch.float32,
            device=self.device,
        )
        qpos_meas = torch.tensor(
            self.sim_data.qpos[self.meas_ids_qpos],
            dtype=torch.float32,
            device=self.device,
        )
        qvel_meas = torch.tensor(
            self.sim_data.qvel[self.meas_ids_qvel],
            dtype=torch.float32,
            device=self.device,
        )
        obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                cmd_vel,
                qpos_motor,
                qvel_motor,
                qpos_meas,
                qvel_meas,
                last_action,
            ]
        )
        return obs

    def run_policy(
        self,
        num_steps: int = 1000,
        x_vel_target: float = 0.5,
        y_vel_target: float = 0.0,
        heading_target: float = 0.0,
    ) -> None:
        """
        Run the closed-loop control of the robot using the learned policy.

        Args:
            num_steps (int): Number of control steps to simulate.
            x_vel_target (float): Desired forward velocity (m/s).
            y_vel_target (float): Desired lateral velocity (m/s).
            heading_target (float): Desired heading angle (radians).
        """
        with mujoco.viewer.launch_passive(
            self.mj_model, self.sim_data
        ) as viewer:
            with viewer.lock():
                set_tracking_viewer(self.mj_model, viewer, "torso")

            # Initialize simulation
            self.sim_data.qpos[:] = self.init_qpos
            for i in range(12):
                self.sim_data.ctrl[self.map_motor_ids[i]] = (
                    self.init_motor_cmd[i]
                )
            motor_cmd = self.init_motor_cmd.copy()
            action = torch.zeros(
                self.mj_model.nu, dtype=torch.float32, device=self.device
            )

            # Loop over policy updates
            for i in range(num_steps):
                quat_w = torch.tensor(
                    self.sim_data.qpos[3:7],
                    dtype=torch.float32,
                    device=self.device,
                )
                cmd_vel = compute_cmd_vel(
                    x_vel_target,
                    y_vel_target,
                    quat_w,
                    heading_target,
                )

                obs = self.get_obs(action, cmd_vel)

                with torch.no_grad():
                    action = self.policy(obs)

                # Save last motor commands for FOH filtering
                last_motor_cmd = motor_cmd.copy()

                motor_cmd = (
                    squash_action(
                        action,
                        self.clamp_offset,
                        self.motor_max,
                        self.motor_min,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                )

                # Filter references via FOH in-between policy updates
                for k in range(self.hold_steps):
                    for test_joint_index in range(12):
                        old_ref: float = last_motor_cmd[test_joint_index]
                        new_ref: float = motor_cmd[test_joint_index]
                        filt_ref: float = foh(
                            new_ref, old_ref, self.hold_steps, k
                        )

                        self.sim_data.ctrl[
                            self.map_motor_ids[test_joint_index]
                        ] = filt_ref

                    mujoco.mj_step(self.mj_model, self.sim_data)

                if viewer.is_running():
                    viewer.sync()
