# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import math
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import to_torch, tensor_clamp, quat_conjugate, quat_to_angle_axis, quat_mul, quat_axis
from isaacgymenvs.tasks.base.vec_task import VecTask
import gym

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaToy(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.box_pos_noise = self.cfg["env"]["boxPosNoise"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        if self.control_type == "admittance" or self.control_type == "position":
            self.n_control_loop = self.cfg["env"]["nControlLoop"]
        else:
            self.n_control_loop = 1 # This is not used

        # obs include: eef_pose (2) + eef_vel (2) + eef_force (2) + box_pose (2) + box_vel (2) + box_orientation (2) + box_angular_vel (1)
        self.cfg["env"]["numObservations"] = 13
        # actions include: delta EEF (2) + control params (2) for admittance control, delta EEF (2) for position control and osc control
        if self.control_type == "admittance":
            self.cfg["env"]["numActions"] = 4
            self.integration_var = self.cfg["env"]["integrationVar"]
        elif self.control_type == "position":
            self.cfg["env"]["numActions"] = 2
        elif self.control_type == "osc":
            self.cfg["env"]["numActions"] = 2
        else:
            raise ValueError("Invalid control type specified. Must be one of: {osc, admittance, position}")

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._error = None                      # x_d(t) - x_r(t) for admittance control
        self._error_dot = None                  # x_d'(t) - x_r'(t)(=0) for admittance control
        self.x_r_pos_prev = None                # x, y component of x_r(t-1) is required when integrationVar is x_d
        self.distance_prev = None               # 2D distance between box and target at t-1. Required for reward calculation

        # Tensor placeholders
        self._root_state = None                 # State of root body        (n_envs, 13)
        self._dof_state = None                  # State of all joints       (n_envs, n_dof)
        self._q = None                          # Joint positions           (n_envs, n_dof)
        self._qd = None                         # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None           # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._eef_state = None                  # end effector state
        self._j_eef = None                      # Jacobian for end effector
        self._mm = None                         # Mass matrix
        self._effort_control = None             # Torque actions
        self._franka_effort_limits = None         # Actuator effort limits for franka
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.friction_min, self.friction_max = self.cfg["env"]["minfriction"], self.cfg["env"]["maxfriction"]
        self.inertia_min, self.inertia_max = self.cfg["env"]["mininertia"], self.cfg["env"]["maxinertia"]
        self.stiffness_min, self.stiffness_max = self.cfg["env"]["minstiffness"], self.cfg["env"]["maxstiffness"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch([- (1 / 4) * np.pi, (1 / 8) * np.pi, 0, - (6 / 8) * np.pi, 0, 2.9416, 0], device=self.device)

        # OSC Gains
        self.kp = to_torch([100., 100., 1000, 1000, 1000, 1000], device=self.device) # More weight on z-axis and orientation
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim_params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"]["assetRoot"])
        franka_asset_file = self.cfg["env"]["asset"]["assetFileNameFranka"]

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[2.4, 2.4, table_thickness], table_opts)


        # Create box asset
        box_size = 0.1
        box_density = 1000
        self._box_init_pos = [0.0, 0.0, 1.0 + table_thickness / 2 + box_size / 2]
        box_opts = gymapi.AssetOptions()
        box_opts.density = box_density
        box_asset = self.gym.create_box(self.sim, *[box_size] * 3, box_opts)
        box_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Set target position, create a target asset
        self._target_pos = [0.8, 0, 1.0]
        target_size = 0.1
        target_opts = gymapi.AssetOptions()
        target_opts.fix_base_link = True
        target_asset = self.gym.create_box(self.sim, *[target_size, target_size, table_thickness+1e-3], target_opts)
        target_color = gymapi.Vec3(0.0, 0.6, 0.1)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)

        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []

        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT

            franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            franka_dof_props['damping'][i] = franka_dof_damping[i]

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for box
        box_start_pose = gymapi.Transform()
        box_start_pose.p = gymapi.Vec3(*self._box_init_pos)
        box_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for target
        target_start_pose = gymapi.Transform()
        target_start_pose.p = gymapi.Vec3(*self._target_pos)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # The reference code uses aggregate mode for performance reasons, but we omit it here for simplicity

        self.envs = []
    
        # Create environments
        friction = torch.linspace(self.friction_min, self.friction_max, num_envs, device=self.device)
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create franka
            franka_actor_handle = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor_handle, franka_dof_props)

            # Create table
            table_actor_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            t_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor_handle)
            # set coeffecient of friction of the table to 0 for easy debugging (Effective friction coefficient is 0.5*box_friction)
            t_shape_props[0].friction = 0.
            t_shape_props[0].rolling_friction = 0. # Not sure if this is used
            t_shape_props[0].torsion_friction = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor_handle, t_shape_props)
            
            # Create box
            """
            @@@@@@@@@@@@@@@@@@@@@@@@@@@Problem@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            We can set friction here (_create_envs()), but not in reset_idx()  
            because set_actor_rigid_shape_properties() has no effect after calling gym.prepare_sim().
            """
            box_actor_handle = self.gym.create_actor(env_ptr, box_asset, box_start_pose, "box", i, 2, 0)
            b_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, box_actor_handle)
            # Box actor has only one rigid shape
            b_shape_props[0].friction = 2 * friction[i]
            b_shape_props[0].rolling_friction = 2 * friction[i] # Not sure if this is used
            b_shape_props[0].torsion_friction = 2 * friction[i]
            self.gym.set_actor_rigid_shape_properties(env_ptr, box_actor_handle, b_shape_props)
            # Set color
            self.gym.set_rigid_body_color(env_ptr, box_actor_handle, 0, gymapi.MESH_VISUAL, box_color)

            # Create target
            target_actor_handle = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "target", i, 3, 1)
            self.gym.set_rigid_body_color(env_ptr, target_actor_handle, 0, gymapi.MESH_VISUAL, target_color)
            # Store the created env pointers
            self.envs.append(env_ptr)
            
        # # Setup data
        self.init_data()


    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_actor_handle = self.gym.find_actor_handle(env_ptr, "franka")
        box_actor_handle = self.gym.find_actor_handle(env_ptr, "box")
        self.handles = {
            "franka_fingertip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_handle, "panda_fingertip"),
            "box": self.gym.find_actor_rigid_body_handle(env_ptr, box_actor_handle, "box"),
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._contact_forces = gymtorch.wrap_tensor(_contact_forces_tensor).view(self.num_envs, -1, 3)

        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["franka_fingertip"], :]
        self._box_state = self._root_state[:, box_actor_handle, :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_actor_handle)['panda_finger_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]

        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]


        # Initialize actions
        self._effort_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._error = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self._error_dot = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.x_r_pos_prev = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        self.distance_prev = torch.zeros((self.num_envs), dtype=torch.float, device=self.device) # Set to 0 for now, will be initialized in reset_idx

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32, device=self.device).view(self.num_envs, -1) # 4 is the number of actors in each env

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_force": self._contact_forces[:, self.handles['franka_fingertip']],
            # Box
            "box_pos": self._box_state[:, :3],
            "box_quat": self._box_state[:, 3:7],
            "box_vel": self._box_state[:, 7:],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        # Compute current 2D distance between box and target
        distance = torch.norm(self.states["box_pos"][:, :2] - torch.tensor(self._target_pos[:2], device=self.device), dim=-1)
        delta_distance = distance - self.distance_prev
        self.distance_prev = distance
        reward = -delta_distance/self.dt/self.n_control_loop
        # Success when the box is close to the target & velocity is zero
        is_terminal = (distance < 0.1) & (torch.norm(self.states["box_vel"][:, :2], dim=-1) < 1e-3)
        reward += torch.where(is_terminal, torch.tensor(100., device=self.device), torch.tensor(0., device=self.device))
        self.rew_buf[:] = reward
        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1) | is_terminal, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        self._refresh()
        obs = ["eef_pos", "eef_vel", "eef_force", "box_pos", "box_vel"]
        self.obs_buf = torch.cat([self.states[ob][:, :2] for ob in obs], dim=-1) # Only take x, y components
        box_orientation_2D = self.states["box_quat"][:, 2:4] # Only take z, w components
        box_angular_vel_2D = self.states["box_vel"][:, 5] # Only take z component
        self.obs_buf = torch.cat([self.obs_buf, box_orientation_2D, box_angular_vel_2D.unsqueeze(-1)], dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 7), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        self._error[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)
        self._error_dot[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)
        self.x_r_pos_prev[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32)
        )

        # Randomize the box position & orientation, set the velocity to 0
        # 1. Set position
        pos = torch.zeros((len(env_ids), 3), device=self.device)
        box_init_pos = torch.tensor(self._box_init_pos, device=self.device)
        pos[:, :2] = box_init_pos[:2] + self.box_pos_noise * 2.0 * (torch.rand((len(env_ids), 2), device=self.device) - 0.5)
        pos[:, 2] = box_init_pos[2]
        self._box_state[env_ids, :3] = pos
        # 2. Set orientation
        axis_angle = torch.zeros((len(env_ids), 3), device=self.device)
        axis_angle[:, 2] = math.pi * (2 * torch.rand((len(env_ids),), device=self.device) - 1) # rotate around z axis by -pi to pi
        quat = axisangle2quat(axis_angle)
        self._box_state[env_ids, 3:7] = quat
        # 3. Set velocity
        self._box_state[env_ids, 7:] = torch.zeros((len(env_ids), 6), device=self.device)

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, 2].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        # Initialize distance_prev
        self.distance_prev[env_ids] = torch.norm(self._box_state[env_ids, :2] - torch.tensor(self._target_pos[:2], device=self.device), dim=-1)

        # Set the progress and reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        try:
            m_eef = torch.inverse(m_eef_inv)
        except:
            m_eef_inv += 1e-3 * torch.eye(6, device=self.device).unsqueeze(0)
            m_eef = torch.linalg.pinv(m_eef_inv, hermitian=True)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * ((self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1), -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def _convert_2D_to_3D(self, d_pose_2d):
        """
        convert 2D delta pose given as (delta x, delta y) to 3D delta pose (delta x, delta y, delta z, delta orientation)
        delta orientation is represented as axis-angle
        delta z is calculated to match the box center, delta orientation is calculated to match the goal orientation, which is Rot(\hat{x}, \pi)
        """
        num_envs = d_pose_2d.shape[0]
        eef_z_axis = quat_axis(self._eef_state[:, 3:7].clone(), 2)
        minus_z_axis = torch.tensor([0., 0., -1.], device=self.device).repeat(num_envs, 1)
        axis = torch.cross(eef_z_axis, minus_z_axis)
        angle = torch.acos(torch.sum(eef_z_axis * minus_z_axis, dim=-1))
        d_pose_3d = torch.zeros((num_envs, 6), device=self.device)
        d_pose_3d[:, :2] = d_pose_2d.clone()
        d_pose_3d[:, 2] = torch.tensor(self._box_init_pos[2], device=self.device).unsqueeze(-1) - self._eef_state[:, 2].clone()
        d_pose_3d[:, 3:] = angle.unsqueeze(-1) * axis
        return d_pose_3d


    def _delta_pose_to_pose(self, dpose):
        """
        delta pose: (delta x, delta y, delta z, delta orientation) where delta orientation is represented as axis-angle
        pose: (x, y, z, quaternion)
        """
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = (self.states["eef_pos"] + dpose[:, :3]).clone()
        dquat = axisangle2quat(dpose[:, 3:])
        pose[:, 3:] = quat_mul(dquat, self.states["eef_quat"])
        return pose


    def _pose_to_delta_pose(self, pose):
        """
        pose: (x, y, z, quaternion)
        delta pose: (delta x, delta y, delta z, delta orientation) where delta orientation is represented as axis-angle
        """
        delta_pose = torch.zeros((self.num_envs, 6), device=self.device)
        delta_pose[:, :3] = (pose[:, :3] - self.states["eef_pos"]).clone()
        angle, axis = quat_to_angle_axis(quat_mul(pose[:, 3:], quat_conjugate(self.states["eef_quat"])))
        delta_pose[:, 3:] = angle.unsqueeze(-1) * axis
        return delta_pose


    def pre_physics_step(self, actions):
        # Control arm (scale value first)
        self.actions = actions.clone().to(self.device)
        delta_x_r = self._convert_2D_to_3D(self.actions[:, :2].clone() * 0.2) # Scale the actions by 0.2
        if self.control_type == "position":
            x_r = self._delta_pose_to_pose(delta_x_r)
            for i in range(self.n_control_loop):
                # Use OSC for position control
                delta_x_r = self._pose_to_delta_pose(x_r)
                self._effort_control = self._compute_osc_torques(dpose=delta_x_r)
                # Deploy actions
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
                if i < self.n_control_loop - 1: # Skip the last step Since the last step will be done in step()
                    # step physics and render each frame
                    for i in range(self.control_freq_inv):
                        if self.force_render:
                            self.render()
                        self.gym.simulate(self.sim)
                        self._refresh()
                       
        elif self.control_type == "admittance":
            normalized_control_params = (self.actions[:, 2:].clone() + 1.0) / 2
            min_val = torch.tensor([self.inertia_min, self.stiffness_min], device=self.device)
            max_val = torch.tensor([self.inertia_max, self.stiffness_max], device=self.device)
            control_params = min_val * (max_val / min_val) ** normalized_control_params
            
            x_r = self._delta_pose_to_pose(delta_x_r)
            x_d = x_r.clone()
            x_r_pos = x_r[:, :2].clone() # (num_envs, 2)
            if self.integration_var == "x_d":
                self._error -= (x_r_pos - self.x_r_pos_prev)
            self.x_r_pos_prev = x_r_pos

            inertia = control_params[:, 0].clone().unsqueeze(-1)  # (num_envs, 1)
            stiffness = control_params[:, 1].clone().unsqueeze(-1)  # (num_envs, 1)
            damping = 2 * torch.sqrt(inertia * stiffness) # (num_envs, 1), Assume critical damping
            for i in range(self.n_control_loop):
                force = self.states['eef_force'][:, :2] # (num_envs, 2)
                # Solve x' = f(x), where x = [dpos, vel] 
                error_two_dot = (force - damping * self._error_dot - stiffness * self._error) / inertia
                self._error += self._error_dot * self.dt + error_two_dot * self.dt ** 2 / 2
                self._error_dot += error_two_dot * self.dt
                # Use OSC for position control
                x_d[:, :2] = x_r[:, :2] + self._error
                delta_x_d = self._pose_to_delta_pose(x_d)
                self._effort_control = self._compute_osc_torques(dpose=delta_x_d)
                # Deploy actions
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
                if i < self.n_control_loop - 1: # Skip the last step Since the last step will be done in step()
                    # step physics and render each frame
                    for i in range(self.control_freq_inv):
                        if self.force_render:
                            self.render()
                        self.gym.simulate(self.sim)
                        self._refresh()

        elif self.control_type == "osc":
            self._effort_control = self._compute_osc_torques(dpose=delta_x_r)

            # Deploy actions
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

