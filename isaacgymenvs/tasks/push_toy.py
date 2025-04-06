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


class PushToy(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.obs_force = self.cfg["env"]["obsForce"]
        self.init_box_radius = self.cfg["env"]["initboxradius"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"] # admittance, velocity
        self.n_control_loop = self.cfg["env"]["nControlLoop"]

        # actions include: delta EEF (2) + control params (2) for admittance control, delta EEF (2) for velocity control
        if self.control_type == "admittance":
            self.cfg["env"]["numActions"] = 4
            self.integration_var = self.cfg["env"]["integrationVar"]
        elif self.control_type == "velocity":
            self.cfg["env"]["numActions"] = 2
        else:
            raise ValueError("Invalid control type specified. Must be one of: {admittance, velocity}")
        
        # Full obs: eef_pose (2) + eef_vel (2) + eef_force (2) + previous actions
        self.cfg["env"]["numObservations"] = 4 + self.cfg["env"]["numActions"]
        if self.obs_force:
            self.cfg["env"]["numObservations"] += 2


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
        self.mass_min, self.mass_max = self.cfg["env"]["minmass"], self.cfg["env"]["maxmass"]
        self.friction_min, self.friction_max = self.cfg["env"]["minfriction"], self.cfg["env"]["maxfriction"]
        self.inertia_min, self.inertia_max = self.cfg["env"]["mininertia"], self.cfg["env"]["maxinertia"]
        self.stiffness_min, self.stiffness_max = self.cfg["env"]["minstiffness"], self.cfg["env"]["maxstiffness"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[2.4, 2.4, table_thickness], table_opts)

        # Create eef asset
        eef_radius = 0.05
        eef_width = 0.3
        self._eef_init_pos = [0.0, 0.0, 1.0 + table_thickness / 2 + eef_width + 0.05] # 0.05 margin
        eef_opts = gymapi.AssetOptions()
        eef_opts.density = 10e8 # Set density to a very high value to make it fixed
        eef_opts.disable_gravity = True
        eef_asset = self.gym.create_capsule(self.sim, eef_radius, eef_width, eef_opts)
        eef_color = gymapi.Vec3(0.0, 0.0, 0.6)
        eef_idx = self.gym.find_asset_rigid_body_index(eef_asset, "capsule")
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        self.ftsensor_idx = self.gym.create_asset_force_sensor(eef_asset, eef_idx, sensor_pose)

        # Create box asset
        box_size = 0.1
        self._box_init_pos = [0.0, 0.0, 1.0 + table_thickness / 2 + box_size / 2]
        box_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Set target position, create a target asset
        self._target_pos = [-0.1, 0.3, 1.0]
        target_size = 0.1
        target_opts = gymapi.AssetOptions()
        target_opts.fix_base_link = True
        target_asset = self.gym.create_box(self.sim, *[target_size, target_size, table_thickness+1e-3], target_opts)
        target_color = gymapi.Vec3(0.0, 0.6, 0.1)


        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for eef
        eef_start_pose = gymapi.Transform()
        eef_start_pose.p = gymapi.Vec3(*self._eef_init_pos)
        eef_start_pose.r = gymapi.Quat(0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)) # 90 degree rotation around y axis

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
        friction = torch.exp(torch.linspace(np.log(self.friction_min), np.log(self.friction_max), int(np.sqrt(num_envs)), device=self.device))
        mass = torch.exp(torch.linspace(np.log(self.mass_min), np.log(self.mass_max), int(np.sqrt(num_envs)), device=self.device))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create table
            table_actor_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            t_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor_handle)
            # set coeffecient of friction of the table to 0 for easy debugging (Effective friction coefficient is 0.5*box_friction)
            t_shape_props[0].friction = 0.
            t_shape_props[0].rolling_friction = 0. # Not sure if this is used
            t_shape_props[0].torsion_friction = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor_handle, t_shape_props)

            # Create eef
            eef_actor_handle = self.gym.create_actor(env_ptr, eef_asset, eef_start_pose, "eef", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, eef_actor_handle, 0, gymapi.MESH_VISUAL, eef_color)

            # Create box
            """
            @@@@@@@@@@@@@@@@@@@@@@@@@@@Problem@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            We can set friction here (_create_envs()), but not in reset_idx()  
            because set_actor_rigid_shape_properties() has no effect after calling gym.prepare_sim().
            """
            j = i // int(np.sqrt(num_envs))
            k = i % int(np.sqrt(num_envs))
            box_opts = gymapi.AssetOptions()
            box_density = 1000 * mass[j]
            box_opts.density = box_density
            box_asset = self.gym.create_box(self.sim, *[box_size] * 3, box_opts)

            box_actor_handle = self.gym.create_actor(env_ptr, box_asset, box_start_pose, "box", i, 2, 0)
            b_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, box_actor_handle)
            # Box actor has only one rigid shape
            b_shape_props[0].friction = 2 * friction[k]
            b_shape_props[0].rolling_friction = 2 * friction[k] # Not sure if this is used
            b_shape_props[0].torsion_friction = 2 * friction[k]
            self.gym.set_actor_rigid_shape_properties(env_ptr, box_actor_handle, b_shape_props)
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
        eef_actor_handle = self.gym.find_actor_handle(env_ptr, "eef")
        box_actor_handle = self.gym.find_actor_handle(env_ptr, "box")
        self.handles = {
            "eef": self.gym.find_actor_rigid_body_handle(env_ptr, eef_actor_handle, "capsule"),
            "box": self.gym.find_actor_rigid_body_handle(env_ptr, box_actor_handle, "box"),
        }

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._contact_forces = gymtorch.wrap_tensor(_contact_forces_tensor).view(self.num_envs, -1, 3)
        self._fsdata = gymtorch.wrap_tensor(_fsdata).view(self.num_envs, -1, 6)

        self._eef_state = self._root_state[:, eef_actor_handle, :]
        self._box_state = self._root_state[:, box_actor_handle, :]

        # Initialize actions
        self._error = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self._error_dot = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.x_r_pos_prev = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        self.distance_prev = torch.zeros((self.num_envs), dtype=torch.float, device=self.device) # Set to 0 for now, will be initialized in reset_idx

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32, device=self.device).view(self.num_envs, -1) # 4 is the number of actors in each env

    def _update_states(self):
        self.states.update({
            # eef
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_force": self._contact_forces[:, self.handles['eef']],
            "eef_ft": self._fsdata[:, self.ftsensor_idx, :],
            # Box
            "box_pos": self._box_state[:, :3],
            "box_quat": self._box_state[:, 3:7],
            "box_vel": self._box_state[:, 7:],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
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
        eef_obs = ["eef_pos", "eef_vel"]
        if self.obs_force:
            eef_obs.append("eef_force")
        self.obs_buf = torch.cat([self.states[ob][:, :2] for ob in eef_obs], dim=-1)
        self.obs_buf = torch.cat([self.obs_buf, self.actions], dim = -1)
        return self.obs_buf

    def reset_idx(self, env_ids):

        self._error[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)
        self._error_dot[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)
        self.x_r_pos_prev[env_ids, :] = torch.zeros((len(env_ids), 2), device=self.device)

        # Reset the EEF state
        self._eef_state[env_ids, :3] = torch.tensor(self._eef_init_pos, device=self.device)
        self._eef_state[env_ids, 3:7] = torch.tensor([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=torch.float, device=self.device) # 90 degree rotation around y axis
        self._eef_state[env_ids, 7:] = torch.zeros((len(env_ids), 6), device=self.device)

        # Randomize the box position & orientation, set the velocity to 0
        # 1. Set position
        pos = torch.zeros((len(env_ids), 3), device=self.device)
        radius = 0.1 + torch.rand((len(env_ids),), device=self.device) * (self.init_box_radius-0.1) # 0.1 to init_box_radius
        theta = 2 * math.pi * torch.rand((len(env_ids),), device=self.device)
        pos[:, 0] = radius * torch.cos(theta)
        pos[:, 1] = radius * torch.sin(theta)
        pos[:, 2] = self._box_init_pos[2]
        self._box_state[env_ids, :3] = pos
        # 2. Set orientation
        axis_angle = torch.zeros((len(env_ids), 3), device=self.device)
        axis_angle[:, 2] = math.pi * (2 * torch.rand((len(env_ids),), device=self.device) - 1) # rotate around z axis by -pi to pi
        quat = axisangle2quat(axis_angle)
        self._box_state[env_ids, 3:7] = quat
        # 3. Set velocity
        self._box_state[env_ids, 7:] = torch.zeros((len(env_ids), 6), device=self.device)

        # Update eef and cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, 1:3].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        # Initialize distance_prev
        self.distance_prev[env_ids] = torch.norm(self._box_state[env_ids, :2] - torch.tensor(self._target_pos[:2], device=self.device), dim=-1)

        # Set the progress and reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        # Control arm (scale value first)
        not_initial = (self.progress_buf != 0) # At t=0, the states are invalid, so we skip the control step.
        self.actions = actions.clone().to(self.device)
        v_xy_r = self.actions[:, :2].clone() 
        if self.control_type == "velocity":
            v_r = torch.zeros((self.num_envs, 6), device=self.device)
            v_r[:, :2] = v_xy_r
            self._eef_state[:, 7:] = v_r
            ids_eef_int32 = self._global_indices[not_initial, 1].flatten()
            if len(ids_eef_int32) > 0:
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(ids_eef_int32), len(ids_eef_int32))
            for i in range(self.n_control_loop):
                if i < self.n_control_loop - 1: # Skip the last step Since the last step will be done in step()
                    # step physics and render each frame
                    for i in range(self.control_freq_inv):
                        if self.force_render:
                            self.render()
                        self.gym.simulate(self.sim)
                        self._refresh()
                       
        elif self.control_type == "admittance":
            """
            TODO: Implement admittance control
            """
            pass


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

