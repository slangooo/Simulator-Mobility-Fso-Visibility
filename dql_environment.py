from main_controller import SimulationController
import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from parameters import STARTING_SOLAR_HOUR, STARTING_SOLAR_MINUTE, FSO_MAX_DISTANCE, MACRO_BS_LOCATION
from resources.math_utils import line_point_angle
from resources.data_structures import Coords
from tf_agents.networks import q_network
from tf_agents.networks import network

PRINT_FLAG = True


class BackhaulEnv(py_environment.PyEnvironment):
    def __init__(self, simulation_controller):
        self.simulation_controller = simulation_controller
        self.simulation_controller.generate_environment_model(generate_random_bs=False)
        self.last_bs = self.simulation_controller.set_macro_bs(MACRO_BS_LOCATION[0], MACRO_BS_LOCATION[1])
        self.end_location = self.simulation_controller.generate_random_bs_destination()
        self.initial_distance = self.last_bs.coords.get_distance_to(self.end_location, flag_2d=True)
        # self.end_location = Coords(0, 400)  # for testing

        self.simulation_controller.mobility_model.set_time_of_day(STARTING_SOLAR_HOUR, STARTING_SOLAR_MINUTE)
        self.last_location_margin = 20  # m
        # [self.check_reached_destination(), self.check_last_backhaul_failure(),
        # self.check_last_sunlight_failure(), self.check_last_bs_collision(), self.check_boundary_violation()]
        self.rewards = np.array([50, -10, -1, -0, -50], dtype=np.int32)
        self.new_bs_penalty = -0
        self._current_time_step = None
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=0, maximum=1, name='observation')
        self._episode_ended = False
        self.state_scale = np.array((self.simulation_controller.max_xy[0] - self.simulation_controller.min_xy[0],
                                     self.simulation_controller.max_xy[1] - self.simulation_controller.min_xy[1]),
                                    dtype=np.float32)
        self.state_min_values = np.array((self.simulation_controller.min_xy[0], self.simulation_controller.min_xy[1]),
                                         dtype=np.float32)

        self._update_state()

    def scale_state(self, unscaled_state):
        return (unscaled_state - self.state_min_values.repeat(2)) / self.state_scale.repeat(2)

    def unscale_state(self, scaled_state):
        return scaled_state * self.state_scale.repeat(2) + self.state_min_values.repeat(2)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.simulation_controller.mobility_model.set_time_of_day(STARTING_SOLAR_HOUR, STARTING_SOLAR_MINUTE)
        self.simulation_controller.reset_model()
        self.last_bs = self.simulation_controller.macro_bs
        self.end_location = self.simulation_controller.generate_random_bs_destination()
        self._update_state()
        return ts.restart(self._state)

    def _update_state(self):
        self._state = self.scale_state(np.array([self.last_bs.coords.x,
                                                 self.last_bs.coords.y,
                                                 self.end_location.x,
                                                 self.end_location.y], dtype=np.float32))

        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if 0 <= action[0] <= 1 and 0 <= action[1] <= 1:
            self.add_new_bs(action)
        else:
            raise ValueError('`action` value error')

        objectives_flags = self.check_episode_end()
        self._episode_ended = any(objectives_flags * [True, False, False, False, True])

        if PRINT_FLAG:
            print(self.rewards[objectives_flags], action)
        reward = self.rewards[objectives_flags].sum() + self.new_bs_penalty

        self._update_state()

        if self._episode_ended:
            proximity_reward = 50 * (- self.last_bs.coords.get_distance_to(self.end_location, flag_2d=True)) \
                               / self.initial_distance
            reward += proximity_reward
            if PRINT_FLAG:
                print("Episode Ended: ", proximity_reward, "End Reward =", reward)
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def add_new_bs(self, action):
        """action: azimuth, distance"""
        azimuth = action[0] * 360
        distance = (action[1] + 0.1) * FSO_MAX_DISTANCE

        new_location = line_point_angle(distance, [self.last_bs.coords.x, self.last_bs.coords.y], angle_x=azimuth)
        self.simulation_controller.add_bs_station(new_location[0], new_location[1],
                                                  backhaul_bs_id=self.last_bs.base_station_id)

        self.last_bs = self.simulation_controller.base_stations[-1]

    def check_episode_end(self):
        return np.asarray([self.check_reached_destination(), self.check_last_backhaul_failure(),
                           self.check_last_sunlight_failure(), self.check_last_bs_collision(),
                           self.check_boundary_violation()])

    def check_boundary_violation(self):
        return self.simulation_controller.min_xy[0] > self.last_bs.coords.x or \
               self.simulation_controller.min_xy[1] > self.last_bs.coords.y or \
               self.simulation_controller.max_xy[0] < self.last_bs.coords.x or \
               self.simulation_controller.max_xy[1] < self.last_bs.coords.y

    def check_reached_destination(self):
        # print(self.end_location)
        return self.last_bs.coords.get_distance_to(self.end_location, flag_2d=True) < self.last_location_margin

    def check_last_backhaul_failure(self):
        return self.simulation_controller.is_backhaul_blocked(self.last_bs)

    def check_last_sunlight_failure(self):
        return self.last_bs.is_shadowed_flag

    def check_last_bs_collision(self):
        return self.simulation_controller.check_bs_obstacle_collision(self.last_bs)

    def reset(self):
        """Return initial_time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Apply action and return new time_step."""
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step


def get_tf_environment():
    sim_controller = SimulationController(False)
    bckhl_env = BackhaulEnv(sim_controller)
    # bckhl_env = tf_py_environment.TFPyEnvironment(bckhl_env)
    bckhl_env.reset()
    return bckhl_env


if __name__ == '__main__':
    sim_controller = SimulationController(False)
    bckhl_env = BackhaulEnv(sim_controller)
    bckhl_env = tf_py_environment.TFPyEnvironment(bckhl_env)
    time_step = bckhl_env.reset()

    data_spec = (bckhl_env.action_spec(), bckhl_env.observation_spec())
    batch_size = 32
    max_length = 1000

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec,
        batch_size=batch_size,
        max_length=max_length)

    add_new_location_action = tf.convert_to_tensor(np.array([[0.25, 0.25]], dtype=np.float32))
    time_step = bckhl_env.step(add_new_location_action)
    # actions = add_new_location_action* np.ones(data_spec[0].shape.as_list(), dtype=np.float32)

    values = (tf.squeeze(add_new_location_action), tf.squeeze(time_step.observation))
    values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size), values)

    replay_buffer.add_batch(values_batched)

    actor_network = network.Network(
        bckhl_env.time_step_spec().observation, name='ActorNetwork')

    # bckhl_env.unscale_state(time_step.observation)
    #
    # time_step = bckhl_env.step([0.3, 0.5])
