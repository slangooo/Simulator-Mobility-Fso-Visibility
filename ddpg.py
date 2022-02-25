import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import random_uniform
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_normal

CHECKPOINTS_PATH = 'tmp/ddpg'

BATCH_SIZE = 200
BUFFER_SIZE = int(1e6)
TOTAL_EPISODES = 10000
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4
TAU = 0.001

AMSGRAD_FLAG = True

NOISE_STD = 0.2
NOISE_THETA = 0.15
NOISE_DT = 1e-2
NOISE_TYPE = 1  # 1 OUAction, 2 Gaussian

CRITIC_LOSS_TYPE = 2  # 1 MSE, 2 ABS
KERNEL_INITIALIZER = glorot_normal()
CRITIC_ACTIVATION = tf.nn.leaky_relu
ACTOR_ACTIVATION = tf.nn.leaky_relu


class OUActionNoise:
    def __init__(self, mean, std_deviation=NOISE_STD, theta=NOISE_THETA, dt=NOISE_DT, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = None
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt
             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    def __init__(self, state_fc1_dims=600, state_fc2_dims=300, action_fc1_dims=300,
                 fc1_dims=150, fc2_dims=150, name='critic', chkpt_dir=CHECKPOINTS_PATH, n_states=4, n_actions=2):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.state_fc1_dims = state_fc1_dims
        self.state_fc2_dims = state_fc2_dims
        self.action_fc1_dims = action_fc1_dims

        self.model_name = name
        self.num_states = n_states
        self.num_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.state_input = keras.layers.Input(shape=(self.num_states), dtype=tf.float32)
        self.state_fc1 = Dense(self.state_fc1_dims, activation=CRITIC_ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)
        self.state_fc2 = Dense(self.state_fc2_dims, activation=CRITIC_ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)
        self.action_input = keras.layers.Input(shape=(self.num_actions), dtype=tf.float32)
        self.action_fc1 = Dense(self.action_fc1_dims, activation=CRITIC_ACTIVATION,
                                kernel_initializer=KERNEL_INITIALIZER)
        self.fc1 = Dense(self.fc1_dims, activation=CRITIC_ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)
        if self.fc2_dims > 0:
            self.fc2 = Dense(self.fc2_dims, activation=CRITIC_ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)
        else:
            self.fc2 = None

        last_init = tf.random_normal_initializer(stddev=0.00005)
        self.q = Dense(1, kernel_initializer=last_init)

    def call(self, state, action):
        # state_input = self.state_input(state)
        state_out = self.state_fc1(state)
        state_out = keras.layers.BatchNormalization()(state_out)
        state_out = self.state_fc2(state_out)

        # action_input = self.action_input(action)
        action_out = self.action_fc1(action)

        added = keras.layers.Add()([state_out, action_out])
        added = keras.layers.BatchNormalization()(added)

        outs = self.fc1(added)
        if self.fc2:
            outs = self.fc2(outs)
        outs = keras.layers.BatchNormalization()(outs)
        q = self.q(outs)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=600, fc2_dims=600, n_actions=2, n_states=4, name='actor', chkpt_dir=CHECKPOINTS_PATH):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.num_states = n_states

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.state_input = keras.layers.Input(shape=(self.num_states), dtype=tf.float32)
        self.fc1 = Dense(self.fc1_dims, activation=ACTOR_ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)
        self.fc2 = Dense(self.fc2_dims, activation=ACTOR_ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)
        last_init = tf.random_normal_initializer(stddev=0.0005)
        self.mu = Dense(self.n_actions, activation='tanh', kernel_initializer=last_init)

    def call(self, state):
        # state_in = self.state_input(state)
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu


class Agent:
    def __init__(self, input_dims, alpha=ACTOR_LR, beta=CRITIC_LR, env=None, gamma=0.99, n_actions=2, n_states=4,
                 max_size=BUFFER_SIZE,
                 tau=TAU, batch_size=BATCH_SIZE, noise_std=NOISE_STD, max_action=1, min_action=-1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_states = n_states
        self.noise_std = noise_std
        self.max_action = max_action
        self.min_action = min_action

        self.noise_generator = OUActionNoise(mean=np.zeros(self.n_actions))

        self.actor = ActorNetwork(n_actions=n_actions, n_states=n_states, name='actor')
        self.critic = CriticNetwork(n_actions=n_actions, n_states=n_states, name='critic')

        self.target_actor = ActorNetwork(n_actions=n_actions, n_states=n_states, name='target_actor')
        self.target_critic = CriticNetwork(n_actions=n_actions, n_states=n_states, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha, amsgrad=AMSGRAD_FLAG))
        self.critic.compile(optimizer=Adam(learning_rate=beta, amsgrad=AMSGRAD_FLAG))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha, amsgrad=AMSGRAD_FLAG))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta, amsgrad=AMSGRAD_FLAG))

        self.update_network_parameters(tau=1)

        # # define update weights with tf.function for improved performance
        # @tf.function(
        #     input_signature=[
        #         tf.TensorSpec(shape=(None, n_states), dtype=tf.float32),
        #         tf.TensorSpec(shape=(None, n_actions), dtype=tf.float32),
        #         tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        #         tf.TensorSpec(shape=(None, n_states), dtype=tf.float32),
        #         tf.TensorSpec(shape=(None, 1), dtype=tf.float32), ])

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value) if CRITIC_LOSS_TYPE == 1 else \
                tf.math.reduce_mean(tf.abs(target - critic_value))

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

        # self.learn = learn

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        target = self.target_actor.weights

        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + target[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        target = self.target_critic.weights

        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + target[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... Saving Models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... Loading Models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += (self.noise_generator() if NOISE_TYPE == 1 else tf.random.normal(shape=[self.n_actions],
                                                                                        mean=0.0, stddev=self.noise_std))

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]
