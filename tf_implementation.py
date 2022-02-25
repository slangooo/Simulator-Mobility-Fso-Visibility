import base64
import matplotlib.pyplot as plt
import os
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from dql_environment import BackhaulEnv, get_tf_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver

if __name__ == '__main__':
    tempdir = tempfile.gettempdir()

    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    num_iterations = 100000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 10000  # @param {type:"integer"}

    batch_size = 256  # @param {type:"integer"}

    critic_learning_rate = 3e-4  # @param {type:"number"}
    actor_learning_rate = 3e-4  # @param {type:"number"}
    alpha_learning_rate = 3e-4  # @param {type:"number"}
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.99  # @param {type:"number"}
    reward_scale_factor = 1.0  # @param {type:"number"}

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 5000  # @param {type:"integer"}

    num_eval_episodes = 20  # @param {type:"integer"}
    eval_interval = 10000  # @param {type:"integer"}

    policy_save_interval = 5000  # @param {type:"integer"}

    env = get_tf_environment()
    num_states = env.observation_spec().shape[0]
    print('Observation Spec:')
    print(env.time_step_spec().observation)
    print('Action Spec:')
    print(env.action_spec())

    collect_env = get_tf_environment()
    eval_env = get_tf_environment()

    use_gpu = True
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))

    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step)

        tf_agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity)

    # Add an observer that adds to the replay buffer:
    replay_observer = replay_buffer.add_batch

    dataset = replay_buffer.as_dataset().prefetch(50)
    experience_dataset_fn = lambda: dataset

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec())

    # initial_collect_actor = actor.Actor(
    #     collect_env,
    #     random_policy,
    #     train_step,
    #     steps_per_run=initial_collect_steps,
    #     observers=[replay_observer])
    # initial_collect_actor.run()

    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        collect_env,
        random_policy,
        observers=[replay_observer],
        num_episodes=50).run()

    env_step_metric = py_metrics.EnvironmentSteps()
    # collect_actor = actor.Actor(
    #     collect_env,
    #     collect_policy,
    #     train_step,
    #     steps_per_run=1,
    #     metrics=actor.collect_metrics(10),
    #     summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    #     observers=[replay_observer, env_step_metric])

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        collect_env,
        collect_policy,
        observers=[replay_observer],
        num_episodes=50)
    #
    # eval_actor = actor.Actor(
    #     eval_env,
    #     eval_policy,
    #     train_step,
    #     episodes_per_run=num_eval_episodes,
    #     metrics=actor.eval_metrics(num_eval_episodes),
    #     summary_dir=os.path.join(tempdir, 'eval'),
    # )
    #
    dynamic_episode_driver.DynamicEpisodeDriver(
        eval_env,
        eval_policy,
        num_episodes=50)

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=strategy)


    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]


    def collect_episode(environment, policy, num_episodes):
        driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(
                policy, use_tf_function=True),
            [replay_observer],
            max_episodes=num_episodes)
        initial_time_step = environment.reset()
        driver.run(initial_time_step)


    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        collect_env, tf_agent.collect_policy, 50)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)