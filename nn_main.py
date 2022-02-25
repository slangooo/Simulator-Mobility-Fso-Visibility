from ddpg import *
from dql_environment import BackhaulEnv, get_tf_environment
import gym

RL_TASK= 'BipedalWalker-v3'
LOAD_CHECK_POINT = False

if __name__ == '__main__':
    # env = get_tf_environment()
    # num_states = env.observation_spec().shape[0]
    # print("Size of State Space ->  {}".format(num_states))
    # num_actions = env.action_spec().shape[0]
    # print("Size of Action Space ->  {}".format(num_actions))
    #
    # upper_bound = env.action_spec().maximum
    # lower_bound = env.action_spec().minimum

    env = gym.make(RL_TASK)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    agent = Agent(input_dims=[num_states], env=env, n_actions=num_actions, n_states=num_states)

    n_episodes = TOTAL_EPISODES

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    best_score = -100000

    if LOAD_CHECK_POINT:
        n_steps = 0
        while n_steps <= BATCH_SIZE:
            observation = env.reset()
            action = np.random.uniform(low=0.0, high=1.0, size=2)
            time_step = env.step(action)
            observation_, reward, done = time_step.observation, time_step.reward, time_step.is_last()
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1

        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for episode_number in range(n_episodes):
        time_step = env.reset()
        # observation = time_step.observation
        observation = time_step
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            # print(action)
            time_step = env.step(action)
            # observation_, reward, done = time_step.observation, time_step.reward, time_step.is_last()
            observation_, reward, done, info =time_step
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not LOAD_CHECK_POINT:
                agent.learn()
            observation = observation_

        ep_reward_list.append(score)
        avg_reward = np.mean(ep_reward_list[-100:])

        if avg_reward > best_score:
            best_score = avg_reward
            if not LOAD_CHECK_POINT:
                agent.save_models()
        print("Episode * {} * Avg Reward is ==> {}".format(episode_number, avg_reward))
