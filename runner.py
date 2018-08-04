import gym
import json
from agents.dqnagent import DQNAgent
import tensorflow as tf
import numpy as np
import sys
import os

if __name__ == '__main__':

    # Load the config file
    conf = json.load(open('conf.json', 'r'))

    # Get the env name from args its config from conf
    env_name = sys.argv[1]

    conf_env = conf[env_name]
    num_trials = conf_env['num-trials']
    num_episodes = conf_env['num-episodes']
    opt_dict = conf['optimizer-set'][conf_env['optimizer']]     # Optimizer
    #print opt_dict
    update_frequency = conf_env['update-frequency']
    target_network_update_frequency = conf_env['target-network-update-frequency']

    # Get the environment from gym
    env = gym.make(env_name)
    env.seed(0)
    env.reset()

    os.chdir(os.getcwd())

    # Initialize a tensorflow session
    sess = tf.Session()

    rewards_matrix = np.zeros(num_episodes)

    # Start Q learning
    for T in range(num_trials):
        tf.reset_default_graph()
        agent = DQNAgent(env=env, conf_dict=conf_env, optimizer_dict=opt_dict, sess=tf.Session(), trial_number=T)
        rewards_over_episodes = []

        t = 0

        for e in range(num_episodes):
            terminal_state = False
            print("############################  RUNNING EPISODE {0}  ####################################".format(e))
            agent.reset_env()           # Reset environment at start of each episode
            total_episode_reward = 0

            while terminal_state is not True:
                agent.enable_env_render()

                # Get action every action-repeat length
                curr_state = agent.get_state()
                a_t = agent.get_action(curr_state)

                next_state, reward, terminal_state = agent.take_action(a_t)

                total_episode_reward = total_episode_reward + reward

                # Add experience occurs every skip-state length
                exp = agent.prepare_experience(curr_state, a_t, reward, next_state, terminal_state)
                agent.add_experience(exp)

                # Train occurs every update frequency length
                if t % update_frequency == 0:
                    #print("Training Q network now.....")
                    agent.train_q_networks()

                # Happens every target-network-update-frequency length
                if t % target_network_update_frequency == 0:
                   # print("Transferring weights from target network to Q network.....")
                    agent.transfer_weights_target_net()

                t = t + 1


            print("############################  EPISODE {0} REWARD : {1}  ####################################".format(e,total_episode_reward))
            rewards_over_episodes.append(total_episode_reward)

        rewards_matrix = np.vstack((rewards_matrix, rewards_over_episodes))

    np.save(conf_env['model_path'] + 'rewards_matrix_cpole.npy', rewards_matrix)
























