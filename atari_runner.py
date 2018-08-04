import gym
import json
from agents.atariagent import AtariAgent
import tensorflow as tf
import numpy as np
import sys

if __name__ == '__main__':

    # Load the config file
    conf = json.load(open('conf.json', 'r'))

    # Get the env name from args its config from conf
    env_name = sys.argv[1]

    conf_env = conf[env_name]
    T = conf_env['T']
    opt_dict = conf['optimizer-set'][conf_env['optimizer']]     # Optimizer

    update_frequency = conf_env['update-frequency']
    target_network_update_frequency = conf_env['target-network-update-frequency']

    # Get the environment from gym
    env = gym.make(env_name).unwrapped
    env.seed(0)
    env.reset()

    # Initialize a tensorflow session
    sess = tf.Session()

    tf.reset_default_graph()

    agent = AtariAgent(env=env, conf_dict=conf_env, optimizer_dict=opt_dict, sess=tf.Session())

    rewards_over_episodes = []
    reward_this_episode = 0

    train_update_param = 0
    transfer_param = 0
    reward_save_param = 0

    mode = 'train'

    # Start Q learning
    if mode =='train':
        while agent.get_age() < T:
            # Get action every action-repeat length
            curr_state = agent.get_state()
            a_t = agent.get_action(curr_state)

            next_state, reward, terminal, consider_terminal_state = agent.take_action(a_t)

            if consider_terminal_state is False:
                reward_this_episode += reward
            else:
                rewards_over_episodes.append(reward_this_episode)
                print("T = {0}, Reward over episode {1} : {2}".format(agent.get_age(),len(rewards_over_episodes), reward_this_episode))
                reward_this_episode = 0

            #print("Reward at timestep {0} : {1}".format(agent.get_age(), reward))
            # Add experience occurs every skip-state length
            agent.add_experience(curr_state, a_t, reward, next_state, consider_terminal_state)

            # Train occurs every update frequency length
            if int(agent.get_age() / update_frequency) > train_update_param:
                #print("Training Q network now.....")
                agent.train_q_networks()
                train_update_param += 1

            # Happens every target-network-update-frequency length
            if int(agent.get_age() / target_network_update_frequency) > transfer_param:
                print("Transferring weights from target network to Q network.....")
                agent.transfer_weights_target_net()
                transfer_param += 1

            if int(agent.get_age() / 50000) > reward_save_param:
                np.save('rewards_{}.npy'.format(agent.get_age()), rewards_over_episodes)
                reward_save_param+=1
























