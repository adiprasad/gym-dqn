import numpy as np
import os
import sys
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from networks.mlp import MLP
from networks.network import Network
import tensorflow as tf
import random

class DQNAgent(object):
    def __init__(self, env, conf_dict, optimizer_dict, sess, trial_number):
        self._env = env
        self._conf = conf_dict
        self._sess = sess
        self._eps = conf_dict['eps-init']
        self._eps_min = conf_dict['eps-final']
        self._eps_decay = conf_dict['eps-decay']
        self._action_set_size = self._env.action_space.n
        self._state_set_size = self._env.observation_space.shape[0]

        #Seed
        random.seed(0)
        np.random.seed(0)

        self.trial_number = trial_number

        # Initialize prediction and target networks
        self.__init_pred_network(optimizer_dict)
        self.__init_target_network()

        self.__init_network_graphs()
        self._age_of_agent = 0                  # Number of time steps since the agent was initialized

        #Reset environment
        self.reset_env()

        # Experience format [prev_state, action, reward, curr_state, terminal?]
        self._experience = np.zeros(2 * self._state_set_size + 3)

        # Initialize experience using random policy
        self.init_experience()

        self.saver = tf.train.Saver()


    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def get_elapsed_time(self):
        return self._age_of_episode

    '''
    Return action using eps-greedy policy
    Random action within eps and through the network after eps
    '''
    def get_action(self, state):
        state = np.reshape(state, (1,state.shape[0]))

        #if self._age_of_agent >= self._conf['replay-start-size']:
        a = self.__eps_greedy_policy(state)
        # else:
        #     a = self.__random_policy()

        #if (self._age_of_agent >= self._conf['replay-start-size']) and (self._age_of_agent <= self._conf['exploration-time']):
        if self._age_of_agent <= self._conf['exploration-time']:
            self._eps -= (self._conf['eps-init'] - self._conf['eps-final'])/(self._conf['exploration-time'])

        return a

    '''
    Add current experience to the existing experience
    Trim the experiences to include the last replay-max-size experiences only 
    '''
    def add_experience(self, experience_blob):
        self._experience = np.vstack((self._experience, experience_blob))

        if len(self._experience) > self._conf['replay-max-size']:
            self._experience = self._experience[-self._conf['replay-max-size']:]

    '''
    Initialize experience using random policy
    according to init-experience-size
    '''
    def init_experience(self):
        for i in range(self._conf['replay-start-size']):
            state = self.get_state()
            a = self.__random_policy()
            next_state, reward, done = self.take_action(a)
            experience_blob = self.prepare_experience(state, a, next_state, reward, done)

            self.add_experience(experience_blob)

            if done:
                self.reset_env()

        self._experience = self._experience[1:]         # Cutting off the first row of zeros
        print("Experience initialized")
        self.transfer_weights_target_net()

        self.reset_env()  # Reset environment

    def prepare_experience(self, state, a, reward, next_state, done):
        experience_blob = np.array([])

        experience_blob = np.append(experience_blob, state)
        experience_blob = np.append(experience_blob, a)
        experience_blob = np.append(experience_blob, reward)
        experience_blob = np.append(experience_blob, next_state)
        experience_blob = np.append(experience_blob, done)

        return experience_blob


    # Takes action in the environment
    # Increases the time steps elapsed
    # Returns next state, reward, done
    def take_action(self, a):
        next_state, reward, terminal, _ = self._env.step(a)

        # Start counting only after experience replay has been populated
        if len(self._experience) >= self._conf['replay-start-size']:
            self._age_of_episode = self._age_of_episode + 1
            self._age_of_agent = self._age_of_agent + 1

        self.set_state(next_state)

        return next_state, reward, terminal


    '''
    Pick experience relays and train the network
    Also transfer the network weights from prediction network
    to target network according to target-network-update-frequency
    '''
    def train_q_networks(self):
        minibatch_exp = self.__sample_minibatch_exp()

        s_pred_net = minibatch_exp[:,0:self._state_set_size]

        a = minibatch_exp[:,self._state_set_size]
        a = self.__prepare_actions(a)           # Convert actions to be able to index the tensor inside

        rewards = minibatch_exp[:,self._state_set_size + 1]
        s_target_net = minibatch_exp[:,self._state_set_size + 2:2*self._state_set_size + 2]
        done = minibatch_exp[:,2*self._state_set_size + 2]

        # q_target = self._sess.run([self._target_net.out],
        #                           feed_dict = {self._target_net.s : s_target_net})

        q_target = self._target_net.calc_output(s_target_net, self._sess)
        #q_target = self._predict_net.calc_output(s_target_net, self._sess)              # Running without target networks

        y_targets = self.__prepare_targets(rewards, q_target, done)

        loss, step, summary, _ = self._sess.run([self._predict_net.loss, self._predict_net.global_step,
                                                 self._predict_net.summary_op, self._predict_net.learning_step],
                                                feed_dict={self._predict_net.s: s_pred_net,
                                                           self._predict_net.a: a,
                                                           self._predict_net.y: y_targets}
                                                )

        T = self._age_of_episode

        print("T = {0}, Loss = {1}, Minibatch update = {2}, Num Iterations = {3}, Trial {4}".format(T, loss, step, self.get_age(), self.trial_number))

        if step % 2000 == 0:
            self._predict_net.save_model(self.saver, self._sess, self._conf['model_path']
                                         + "trial_{0}/".format(self.trial_number) + "{0}".format(step))

    def __prepare_targets(self, rewards, q_target, done):

        gamma = self._conf['gamma']

        not_done_idx = np.where(done==0)[0]
        q_s_a_max = np.max(q_target, axis=1)

        targets = rewards
        targets[not_done_idx] = targets[not_done_idx] + gamma*q_s_a_max[not_done_idx]

        return targets

    def __prepare_actions(self, a):
        coordinates = zip(range(len(a)),a)
        list_coordinates = list(map(lambda x : list(x), coordinates))

        return list_coordinates

    '''
    Transfer the weights from prediction network to target network
    '''
    def transfer_weights_target_net(self):
        pred_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred-net')
        target_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target-net')

        # print pred_net_vars
        # print target_net_vars

        pred_net_dict = self.__return_weights_dict(pred_net_vars)
        target_net_dict = self.__return_weights_dict(target_net_vars)

        op_holder = []

        for key in target_net_dict.keys():
            op_holder.append(target_net_dict[key].assign(pred_net_dict[key].value()))


        # for key in target_net_dict.keys():
        #     assign_node = tf.assign(target_net_dict[key], pred_net_dict[key])
        #     self._sess.run(assign_node)

        self._sess.run(op_holder)

    def __return_weights_dict(self, vars):
        weights_dict = {}

        for var in vars:
            var_name = var.name
            var_name_array = var_name.split('/')
            var_name_without_scope = '/'.join(var_name_array[1:])
            weights_dict[var_name_without_scope] = var

        return weights_dict

    '''
    Returns a minibatch of experiences 
    '''
    def __sample_minibatch_exp(self):
        minibatch = np.random.choice(len(self._experience), self._conf['mini-batch-size'], replace=False)

        return self._experience[minibatch]

    '''
    Take action using a random policy
    '''
    def __random_policy(self):
        a = self._env.action_space.sample()

        return a

    '''
        Take action using epsillon greedy policy
    '''
    def __eps_greedy_policy(self, state):
        if random.random() < self._eps:
            a = random.randint(0, self._action_set_size -1)
        else:
            q_out = self._predict_net.calc_output(state, self._sess)

            a = np.argmax(q_out[0])

        return a

    def reset_env(self):
        init_state = self._env.reset()
        self._age_of_episode = 0


        self.set_state(init_state)

    def enable_env_render(self):
        self._env.render()

    def __init_pred_network(self, opt_dict):
        self._predict_net = MLP(hidden_layers=self._conf['hidden_layers'],
                                state_set_size=self._state_set_size,
                                model_path=self._conf['model_path'],
                                trial_number=self.trial_number,
                                action_set_size=self._action_set_size,
                                scope='pred-net',
                                optimizer=opt_dict,
                                trainable=True
                                )

    def __init_target_network(self):
        self._target_net = Network(hidden_layers=self._conf['hidden_layers'],
                                   state_set_size=self._state_set_size,
                                   action_set_size=self._action_set_size,
                                   scope='target-net'
                                   )

    def __init_network_graphs(self):
        init = tf.global_variables_initializer()
        self._sess.run(init)

    def get_experience_size(self):
        return len(self._experience)

    def get_age(self):
        return self._age_of_agent