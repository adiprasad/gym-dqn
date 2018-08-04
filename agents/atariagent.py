import numpy as np
import os
import sys
sys.path.insert(1, os.path.realpath(os.path.join(sys.path[0], os.pardir)))
from networks.cnn import CNNPred
from networks.cnn import CNNTarget
import tensorflow as tf
import random
from util.experience import Experience
from collections import deque
from scipy.misc import imresize

class AtariAgent(object):
    def __init__(self, env, conf_dict, optimizer_dict, sess):
        self._env = env
        self._conf = conf_dict
        self._sess = sess
        self._eps = conf_dict['eps-init']
        self._eps_min = conf_dict['eps-final']
        self._eps_decay = conf_dict['eps-decay']
        self._action_set_size = self._env.action_space.n
        self._state_image_size = [84,84,4]

        #Seed
        random.seed(0)
        np.random.seed(0)

        # Initialize prediction and target networks
        self.__init_pred_network(optimizer_dict)
        self.__init_target_network()

        self.__init_network_graphs()
        self._age_of_agent = 0                  # Number of time steps since the agent was initialized

        self._frame_holder = deque(maxlen=4)        # Hold the last 4 preprocessed frames, will be used to create _state

        #Reset environment
        self._last_frame = self._agent_start_frame = self._env.reset()
        self._lives_remaining = 5

        # Experience format [prev_state, action, reward, curr_state, terminal?]
        self._experience = Experience(int(conf_dict['replay-max-size']))

        # Initialize experience using random policy
        print("Initializing experience")
        self.init_experience()

        self.saver = tf.train.Saver()


    def get_state(self):
        return np.stack(list(self._frame_holder), axis=2)           # State comes from frame holder

    def get_elapsed_time(self):
        return self._age_of_episode

    '''
    Return action using eps-greedy policy
    Random action within eps and through the network after eps
    '''
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)

        a = self.__eps_greedy_policy(state)

        if self._age_of_agent <= self._conf['exploration-time']:
            self._eps -= (self._conf['eps-init'] - self._conf['eps-final'])/(self._conf['exploration-time'])

        return a

    '''
    Add current experience to the existing experience
    Trim the experiences to include the last replay-max-size experiences only 
    '''
    def add_experience(self, state, action, reward, next_state, terminal):
        self._experience.add_experience(state, action, reward, next_state, terminal)

    '''
    Initialize experience using random policy
    according to init-experience-size
    '''
    def init_experience(self):
        frame = np.dot(self._agent_start_frame[...,:3], [0.299, 0.587, 0.114])
        frame = imresize(frame / 255., (84, 84))

        for i in range(4):              # use the same frame for init state
            self._frame_holder.append(frame)

        for i in range(int(self._conf['replay-start-size'])):
            state = self.get_state()
            a = self.__random_policy()
            next_state, reward, done, consider_terminal_state = self.take_action(a)
            self.add_experience(state, a, reward, next_state, consider_terminal_state)

            if done:
                self.reset_env()

        print("Experience initialized")

    # Takes action in the environment
    # Increases the time steps elapsed
    # Returns next state, reward, done
    def take_action(self, a):
        reward_over_frames = 0
        lives_at_start = self._lives_remaining
        consider_terminal_state = False

        for i in range(self._conf['action-repeat']):
            next_frame, reward, terminal, lives = self._env.step(a)
            self._lives_remaining = lives['ale.lives']

            reward_over_frames += reward

            # Start counting only after experience replay has been populated
            if self.get_experience_size() >= self._conf['replay-start-size']:
                #self._age_of_episode = self._age_of_episode + 1
                self._age_of_agent = self._age_of_agent + 1

            if terminal is True:
                self.reset_env()
                break

        self.add_frame_to_state(next_frame)
        next_state = self.get_state()


        if (terminal is True) or (self._lives_remaining < lives_at_start):
            consider_terminal_state = True

        return next_state, np.clip(np.sum(reward_over_frames),-1,1), terminal, consider_terminal_state


    '''
    Pick experience relays and train the network
    Also transfer the network weights from prediction network
    to target network according to target-network-update-frequency
    '''
    def train_q_networks(self):
        s_pred_net, a, rewards, s_target_net, done = self._experience.get_minibatch(self._conf['mini-batch-size'])
        #print(a)

        a = self.__prepare_actions(a)           # Convert actions to be able to index the tensor inside

        q_target = self._target_net.calc_output(s_target_net, self._sess)
        #q_target = self._predict_net.calc_output(s_target_net, self._sess)              # Running without target networks

        y_targets = self.__prepare_targets(rewards, q_target, done)


        loss, step, summary, _ = self._sess.run([self._predict_net.loss, self._predict_net.global_step,
                                                 self._predict_net.summary_op, self._predict_net.learning_step],
                                                feed_dict={self._predict_net.input: s_pred_net,
                                                           self._predict_net.a: a,
                                                           self._predict_net.y: y_targets}
                                                )
        # Save model every 50000th step
        if step % 50000 == 0:
            self._predict_net.save_model(self.saver, self._sess, step)

        T = self._age_of_agent

        #print("T = {0}, Loss = {1}, Minibatch update = {2}, Num Iterations = {3}".format(T, loss, step, self.get_age()))


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
        init_frame = self._env.reset()
        self._age_of_episode = 0

        no_op_frame = self.start_no_op(init_frame)

        self.add_frame_to_state(no_op_frame)

        state = self.get_state()

        self._lives_remaining = 0

        return state

    def start_no_op(self, init_frame):
        no_op_length = np.random.randint(0,self._conf['no-op-max'])
        state = init_frame

        for i in range(no_op_length):
            state, _, _, _ = self._env.step(0)

        return state


    def add_frame_to_state(self, frame):
        pp_frame = self.preprocess_frame(frame)
        self.__add_to_frame_holder(pp_frame)

    def preprocess_frame(self, frame):
        last_frame = self._last_frame
        self._last_frame = frame

        frame = np.maximum(last_frame, frame)
        frame =  np.dot(frame[..., :3], [0.299, 0.587, 0.114])

        return imresize(frame / 255., (84,84))

    # Add preprocessed frame to frame holder
    def __add_to_frame_holder(self, frame):
        self._frame_holder.append(frame)

    def enable_env_render(self):
        self._env.render()

    def __init_pred_network(self, opt_dict):
        self._predict_net = CNNPred(image_size=self._state_image_size,
                                action_set_size=self._action_set_size,
                                model_path="model_breakout_atari/",
                                scope='pred-net',
                                optimizer=opt_dict,
                                trainable=True
                                )

    def __init_target_network(self):
        self._target_net = CNNTarget(image_size=self._state_image_size,
                                    action_set_size=self._action_set_size,
                                    scope='target-net'
                                   )

    def __init_network_graphs(self):
        init = tf.global_variables_initializer()
        self._sess.run(init)

    def get_experience_size(self):
        return self._experience.get_size()

    def get_age(self):
        return self._age_of_agent