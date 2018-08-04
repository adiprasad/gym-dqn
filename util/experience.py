from collections import deque
import numpy as np

class Experience(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.state = deque(maxlen=max_size)
        self.action = deque(maxlen=max_size)
        self.reward = deque(maxlen=max_size)
        self.next_state = deque(maxlen=max_size)
        self.done = deque(maxlen=max_size)

    def get_minibatch(self, minibatch_size):
        minibatch = np.random.choice(self.get_size(), minibatch_size,  replace=False)

        state_list = np.array(list(self.state))
        action_list = np.array(list(self.action))
        reward_list = np.array(list(self.action))
        next_state_list = np.array(list(self.next_state))
        done_list = np.array(list(self.done))

        return np.stack(state_list[minibatch],axis=0), action_list[minibatch], \
               reward_list[minibatch], np.stack(next_state_list[minibatch],axis=0), done_list[minibatch]

    def add_experience(self, state, action, reward, next_state, terminal):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(terminal)

    def get_size(self):
        return len(self.done)


