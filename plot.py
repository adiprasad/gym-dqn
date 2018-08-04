import numpy as np 
import os
import matplotlib.pyplot as plt


PLOT_DIR='/Volumes/Data/School/Study/687/Project/gym-dqn'

def plot_data(discounted_returns, **kwargs):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    filename = os.path.join(PLOT_DIR, kwargs['name']+'.png')
    print('\nPloting data...')
    x = np.array(range(discounted_returns.shape[0]))
    y = np.mean(discounted_returns, axis=1)
    e = np.std(discounted_returns, axis=1)
    plt.errorbar(x, y, e, linestyle='--', marker='o')
    if kwargs['label_plot'] != 'default':
        title = kwargs['label_plot'] + ' {0}: ({1} trials)'.format('Deep Q-learning' if kwargs['update_type'] == 'DQN' else 'sarsa', kwargs['num_trials'])
    else:
        title = 'Acrobot' + ' {0}: ({1} trials)'.format('Deep Q-learning' if kwargs['update_type'] == 'DQN' else 'sarsa', kwargs['num_trials'])

    if kwargs['label_x'] == 'default':
        x_label = 'Number of training episodes'
    else:
        x_label = kwargs['label_x']

    if kwargs['label_y'] == 'default':
        y_label = 'Mean return'
    else:
        y_label = kwargs['label_y']

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    print('Plot saved in {0}'.format(filename))


if __name__=="__main__":
    rewards_matrix = np.load("rmat_acrobot.npy")

    rewards_matrix = np.transpose(rewards_matrix)

    plot_data(rewards_matrix, label_plot='default', update_type='DQN', num_trials=5, label_x = 'default', label_y ='default', name='cartpole')

