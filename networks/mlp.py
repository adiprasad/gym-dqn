import tensorflow as tf
import tensorflow.contrib.layers as layers

class MLP(object):
    '''
    Inputs :-
    hidden_layers : array of hidden layer sizes
    state_set_size : input layer
    action_set_size : output layer
    '''
    def __init__(self, hidden_layers, state_set_size, model_path, trial_number, action_set_size, scope, optimizer, trainable=False):
        self._hidden_layers = hidden_layers
        self._input_size = state_set_size
        self.model_path = model_path
        self.trial_number = trial_number
        self._output_size = action_set_size
        self._scope = scope
        self._trainable = trainable

        self._optimizer = optimizer         # Dict containing optimizer info

        # Build the network
        self.__build_graph()

    def __build_graph(self):
        with tf.variable_scope(self._scope):
            self.s = tf.placeholder(tf.float32, [None, self._input_size], name='state')
            summary_path = self.model_path.split('/')[0] + '/dqn_summary_{0}/'.format(self.trial_number)

            inputs = self.s

            for layer_num, layer_size in zip(range(1, len(self._hidden_layers) + 1), self._hidden_layers):
                var_name = self._make_var_name('out', layer_num)

                inputs = layers.fully_connected(inputs, layer_size,
                                                                  activation_fn=tf.nn.relu,
                                                                  # weights_initializer=layers.xavier_initializer(),
                                                                  # biases_initializer=layers.xavier_initializer(),
                                                                  trainable=self._trainable,
                                                                  reuse=None)


            # Make the output layer finally
            self.out = layers.fully_connected(inputs, self._output_size,
                                              activation_fn=None,
                                              # weights_initializer=layers.xavier_initializer(),
                                              # biases_initializer=layers.xavier_initializer(),
                                              trainable=self._trainable,
                                              reuse=None)

            if self._trainable is True:
                self.y = tf.placeholder(tf.float32, shape=[None], name='y')   # y coming in for the prediction network
                self.a = tf.placeholder(tf.int64, shape=[None, 2], name='a')    # Action coming in as [batch_size, action_idx] for gather_nd

                self.q_s_a = tf.gather_nd(self.out, self.a)
                # self.target_minus_prediction = tf.clip_by_value(tf.subtract(self.y, self.q_s_a), -1, 1)
                self.target_minus_prediction = tf.subtract(self.y, self.q_s_a)
                self.loss = tf.reduce_mean(tf.square(self.target_minus_prediction))

                self.avg_q = tf.reduce_mean(self.q_s_a)

                self.global_step = tf.Variable(0, trainable=False)

                if self._optimizer['name'] == 'adam':
                    self.learning_step = tf.train.AdamOptimizer(self._optimizer['lr']).minimize(self.loss, global_step=self.global_step)

                self.fwriter = tf.summary.FileWriter(summary_path,
                                                graph=tf.get_default_graph())

                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("q_value", self.avg_q)
                self.summary_op = tf.summary.merge_all()


    '''
    Input : str1, str2 
    Output :  str1_str2
    '''
    def _make_var_name(self, var_type, var_idx):
        return "{0}_{1}".format(var_type, var_idx)

    def calc_output(self, state, sess):
        return self.out.eval({self.s : state}, session=sess)

    def save_model(self, saver, sess, my_path):
        pass
        #saver.save(sess, my_path)

# if __name__=='__main__':
#     opt_dict = {'name': 'adam', 'lr': 0.001}
#
#     target_net = MLP(hidden_layers=[10, 10], state_set_size=4, action_set_size=2, scope='target-net',
#                      optimizer=opt_dict, trainable=True)