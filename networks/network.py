import tensorflow as tf
import tensorflow.contrib.layers as layers


class Network(object):
    '''
    Inputs :-
    hidden_layers : array of hidden layer sizes
    state_set_size : input layer
    action_set_size : output layer
    '''
    def __init__(self, hidden_layers, state_set_size, action_set_size, scope, trainable=False):
        self._hidden_layers = hidden_layers
        self._input_size = state_set_size
        self._output_size = action_set_size
        self._scope = scope
        self._trainable = trainable
        self._var_dict = {}

        self.__build_graph()

    def __build_graph(self):

        self.s = tf.placeholder(tf.float32, [None, self._input_size], name='state')

        inputs = self.s

        with tf.variable_scope(self._scope):
            for layer_num, layer_size in zip(range(1, len(self._hidden_layers) + 1), self._hidden_layers):
                var_name = self._make_var_name('out', layer_num)

                self._var_dict[var_name] = layers.fully_connected(inputs, layer_size,
                                                                  activation_fn=tf.nn.relu,
                                                                  # weights_initializer=layers.xavier_initializer(),
                                                                  biases_initializer=layers.xavier_initializer(),
                                                                  trainable=self._trainable,
                                                                  reuse=None)

                inputs = self._var_dict[var_name]

            # Make the output layer finally
            self._var_dict['final_out'] = self.out = layers.fully_connected(inputs, self._output_size,
                                                                            activation_fn=None,
                                                                            # weights_initializer=layers.xavier_initializer(),
                                                                            biases_initializer=layers.xavier_initializer(),
                                                                            trainable=self._trainable,
                                                                            reuse=None)

    # '''
    # Builds the hidden layers
    # '''
    # def __build_hidden_layers(self):


    '''
    Input : str1, str2 
    Output :  str1_str2
    '''
    def _make_var_name(self, var_type, var_idx):
        return "{0}_{1}".format(var_type, var_idx)


    '''
    Returns the output
    '''
    def calc_output(self, state, sess):
        #return self.out
        return self.out.eval({self.s : state}, session= sess)



