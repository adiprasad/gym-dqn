import tensorflow as tf

class CNNTarget(object):
    def __init__(self, image_size,  action_set_size, scope, trainable=False):
        self.image_size = image_size
        self.action_set_size = action_set_size
        self.scope = scope
        self.trainable = trainable

        self.batch_input_size = [None] + image_size

        self.create_graph()

    def create_graph(self):
        with tf.variable_scope(self.scope):
            self.input = tf.placeholder(dtype=tf.float32, shape=self.batch_input_size, name='input')

            net = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8,8], strides = 4, activation=tf.nn.relu,name='conv1')
            net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu,name='conv2')
            net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, name='conv3')
            net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu, use_bias=True, name='fc1')
            net = tf.contrib.layers.flatten(net)
            self.out = tf.layers.dense(inputs=net, units=self.action_set_size, activation=None, use_bias=True,name='output')

    def calc_output(self, state, sess):
        return self.out.eval({self.input : state}, session=sess)



class CNNPred(object):
    def __init__(self, image_size, action_set_size, model_path, scope, optimizer, trainable=False):
        self.image_size = image_size
        self.action_set_size = action_set_size
        self.model_path = model_path
        self.scope = scope
        self._optimizer = optimizer
        self.trainable = trainable

        self.batch_input_size = [None] + image_size

        self.create_graph()

    def create_graph(self):
        with tf.variable_scope(self.scope):
            summary_path = 'summary_' + self.model_path.split('/')[0] + '/atari_summary/'
            self.input = tf.placeholder(dtype=tf.float32, shape=self.batch_input_size, name='input')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, ], name='y')
            self.a = tf.placeholder(dtype=tf.int64, shape=[None, 2], name='a')            # For gather ND

            net = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=[8,8], strides = 4, activation=tf.nn.relu,name='conv1')
            net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[4, 4], strides=2, activation=tf.nn.relu,name='conv2')
            net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, name='conv3')
            net = tf.layers.dense(inputs=net, units=512, activation=tf.nn.relu, use_bias=True, name='fc1')
            net = tf.contrib.layers.flatten(net)
            self.out = tf.layers.dense(inputs=net, units=self.action_set_size, activation=None, use_bias=True, name='output')

            self.q_s_a = tf.gather_nd(self.out, self.a)

            self.loss = tf.losses.huber_loss(labels = self.y, predictions = self.q_s_a, delta=1)
            #self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.q_s_a)

            self.avg_q = tf.reduce_mean(self.q_s_a)

            self.global_step = tf.Variable(0, trainable=False)

            if self._optimizer['name'] == 'RMSProp':
                self.learning_step = tf.train.RMSPropOptimizer(learning_rate=self._optimizer['lr']).minimize(self.loss,
                                                            global_step=self.global_step)

            self.fwriter = tf.summary.FileWriter(summary_path,
                                                 graph=tf.get_default_graph())

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("q_value", self.avg_q)
            self.summary_op = tf.summary.merge_all()


    def calc_output(self, state, sess):
        return self.out.eval({self.input : state}, session=sess)

    def save_model(self, saver, sess, step):
        # helper function to save model

        my_path = self.model_path + "weights_{0}.ckpt".format(step)

        saver.save(sess, my_path)