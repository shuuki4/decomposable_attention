import tensorflow as tf
import numpy as np


class BaseModel:
    """
    Base Interface Neural network model
    """

    def __init__(self, config):
        self.config = config
        self._inputs = {}

        self._build_graph()

    def _build_graph(self):
        raise NotImplementedError

    def make_word_embedding(self):
        if self.config['word']['pretrained_word_path']:
            initializer = np.load(self.config['word']['pretrained_word_path'])
        else:
            initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope('word_embedding'):
            word_embedding = tf.get_variable(
                name="word_embedding",
                shape=(self.config['data']['num_word'], self.config['word']['embedding_dim']),
                initializer=initializer,
                dtype=tf.float32
            )
        return word_embedding

    def make_feed_dict(self, data_dict, is_training=True):
        feed_dict = {self._inputs['is_training']: is_training}
        for key in data_dict.keys():
            try:
                feed_dict[self._inputs[key]] = data_dict[key]
            except KeyError:
                continue
        return feed_dict

    def _build_train_step(self, loss):
        with tf.name_scope('train'):
            train_step = tf.Variable(0, name='global_step', trainable=False)
            lr = self.config['training']['learning_rate']
            opt = tf.train.AdamOptimizer(learning_rate=lr)

            train_variables = tf.trainable_variables()
            grads_vars = opt.compute_gradients(loss, train_variables)
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, 1.0), var)
            apply_gradient_op = opt.apply_gradients(grads_vars, global_step=train_step)
            with tf.control_dependencies([apply_gradient_op]):
                train_op = tf.no_op(name='train_step')

        return train_step, train_op
