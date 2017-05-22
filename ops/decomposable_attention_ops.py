import tensorflow as tf
from ops.layer_util import MLP


def _masked_softmax(logits, lengths):
    """
    Softmax on last axis with proper mask
    """
    eps = 1e-12
    sequence_mask = tf.expand_dims(
        tf.sequence_mask(
            lengths, maxlen=tf.shape(logits)[-1], dtype=tf.float32),
        dim=1
    )

    max_logits = tf.reduce_max(logits, axis=-1, keep_dims=True)
    masked_logit_exp = tf.exp(logits - max_logits) * sequence_mask
    logit_sum = tf.reduce_sum(masked_logit_exp, axis=-1, keep_dims=True)

    probs = masked_logit_exp / logit_sum
    return probs


def attend(input1, input2,
           length1, length2,
           attention_mapper_num_layers=None,
           attention_mapper_l2_coef=0.003,
           is_training=True):
    """
    :param input1: first sentence representation of shape
     `[batch_size, max_time1, embedding_dim]'
    :param input2: second sentence representation of shape
     `[batch_size, max_time2, embedding_dim]`
    :param length1: lengths of first sentence
    :param length2: lengths of second sentence
    :param attention_mapper_num_layers: size of hidden layers
     for sentence_representation-to-attention mapping mlp
    :param attention_mapper_l2_coef: coefficent for attention
     mapping MLP's L2 regularization
    :param is_training: Python boolean or tensor indicating
     if it is training or not
    :return: (attend_output1, attend_output2, attention_weights)
    """

    embedding_dim = input1.get_shape().as_list()[-1]
    if attention_mapper_num_layers is None:
        attention_mapper_num_layers = [embedding_dim] * 2

    attention_mapper = MLP(
        attention_mapper_num_layers,
        dropout=True,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            scale=attention_mapper_l2_coef),
        name='attend_attention_mapper'
    )

    att_map1 = attention_mapper.apply(input1, is_training=is_training)
    att_map2 = attention_mapper.apply(input2, is_training=is_training)
    att_inner_product = tf.matmul(
        att_map1,
        tf.transpose(att_map2, [0, 2, 1])
    )  # [batch_size, input1_length, input2_length]

    input1_weights = _masked_softmax(
        tf.transpose(att_inner_product, [0, 2, 1]), length1)
    input2_weights = _masked_softmax(att_inner_product, length2)
    output1 = tf.matmul(input1_weights, input1)
    output2 = tf.matmul(input2_weights, input2)

    return output1, output2, input1_weights, input2_weights


def compare(orig_input1, orig_input2, attend_input1, attend_input2,
            mapper_num_layers=None,
            mapper_l2_coef=0.003,
            is_training=True):
    """
    :param orig_input1: original first sentence representation of shape
     '[batch_size, max_time1, embedding_dim]
    :param orig_input2: original second sentence representation of shape
     '[batch_size, max_time2, embedding_dim]
    :param attend_input1: attended aligned phrase of first sentence
     of shape `[batch_size, max_time2, attend_dim]`
    :param attend_input2: attended aligned phrase of second sentence
     of shape `[batch_size, max_time1, attend_dim]`
    :param mapper_num_layers: size of hidden layers
     for concat-to-compare mapping mlp
    :param mapper_l2_coef: coefficient for mapper l2 regularization
    :param is_training: Python boolean or tensor indicating
     if it is training or not
    :return: (compare1, compare2)
    """

    embedding_dim = orig_input1.get_shape().as_list()[-1]
    attend_dim = attend_input1.get_shape().as_list()[-1]
    if mapper_num_layers is None:
        mapper_num_layers = [(embedding_dim + attend_dim) / 2] * 2

    mapper = MLP(
        mapper_num_layers,
        dropout=True,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            scale=mapper_l2_coef),
        name='compare_mapper'
    )

    compare1 = mapper.apply(
        tf.concat([orig_input1, attend_input2], 2),
        is_training=is_training)
    compare2 = mapper.apply(
        tf.concat([orig_input2, attend_input1], 2),
        is_training=is_training)

    return compare1, compare2


def aggregate(compare1, compare2,
              length1, length2,
              mapper_num_layers,
              mapper_l2_coef=0.003,
              is_training=True):
    """
    :param compare1: compare result1 from compare(), of shape
      '[batch_size, max_time1, compare_dim]`
    :param compare2: compare result2 from compare(), of shape
     `[batch_size, max_time2, compare_dim]`
    :param length1: lengths of first sentence
    :param length2: lengths of second sentence
    :param mapper_num_layers: size of hidden layers for
     final result mapper, where mapper_num_layers[-1]
     should be the number of category
    :param mapper_l2_coef: coef for mapper l2 regularization
    :param is_training: Python boolean or tensor indicating
     if it is training or not
    :return: result: final logit tensor
    """

    sequence_mask1 = tf.expand_dims(
        tf.sequence_mask(
            length1, maxlen=tf.shape(compare1)[1], dtype=tf.float32),
        dim=-1)
    sequence_mask2 = tf.expand_dims(
        tf.sequence_mask(
            length2, maxlen=tf.shape(compare2)[1], dtype=tf.float32),
        dim=-1)
    masked_compare1 = sequence_mask1 * compare1
    masked_compare2 = sequence_mask2 * compare2

    compare1_sum = tf.reduce_sum(masked_compare1, 1)
    compare2_sum = tf.reduce_sum(masked_compare2, 1)

    mapper = MLP(
        mapper_num_layers,
        dropout=True,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(
            scale=mapper_l2_coef),
        name='aggregate_mapper'
    )
    result = mapper.apply(
        tf.concat([compare1_sum, compare2_sum], 1),
        is_training=is_training)

    return result