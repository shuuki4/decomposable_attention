import tensorflow as tf
from ops import decomposable_attention_ops as decom_ops
from model.base_model import BaseModel
from tensorflow.contrib.rnn import GRUCell


class DecomposableAttentionRankingModel(BaseModel):

    def __init__(self, config):
        super(DecomposableAttentionRankingModel, self).__init__(config)

    def _build_graph(self):
        with tf.name_scope('inputs'):
            sentence1, sentence1_lengths, \
                sentence2_pos, sentence2_pos_lengths,\
                sentence2_neg, sentence2_neg_lengths,\
                is_training = self._build_inputs()

        self._word_embedding = self.make_word_embedding()

        with tf.name_scope('rnn_encode'):
            sentence1_embed, sentence2_pos_embed, sentence2_neg_embed, \
                sentence1_rnned, sentence2_pos_rnned, sentence2_neg_rnned = \
                self._build_rnn_encoder(
                    sentence1, sentence2_pos, sentence2_neg,
                    sentence1_lengths, sentence2_pos_lengths, sentence2_neg_lengths)

        with tf.name_scope('attend'):
            sentence1_pos_attend, sentence2_pos_attend, \
                pos_att_weights1, pos_att_weights2 = \
                decom_ops.attend(
                    sentence1_rnned, sentence2_pos_rnned,
                    sentence1_lengths, sentence2_pos_lengths,
                    is_training=is_training)

            sentence1_neg_attend, sentence2_neg_attend, \
                neg_att_weights1, neg_att_weights2 = \
                decom_ops.attend(
                    sentence1_rnned, sentence2_neg_rnned,
                    sentence1_lengths, sentence2_neg_lengths,
                    is_training=is_training,
                    reuse=True)

        with tf.name_scope('compare'):
            pos_compare1, pos_compare2 = decom_ops.compare(
                sentence1_embed, sentence2_pos_embed,
                sentence1_pos_attend, sentence2_pos_attend,
                is_training=is_training)
            neg_compare1, neg_compare2 = decom_ops.compare(
                sentence1_embed, sentence2_neg_embed,
                sentence1_neg_attend, sentence2_neg_attend,
                is_training=is_training,
                reuse=True)

        compare_dim = self.config['rnn']['state_size'] + \
            self.config['word']['embedding_dim'] // 2
        with tf.name_scope('aggregate'):
            pos_result = decom_ops.aggregate(
                pos_compare1, pos_compare2,
                sentence1_lengths, sentence2_pos_lengths,
                mapper_num_layers=[int(compare_dim * 0.7), 1],
                is_training=is_training)
            neg_result = decom_ops.aggregate(
                neg_compare1, neg_compare2,
                sentence1_lengths, sentence2_neg_lengths,
                mapper_num_layers=[int(compare_dim * 0.7), 1],
                is_training=is_training,
                reuse=True)

            pos_result = tf.squeeze(tf.tanh(pos_result), axis=[-1])
            neg_result = tf.squeeze(tf.tanh(neg_result), axis=[-1])

        with tf.name_scope('loss'):
            self.loss = self._build_loss(pos_result, neg_result)

        with tf.name_scope('summary'):
            tf.summary.histogram('pos_result', pos_result)
            tf.summary.histogram('neg_result', neg_result)
            attentions = [
                (pos_att_weights1, 'pos_att_weight1', sentence2_pos_lengths),
                (pos_att_weights2, 'pos_att_weight2', sentence1_lengths),
                (neg_att_weights1, 'neg_att_weight1', sentence2_neg_lengths),
                (neg_att_weights2, 'neg_att_weight2', sentence1_lengths)
            ]
            for attention_info in attentions:
                self._build_attention_viz(*attention_info)

        self.pos_inference = pos_result
        self.neg_inference = neg_result
        self.train_step, self.train_op = self._build_train_step(self.loss)
        self.summary_op = tf.summary.merge_all()

    def _build_inputs(self):
        self._inputs['sentence1_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='sentence1_inputs'
        )
        self._inputs['sentence1_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='sentence1_lengths'
        )
        self._inputs['sentence2_pos_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='sentence2_inputs'
        )
        self._inputs['sentence2_pos_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='sentence2_lengths'
        )
        self._inputs['sentence2_neg_inputs'] = tf.placeholder(
            shape=(None, None), # batch_size, max_time
            dtype=tf.int32,
            name='sentence2_inputs'
        )
        self._inputs['sentence2_neg_lengths'] = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='sentence2_lengths'
        )
        self._inputs['is_training'] = tf.placeholder(
            shape=tuple(),
            dtype=tf.bool,
            name='is_training'
        )

        return self._inputs['sentence1_inputs'], self._inputs['sentence1_lengths'], \
               self._inputs['sentence2_pos_inputs'], self._inputs['sentence2_pos_lengths'], \
               self._inputs['sentence2_neg_inputs'], self._inputs['sentence2_neg_lengths'], \
               self._inputs['is_training']

    def _build_rnn_encoder(self, sentence1, sentence2_pos, sentence2_neg,
                           sentence1_lengths, sentence2_pos_lengths, sentence2_neg_lengths):

        with tf.variable_scope('word_embedding'):
            sentence1_embedding = tf.nn.embedding_lookup(self._word_embedding, sentence1)
            sentence2_pos_embedding = tf.nn.embedding_lookup(self._word_embedding, sentence2_pos)
            sentence2_neg_embedding = tf.nn.embedding_lookup(self._word_embedding, sentence2_neg)

        with tf.variable_scope('rnn'):
            def _run_birnn(fw_cell, bw_cell, inputs, lengths):
                (fw_output, bw_output), (fw_final_state, bw_final_state) =\
                    tf.nn.bidirectional_dynamic_rnn(
                        fw_cell, bw_cell,
                        inputs,
                        sequence_length=lengths,
                        time_major=False,
                        dtype=tf.float32
                    )

                output = tf.concat([fw_output, bw_output], 2)
                state = tf.concat([fw_final_state, bw_final_state], 1)
                return output, state

            state_size = self.config['rnn']['state_size']
            forward_cell = GRUCell(state_size)
            backward_cell = GRUCell(state_size)

            sentence1_rnned, _ = _run_birnn(forward_cell, backward_cell,
                                            sentence1_embedding, sentence1_lengths)
            sentence2_rnned, _ = _run_birnn(
                forward_cell, backward_cell,
                tf.concat([sentence2_pos_embedding, sentence2_neg_embedding], 0),
                tf.concat([sentence2_pos_lengths, sentence2_neg_lengths], 0))
            sentence2_pos_rnned, sentence2_neg_rnned = \
                tf.split(sentence2_rnned, num_or_size_splits=2, axis=0)

        return sentence1_embedding, sentence2_pos_embedding, sentence2_neg_embedding, \
               sentence1_rnned, sentence2_pos_rnned, sentence2_neg_rnned

    @staticmethod
    def _build_attention_viz(att_weight, att_name, lengths):
        mask = tf.expand_dims(
            tf.sequence_mask(lengths,
                             maxlen=tf.shape(att_weight)[1],
                             dtype=tf.float32),
            axis=-1)
        att_weight = att_weight * mask

        tf.summary.histogram(att_name, att_weight)
        tf.summary.image(att_name + '_viz',
                         tf.cast(
                             tf.expand_dims(att_weight, -1) * 255.0,
                             dtype=tf.uint8))

    @staticmethod
    def _build_loss(pos_results, neg_results, margin=1.5):
        ranking_loss = tf.reduce_mean(
            tf.maximum(0.0, margin - pos_results + neg_results))
        l2_loss = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        total_loss = ranking_loss + l2_loss

        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('ranking_loss', ranking_loss)
        tf.summary.scalar('l2_loss', l2_loss)
        return total_loss
