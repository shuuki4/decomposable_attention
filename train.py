import tensorflow as tf
import numpy as np
from math import floor

from model import DecomposableAttentionModel
from config import Config
from data.reverse_data import ReverseData
from util import log


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '', """Directory for train/save""")


def interpret_result(input_ids, output_ids, answers, infers, dataset, show=3):
    for i in range(show):
        input_sequence = dataset.interpret(input_ids[i], join_string=' ')
        output_sequence = dataset.interpret(output_ids[i], join_string=' ')

        # temporary for calculation seq data
        print('{} -> {} (Real: {}, Infer: {})'.format(
            input_sequence, output_sequence, answers[i], infers[i]))


def eval_result(answers, infers):
    compare = np.equal(answers, infers)
    _right = np.sum(compare)
    _wrong = len(compare) - _right

    return _right, _wrong


def main(argv=None):
    train_dir = FLAGS.train_dir

    dataset = ReverseData()
    dataset.build(num_train=300000, num_val=10000)

    config = Config(num_words=dataset.num_symbols,
                    num_category=dataset.num_category,
                    word_embedding_dim=30, rnn_state_size=30,
                    batch_size=128)

    max_epoch = 100
    batch_size = config['training']['batch_size']
    steps_in_epoch = int(floor(dataset.num_train_examples / batch_size))

    model = DecomposableAttentionModel(config)
    #saver = tf.train.Saver(tf.global_variables())

    summary_step = 300
    log_step = 300
    #save_step = 50000

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        log.warning("Training Start!")

        step = 0
        for epoch in range(1, max_epoch+1):
            log.warning("Epoch {}".format(epoch))
            for _, data_dict in enumerate(dataset.train_datas(batch_size)):
                feed_dict = model.make_feed_dict(data_dict, is_training=True)
                run_dict = {'train_op': model.train_op,
                            'inference': model.inference,
                            'loss': model.loss}
                if (step + 1) % summary_step == 0:
                    run_dict['summary_op'] = model.summary_op

                run_results = sess.run(run_dict, feed_dict)

                if (step + 1) % log_step == 0:
                    log.info("Step {cur_step:6d} (Epoch {float_epoch:6.3f}) ... Loss: {loss:.5f}"
                             .format(cur_step=step+1,
                                     float_epoch=float(step+1)/steps_in_epoch,
                                     loss=run_results['loss']))

                    interpret_result(data_dict['sentence1_inputs'],
                                     data_dict['sentence2_inputs'],
                                     data_dict['labels'],
                                     run_results['inference'],
                                     dataset)

                if (step + 1) % summary_step == 0:
                    summary_writer.add_summary(run_results['summary_op'], step)
                    summary_writer.flush()
                step += 1

            # eval
            right, wrong = 0.0, 0.0
            for data_dict in dataset.val_datas(batch_size):
                feed_dict = model.make_feed_dict(data_dict, is_training=False)
                inference = sess.run(model.inference, feed_dict)

                now_right, now_wrong = eval_result(data_dict['labels'], inference)
                right += now_right
                wrong += now_wrong

            log.infov("Right: {}, Wrong: {}, Accuracy: {:.2f}%".format(right, wrong, 100*right/(right+wrong)))

if __name__ == '__main__':
    tf.app.run()
