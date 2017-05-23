import tensorflow as tf
import numpy as np
from math import floor
import os

from model.decom_ranking import DecomposableAttentionRankingModel
from config import Config
from data.triplet_corpus_data import TripletCorpusData
from util import log


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '', """Path of pre-built data""")
tf.app.flags.DEFINE_string('vocab_path', '', """Path of vocab""")
tf.app.flags.DEFINE_string('train_dir', '', """Directory for train/save""")
tf.app.flags.DEFINE_string('checkpoint_path', '', """Optional: Path for checkpoint to restore""")
tf.app.flags.DEFINE_string('test_data_path', '', """Optional: Path for test data""")


class TrainingDoneException(Exception):
    pass


def interpret_result(input_ids, pos_ids, neg_ids, pos_infer, neg_infer, dataset, show=3):
    for i in range(show):
        input_sequence = dataset.interpret(input_ids[i], join_string=' ')
        pos_sequence = dataset.interpret(pos_ids[i], join_string=' ')
        neg_sequence = dataset.interpret(neg_ids[i], join_string=' ')

        print('{} -> {} / {} (Pos: {:.4f}, Neg: {:.4f})'.format(
            input_sequence, pos_sequence, neg_sequence,
            pos_infer[i], neg_infer[i]))


def eval_result(pos_infers, neg_infers):
    assert len(pos_infers) == len(neg_infers)

    accuracy = np.sum((pos_infers > neg_infers).astype(np.float32)) / len(pos_infers)
    avg_delta = np.mean(pos_infers - neg_infers)
    return accuracy, avg_delta


def main(argv=None):
    data_path = FLAGS.data_path
    vocab_path = FLAGS.vocab_path
    train_dir = FLAGS.train_dir
    checkpoint_path = FLAGS.checkpoint_path
    test_data_path = FLAGS.test_data_path

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    dataset = TripletCorpusData(max_length=10)
    dataset.load(data_path=data_path, vocab_path=vocab_path,
                 test_data_path=test_data_path)

    config = Config(num_words=dataset.num_symbols,
                    num_category=dataset.num_category,
                    word_embedding_dim=150, rnn_state_size=150,
                    batch_size=256)
    config.save(os.path.join(train_dir, 'config.json'))

    max_epoch = 30
    batch_size = config['training']['batch_size']
    steps_in_epoch = int(floor(dataset.num_train_examples / batch_size))

    model = DecomposableAttentionRankingModel(config)
    saver = tf.train.Saver(tf.global_variables())

    summary_step = 500
    log_step = 2000
    eval_step = 25000
    #test_step = 50000
    save_step = 50000
    max_step = max_epoch * steps_in_epoch

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        if len(checkpoint_path) == 0:
            step = 0
            log.warning("Training Start!")
        else:
            ckpt_basename = os.path.basename(checkpoint_path)
            log.warning("Restoring checkpoint {}".format(ckpt_basename))
            saver.restore(sess, checkpoint_path)
            step = int(ckpt_basename.split('-')[1])
            log.warning("Restoring Done!")

        try:
            while True:
                for _, data_dict in enumerate(dataset.train_datas(batch_size)):
                    feed_dict = model.make_feed_dict(data_dict, is_training=True)
                    run_dict = {'train_op': model.train_op,
                                'pos_inference': model.pos_inference,
                                'neg_inference': model.neg_inference,
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
                                         data_dict['sentence2_pos_inputs'],
                                         data_dict['sentence2_neg_inputs'],
                                         run_results['pos_inference'],
                                         run_results['neg_inference'],
                                         dataset)

                    if (step + 1) % summary_step == 0:
                        summary_writer.add_summary(run_results['summary_op'], step+1)
                        summary_writer.flush()

                    if (step + 1) % save_step == 0:
                        log.warning("Saving Checkpoints...")
                        ckpt_path = os.path.join(train_dir, 'model.ckpt')
                        saver.save(sess, ckpt_path, global_step=step+1)
                        log.warning("Saved checkpoint into {}!".format(ckpt_path))

                    step += 1

                    if (step + 1) % eval_step == 0:
                        pos_infers, neg_infers = [], []
                        for val_data_dict in dataset.val_datas(batch_size):
                            feed_dict = model.make_feed_dict(val_data_dict, is_training=False)
                            pos_infer, neg_infer = sess.run(
                                [model.pos_inference, model.neg_inference],
                                feed_dict)
                            pos_infers.append(pos_infer)
                            neg_infers.append(neg_infer)

                        pos_infers = np.concatenate(pos_infers)
                        neg_infers = np.concatenate(neg_infers)
                        accuracy, avg_delta = eval_result(pos_infers, neg_infers)
                        log.infov("Accuracy: {:.2f}%, Average delta: {:.4f}"
                                  .format(accuracy, avg_delta))
                        eval_summary = tf.Summary(
                            value=[tf.Summary.Value(tag="val_accuracy", simple_value=accuracy),
                                   tf.Sumamry.Value(tag="val_avg_delta", simple_value=avg_delta)])

                        summary_writer.add_summary(eval_summary, step+1)
                        summary_writer.flush()

                    if step + 1 > max_step:
                        raise TrainingDoneException

        except TrainingDoneException:
            log.warning('Training Done!')

if __name__ == '__main__':
    tf.app.run()
