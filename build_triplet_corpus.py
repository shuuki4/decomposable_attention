import tensorflow as tf
from data.triplet_corpus_data import TripletCorpusData

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('corpus_path', '', """Path of corpus""")
tf.app.flags.DEFINE_string('vocab_path', '', """Path of vocab""")
tf.app.flags.DEFINE_string('save_path', '', """Path for saving data""")


def main(argv=None):
    corpus_path = FLAGS.corpus_path
    vocab_path = FLAGS.vocab_path
    save_path = FLAGS.save_path

    dataset = TripletCorpusData(max_length=10)
    dataset.build(corpus_path, vocab_path, save_path)

if __name__ == "__main__":
    tf.app.run()
