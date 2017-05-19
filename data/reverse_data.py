import random
from data.pair_sequence_data import PairSequenceData


class ReverseData(PairSequenceData):
    """
    Toy data for determining if the pair is reverse or not
    """
    @staticmethod
    def make_data(num_data, num_symbols, length, tf_ratio=0.5):
        lst = [x+1 for x in range(num_symbols)]  # exclude PAD

        def make_true():
            seq1 = [random.choice(lst) for _ in range(length)]
            seq2 = seq1[::-1]
            return seq1, seq2, 1

        def make_false():
            seq1 = [random.choice(lst) for _ in range(length)]
            seq2 = random.sample(seq1, length)
            return seq1, seq2, 0

        num_true = int(num_data * tf_ratio)
        num_false = num_data - num_true

        return [make_true() for _ in range(num_true)] + \
               [make_false() for _ in range(num_false)]

    def build(self, num_symbols=10, length=10,
              num_train=50000, num_val=10000):
        self.num_category = 2
        self.symbols = ['_'] + [str(x) for x in range(num_symbols)]

        self.train_data = self.make_data(num_train, num_symbols, length,
                                         tf_ratio=0.5)
        self.val_data = self.make_data(num_val, num_symbols, length,
                                       tf_ratio=0.8)

    def load(self):
        raise NotImplementedError
