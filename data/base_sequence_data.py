import numpy as np
from random import shuffle


class BaseSequenceData:
    """
    Base Interface for Sequencial Dataset
    """

    PAD = 0
    UNK = 1

    def __init__(self):
        self.num_category = None
        self.symbols = None

        self.train_data = list()
        self.val_data = list()
        self.test_data = list()

    @property
    def num_train_examples(self):
        return len(self.train_data)

    @property
    def num_val_examples(self):
        return len(self.val_data)

    @property
    def num_test_examples(self):
        return len(self.test_data)

    @property
    def num_symbols(self):
        return len(self.symbols)

    @property
    def idx_to_symbol(self, symbol_idx):
        return self.symbols[symbol_idx]

    @property
    def initialized(self):
        return (self.num_category is not None) and (self.num_symbols is not None)

    def _next_batch(self, data, batch_idxs):
        """
        Generate next batch.
        :param data: data list to process
        :param batch_idxs: idxs to process
        :return: next data dict of batch_size amount data
        """
        raise NotImplementedError

    def _data_iterator(self, sequence, batch_size, random):
        idxs = list(range(len(sequence)))
        if random:
            shuffle(idxs)

        for start_idx in range(0, len(sequence), batch_size):
            end_idx = start_idx + batch_size
            if end_idx > len(sequence):
                end_idx = len(sequence)
            next_batch = self._next_batch(sequence, idxs[start_idx:end_idx])
            yield next_batch

    def train_datas(self, batch_size=16, random=True):
        """
        Iterate through train data for single epoch
        :param batch_size: batch size
        :param random: if true, iterate randomly
        :return: train data iterator
        """
        assert self.initialized, "Dataset is not initialized!"
        return self._data_iterator(self.train_data, batch_size, random)

    def val_datas(self, batch_size=16, random=True):
        """
        Iterate through validaiton data for single epoch
        :param batch_size: batch size
        :param random: if true, iterate randomly
        :return: validation data iterator
        """
        assert self.initialized, "Dataset is not initialized!"
        return self._data_iterator(self.val_data, batch_size, random)

    def test_datas(self, batch_size=16, random=False):
        assert self.initialized, "Dataset is not initialized!"
        return self._data_iterator(self.test_data, batch_size, random)

    def train_data_by_idx(self, start, end):
        assert start >= 0 and end <= len(self.train_data)
        return self._next_batch(self.train_data, range(start, end))

    def val_data_by_idx(self, start, end):
        assert start >= 0 and end <= len(self.val_data)
        return self._next_batch(self.val_data, range(start, end))

    def test_data_by_idx(self, start, end):
        assert start >= 0 and end <= len(self.test_data)
        return self._next_batch(self.test_data, range(start, end))

    def interpret(self, ids, join_string=''):
        real_ids = []
        for _id in ids:
            if _id != self.PAD:
                real_ids.append(_id)
            else:
                break

        return join_string.join(self.symbols[ri] for ri in real_ids)

    def build(self):
        """
        Build data and save in self.train_sequences, self.val_sequences
        """
        raise NotImplementedError

    def load(self):
        """
        Load data and save in self.train_sequences, self.val_sequences
        """
        raise NotImplementedError
