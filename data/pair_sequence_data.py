import numpy as np
from random import shuffle


class PairSequenceData:
    """
    Pair sequence data class interface for decomposable attention model
    """

    PAD = 0
    UNK = 1

    def __init__(self):
        """
        Data format should be [(sequence1, sequence2, label)]
        """
        self.num_category = None
        self.symbols = None

        self.train_data = list()
        self.val_data = list()

    @property
    def num_train_examples(self):
        return len(self.train_data)

    @property
    def num_val_examples(self):
        return len(self.val_data)

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
        def _normalize_length(_data, max_length):
            return _data + [0] * (max_length - len(_data))

        seq1_data, seq1_lengths, seq2_data, seq2_lengths, labels = \
            [], [], [], [], []
        for idx in batch_idxs:
            seq1, seq2, _ = data[idx]
            seq1_lengths.append(len(seq1))
            seq2_lengths.append(len(seq2))

        seq1_max_length = max(seq1_lengths)
        seq2_max_length = max(seq2_lengths)
        for idx in batch_idxs:
            seq1, seq2, label = data[idx]
            seq1_data.append(_normalize_length(seq1, seq1_max_length))
            seq2_data.append(_normalize_length(seq2, seq2_max_length))
            labels.append(label)

        batch_data_dict = {
            'sentence1_inputs': np.asarray(seq1_data, dtype=np.int32),
            'sentence1_lengths': np.asarray(seq1_lengths, dtype=np.int32),
            'sentence2_inputs': np.asarray(seq2_data, dtype=np.int32),
            'sentence2_lengths': np.asarray(seq2_lengths, dtype=np.int32),
            'labels': np.asarray(labels, dtype=np.int32)
        }
        return batch_data_dict

    def _data_iterator(self, sequence, batch_size, random):
        idxs = list(range(len(sequence)))
        if random:
            shuffle(idxs)

        for start_idx in range(0, len(sequence), batch_size):
            end_idx = start_idx + batch_size
            next_batch = self._next_batch(sequence, idxs[start_idx:end_idx])

            # return batch only if the size of batch is original batch size
            if len(next_batch['sentence1_inputs']) == batch_size:
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

    def train_data_by_idx(self, start, end):
        assert start >= 0 and end <= len(self.train_data)
        return self._next_batch(self.train_data, range(start, end))

    def val_data_by_idx(self, start, end):
        assert start >= 0 and end < len(self.val_data)
        return self._next_batch(self.val_data, range(start, end))

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
