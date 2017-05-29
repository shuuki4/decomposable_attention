import numpy as np
from random import shuffle

from data.base_sequence_data import BaseSequenceData


class TripletSequenceData(BaseSequenceData):
    """
    Pair sequence data class interface for
    decomposable attention ranking model
    """
    def __init__(self):
        """
        Data format should be [(sequence1, sequence2_pos, sequence2_neg)]
        """
        super(TripletSequenceData, self).__init__()
        self.test_label = []

    def _next_batch(self, data, batch_idxs, labels=None):
        """
        Generate next batch.
        :param data: data list to process
        :param batch_idxs: idxs to process
        :param labels: additional labels to include
        :return: next data dict of batch_size amount data
        """
        def _normalize_length(_data, max_length):
            return _data + [self.PAD] * (max_length - len(_data))

        seq1_data, seq1_lengths, seq2_pos_data, \
            seq2_pos_lengths, seq2_neg_data, seq2_neg_lengths = \
            [], [], [], [], [], []
        for idx in batch_idxs:
            seq1, seq2_pos, seq2_neg = data[idx]
            seq1_lengths.append(len(seq1))
            seq2_pos_lengths.append(len(seq2_pos))
            seq2_neg_lengths.append(len(seq2_neg))

        seq1_max_length = max(seq1_lengths)
        seq2_max_length = max(seq2_pos_lengths + seq2_neg_lengths)

        for idx in batch_idxs:
            seq1, seq2_pos, seq2_neg = data[idx]
            seq1_data.append(_normalize_length(seq1, seq1_max_length))
            seq2_pos_data.append(_normalize_length(seq2_pos, seq2_max_length))
            seq2_neg_data.append(_normalize_length(seq2_neg, seq2_max_length))

        batch_data_dict = {
            'sentence1_inputs': np.asarray(seq1_data, dtype=np.int32),
            'sentence1_lengths': np.asarray(seq1_lengths, dtype=np.int32),
            'sentence2_pos_inputs': np.asarray(seq2_pos_data, dtype=np.int32),
            'sentence2_pos_lengths': np.asarray(seq2_pos_lengths, dtype=np.int32),
            'sentence2_neg_inputs': np.asarray(seq2_neg_data, dtype=np.int32),
            'sentence2_neg_lengths': np.asarray(seq2_neg_lengths, dtype=np.int32),
        }

        if labels:
            batch_data_dict['labels'] = np.asarray(
                [labels[idx] for idx in batch_idxs], dtype=np.int32)
        return batch_data_dict

    def _data_iterator(self, sequence, batch_size, random, labels=None):
        idxs = list(range(len(sequence)))
        if random:
            shuffle(idxs)

        for start_idx in range(0, len(sequence), batch_size):
            end_idx = start_idx + batch_size
            if end_idx > len(sequence):
                end_idx = len(sequence)
            next_batch = self._next_batch(sequence, idxs[start_idx:end_idx], labels)
            yield next_batch

    def test_datas(self, batch_size=16, random=False):
        assert self.initialized, "Dataset is not initialized!"
        return self._data_iterator(self.test_data, batch_size, random, labels=self.test_label)

    def test_data_by_idx(self, start, end):
        assert start >= 0 and end <= len(self.test_data)
        return self._next_batch(self.test_data, range(start, end), labels=self.test_label)

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
