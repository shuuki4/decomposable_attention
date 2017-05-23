import numpy as np
from data.base_sequence_data import BaseSequenceData


class TripletSequenceData(BaseSequenceData):
    """
    Pair sequence data class interface for
    decomposable attention ranking model
    """
    def __init__(self):
        """
        Data format should be [(sequence1, sequence2, label)]
        """
        super(TripletSequenceData, self).__init__()

    def _next_batch(self, data, batch_idxs):
        """
        Generate next batch.
        :param data: data list to process
        :param batch_idxs: idxs to process
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
        return batch_data_dict

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
