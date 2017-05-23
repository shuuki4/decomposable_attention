import os
import pickle
import random
from collections import Counter, defaultdict

from data.pair_sequence_data import PairSequenceData
from data.corpus.vectorizer import Vectorizer
from util import log


class PairCorpusData(PairSequenceData):
    """
    Data class for corpus
    """

    def __init__(self, max_length=10):
        super(PairCorpusData, self).__init__()
        self.max_length = max_length
        self.vectorizer = Vectorizer()

    def _quality_check(self, send, recv):
        send_counter = Counter(send)
        recv_counter = Counter(recv)

        if self.vectorizer.UNK in send_counter:
            return False
        n_idx = self.vectorizer.vocab2idx['N']
        if send_counter.get(n_idx, 0) >= 3 or recv_counter.get(n_idx, 0) >= 3:
            return False

        if len(send_counter & recv_counter) == 0:
            return False

        return True

    def _read_paired_corpus(self, corpus_path, min_length=4, quality_check=True):
        data = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                send, recv = line.strip().split('\t')
                send = self.vectorizer.encode(send)
                recv = self.vectorizer.encode(recv)

                if quality_check:
                    if len(send) > self.max_length or len(recv) > self.max_length \
                            or len(send) < min_length or len(recv) < min_length:
                        continue
                    if not self._quality_check(send, recv):
                        continue

                data.append((send, recv, 1))
        return data

    def _build_negative(self, all_data, train_val_ratio=0.9):
        random.shuffle(all_data)
        train_val_cut = int(train_val_ratio * len(all_data))
        train_pos_data = all_data[:train_val_cut]
        val_pos_data = all_data[train_val_cut:]

        train_neg_data, val_neg_data = [], []
        train_recvs = [recv for _, recv, _ in train_pos_data]
        val_recvs = [recv for _, recv, _ in val_pos_data]

        # build negative data from random recv
        # that has common vocab with send,
        # but most 'unimportant' token
        train_recv_mapper = defaultdict(lambda: list())
        val_recv_mapper = defaultdict(lambda: list())
        for recv_idx, recv in enumerate(train_recvs):
            for token_idx in recv:
                train_recv_mapper[token_idx].append(recv_idx)
        for recv_idx, recv in enumerate(val_recvs):
            for token_idx in recv:
                val_recv_mapper[token_idx].append(recv_idx)

        for send, recv, _ in train_pos_data:
            max_token_idx = max((len(train_recv_mapper[idx]), idx)
                                for idx in send)[1]
            negative_recv_idx = random.choice(
                train_recv_mapper[max_token_idx])
            train_neg_data.append((send, train_recvs[negative_recv_idx], 0))
        for send, recv, _ in val_pos_data:
            max_token_idx = max((len(val_recv_mapper[idx]), idx)
                                for idx in send)[1]
            negative_recv_idx = random.choice(
                val_recv_mapper[max_token_idx])
            val_neg_data.append((send, val_recvs[negative_recv_idx], 0))

        return train_pos_data, train_neg_data, \
               val_pos_data, val_neg_data

    def build(self, corpus_path=None, vocab_path=None,
              save_path=None, min_length=4):
        assert save_path is not None and corpus_path is not None \
            and vocab_path is not None

        self.vectorizer.load(vocab_path)
        log.infov('Start Loading Data...')
        potential_data = self._read_paired_corpus(corpus_path,
                                                  min_length,
                                                  quality_check=True)
        log.infov('Loaded {} pair corpus data!'.format(len(potential_data)))

        log.infov('Building dataset...')
        train_pos_data, train_neg_data, val_pos_data, val_neg_data = \
            self._build_negative(potential_data)
        self.train_data = train_pos_data + train_neg_data
        self.val_data = val_pos_data + val_neg_data

        log.infov('Saving...')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'train': self.train_data,
                'val': self.val_data
            }, f)

        self.symbols = self.vectorizer.idx2vocab
        self.num_category = 2

    def load(self, data_path=None, vocab_path=None, test_data_path=None):
        assert data_path is not None and vocab_path is not None

        self.vectorizer.load(vocab_path)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.train_data = data['train']
            self.val_data = data['val']

        if test_data_path is not None:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    send, recv, label = line.split('\t')
                    self.test_data.append(
                        (self.vectorizer.encode(send),
                         self.vectorizer.encode(recv),
                         float(label)))

        self.symbols = self.vectorizer.idx2vocab
        self.num_category = 2

