import os
import pickle
import random
from collections import Counter, defaultdict

from data.triplet_sequence_data import TripletSequenceData
from data.corpus.vectorizer import Vectorizer
from util import log


class TripletCorpusData(TripletSequenceData):
    """
    Data class for corpus
    """

    def __init__(self, max_length=10):
        super(TripletCorpusData, self).__init__()
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

                data.append((send, recv))
        return data

    def _build_negative(self, all_data, train_val_ratio=0.9):
        random.shuffle(all_data)
        train_val_cut = int(train_val_ratio * len(all_data))
        train_data = all_data[:train_val_cut]
        val_data = all_data[train_val_cut:]

        train_recvs = [recv for _, recv in train_data]
        val_recvs = [recv for _, recv in val_data]

        # build negative data from random recv
        # that has common vocab with send
        train_recv_mapper = defaultdict(lambda: list())
        val_recv_mapper = defaultdict(lambda: list())
        for recv_idx, recv in enumerate(train_recvs):
            for token_idx in recv:
                train_recv_mapper[token_idx].append(recv_idx)
        for recv_idx, recv in enumerate(val_recvs):
            for token_idx in recv:
                val_recv_mapper[token_idx].append(recv_idx)

        for i, (send, recv_pos) in enumerate(train_data):
            while True:
                try:
                    token_idx = random.choice(send)
                    negative_recv_idx = random.choice(
                        train_recv_mapper[token_idx])
                    recv_neg = train_recvs[negative_recv_idx]
                except IndexError:
                    continue
                if recv_pos != recv_neg:
                    break
            train_data[i] = (send, recv_pos, recv_neg)

        for i, (send, recv_pos) in enumerate(val_data):
            while True:
                try:
                    token_idx = random.choice(send)
                    negative_recv_idx = random.choice(
                        val_recv_mapper[token_idx])
                    recv_neg = val_recvs[negative_recv_idx]
                except IndexError:
                    continue
                if recv_pos != recv_neg:
                    break
            val_data[i] = (send, recv_pos, recv_neg)

        return train_data, val_data

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
        train_data, val_data = self._build_negative(potential_data)
        self.train_data = train_data
        self.val_data = val_data

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

        if test_data_path:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sent1, sent2_pos, sent2_neg = line.split('\t')
                    self.test_data.append(
                        (self.vectorizer.encode(sent1),
                         self.vectorizer.encode(sent2_pos),
                         self.vectorizer.encode(sent2_neg)))

        self.symbols = self.vectorizer.idx2vocab
        self.num_category = 2

