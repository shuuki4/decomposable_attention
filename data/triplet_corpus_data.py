import os
import pickle
import random
from random import shuffle
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

        #if len(send_counter & recv_counter) == 0:
        #    return False

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

    # supports on-the-fly negative sampling for training
    def train_datas(self, batch_size=16, random=True,
                    rebuild_negative=True):
        assert self.initialized, "Dataset is not initialized!"
        if rebuild_negative:
            send_recvs = [(send, recv_pos) for send, recv_pos, _ in self.train_data]
            negative_recvs = self._sample_negative(send_recvs)
            self.train_data = [(send, recv_pos, recv_neg) for (send, recv_pos), recv_neg
                               in zip(send_recvs, negative_recvs)]
        return self._data_iterator(self.train_data, batch_size, random)

    def _sample_negative(self, send_recvs):
        mapper = defaultdict(list)
        recvs = [recv for _, recv in send_recvs]
        for recv_idx, recv in enumerate(recvs):
            for token_idx in recv:
                mapper[token_idx].append(recv_idx)

        mapper = dict(mapper)
        negative_recvs = []
        for send, recv_pos in send_recvs:
            cnt = 0
            while True:
                try:
                    token_idx = random.choice(send)
                    # pick hardest
                    #lengths = []
                    #for idx in send:
                    #    if idx in mapper:
                    #        length = len(mapper[idx])
                    #        if length > 10:
                    #            lengths.append((len(mapper[idx]), idx))
                    #token_idx = min(lengths)[1]
                    negative_recv_idx = random.choice(mapper[token_idx])
                    recv_neg = recvs[negative_recv_idx]
                except (KeyError, IndexError):
                    continue
                if recv_pos != recv_neg:
                    break
                cnt += 1
                if cnt > 100:
                    print(cnt)
            negative_recvs.append(recv_neg)

        return negative_recvs

    def _build_negative(self, all_data, train_val_ratio=0.9):
        random.shuffle(all_data)
        train_val_cut = int(train_val_ratio * len(all_data))
        train_data = all_data[:train_val_cut]
        val_data = all_data[train_val_cut:]

        train_negative_recvs = self._sample_negative(train_data)
        val_negative_recvs = self._sample_negative(val_data)

        train_data = [(send, recv_pos, recv_neg) for (send, recv_pos), recv_neg
                      in zip(train_data, train_negative_recvs)]
        val_data = [(send, recv_pos, recv_neg) for (send, recv_pos), recv_neg
                    in zip(val_data, val_negative_recvs)]

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
        self.num_category = 0

    def load(self, data_path=None, vocab_path=None, test_data_path=None):
        assert data_path is not None and vocab_path is not None

        self.vectorizer.load(vocab_path)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.train_data = data['train']
            self.val_data = data['val']

        if test_data_path:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                labels = []
                for line in f:
                    sent1, sent2_pos, label = line.split('\t')
                    self.test_data.append(
                        (self.vectorizer.encode(sent1),
                         self.vectorizer.encode(sent2_pos),
                         self.vectorizer.encode('UNK')))
                    labels.append(int(label))
            self.test_label = labels

        self.symbols = self.vectorizer.idx2vocab
        self.num_category = 0

