
class Vectorizer:

    UNK = 1
    
    def __init__(self, top_n_vocab=20000):
        self.top_n_vocab = top_n_vocab
        self._idx2vocab = ['PAD', 'UNK']
        self._vocab2idx = {}

    @property
    def idx2vocab(self):
        return self._idx2vocab

    @property
    def vocab2idx(self):
        return self._vocab2idx

    def load(self, path):
        vocabs = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                vocab, count = line.strip().split()
                count = int(count)
                # skip Unk
                if vocab == 'Unk':
                    continue
                vocabs[vocab] = count

        vocabs = sorted(vocabs.items(), key=lambda x: (x[1], x[0]), reverse=True)
        vocabs = vocabs[:self.top_n_vocab]
        self._idx2vocab.extend([vocab for vocab, _ in vocabs])
        self._vocab2idx = {v: i for i, v in enumerate(self._idx2vocab)}

    def encode(self, string):
        words = string.split()
        return [self._vocab2idx[word] if word in self._vocab2idx else self.UNK for word in words]

