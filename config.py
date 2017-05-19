
class Config:

    def __init__(self, num_words, num_category,
                 learning_rate=0.0001, batch_size=128,
                 word_embedding_dim=100, rnn_state_size=150):

        self.config = dict()

        # dataset related config
        self.config['data'] = data_config = dict()
        data_config['num_word'] = num_words
        data_config['num_category'] = num_category

        # training config
        self.config['training'] = training_config = dict()
        training_config['learning_rate'] = learning_rate
        training_config['batch_size'] = batch_size

        # word config
        self.config['word'] = word_config = dict()
        word_config['embedding_dim'] = word_embedding_dim

        # rnn config
        self.config['rnn'] = encoder_config = dict()
        encoder_config['state_size'] = rnn_state_size

    def __getitem__(self, keys):
        if type(keys) == str:
            try:
                return self.config[keys]
            except KeyError as e:
                raise KeyError('Wrong key {} for config'.format(keys))

        elif type(keys) == list or type(keys) == tuple:
            assert len(keys) == 2
            try:
                return self.config[keys[0]][keys[1]]
            except KeyError as e:
                raise KeyError('Wrong key {} for config'.format(keys))
