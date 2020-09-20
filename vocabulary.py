import os
import pickle
import tokenize
from collections import Counter
from io import StringIO


class Vocabulary:

    def __init__(self):
        self.word2index = {}
        self.index2word = {}

        self.pad_token, self.pad_id = "§PAD§", 0
        self.oov_token, self.oov_id = "§OOV§", 1
        self.indent_token = "§<indent>§"
        self.dedent_token = "§<dedent>§"
        self.number_token = "§NUM§"

    def preprocess(self, tokentype, tokenval):
        if tokentype == tokenize.NUMBER:
            return self.number_token

        elif tokentype == tokenize.INDENT:
            return self.indent_token

        elif tokentype == tokenize.DEDENT:
            return self.dedent_token

        return tokenval

    def word_count(self, filename):
        data = []
        with open(filename, 'r') as f:
            counter = Counter()
            content = f.read()
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            if not content.endswith('\n'):
                content += '\n'

            tokens = tokenize.generate_tokens(StringIO(content).readline)
            data.append([self.preprocess(tokenType, tokenVal) for tokenType, tokenVal, _, _, _
                         in tokens
                         if tokenType != tokenize.COMMENT and
                         not tokenVal.startswith("'''") and
                         not tokenVal.startswith('"""') and
                         (tokenType == tokenize.DEDENT or tokenVal != "")])
            counter.update(*data)

            return counter

    def build_vocab(self, data_path=None, fileplath='processed.txt', oov_threshold=10, force_include=[], limit=None,
                    python_files=None):

        if python_files is None:
            python_files = [os.path.join(data_path, f.replace('\n', '').replace('\r', '').strip()) for f in
                            open(data_path + fileplath).readlines()]

        if limit is not None:
            python_files = python_files[:limit]

        counters = [self.word_count(filename) for filename in python_files]
        vocab = sum(counters, Counter())
        words = [p for p, v in vocab.items() if v > oov_threshold or p in force_include]

        # words, _ = list(zip(*count_pairs))
        self.word2index = dict(zip(words, range(2, len(words) + 2)))
        self.word2index[self.oov_token] = self.oov_id
        self.word2index[self.pad_token] = self.pad_id

        self.index2word = dict(zip(range(2, len(words) + 2), words))
        self.index2word[self.oov_id] = self.oov_token
        self.index2word[self.pad_id] = self.pad_token
        del words

    def to_word(self, index):
        return self.index2word.get(index, self.oov_token)

    def to_index(self, word):
        return self.word2index.get(word, self.oov_id)

    def save_vocab(self, data_path):
        with open(data_path + 'word2index.pkl', "wb") as f:
            pickle.dump(self.word2index, f)

        with open(data_path + 'index2word.pkl', "wb") as f:
            pickle.dump(self.index2word, f)

    def load_vocab(self, data_path):
        with open(data_path + 'word2index.pkl', "rb") as f:
            self.word2index = pickle.load(f)

        with open(data_path + 'index2word.pkl', "rb") as f:
            self.index2word = pickle.load(f)
