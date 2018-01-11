import numpy as np


class VocabBuilder():
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.vocabulary = set(f.read())
            f.close()
            self.charToInd = dict()
            self.indToChar = dict()
            cnt = 0
            for i in self.vocabulary:
                self.charToInd[i] = cnt
                self.indToChar[cnt] = i
                cnt += 1
            self.identity = np.eye(len(self.vocabulary))

    def getVocabCharToInd(self):
        return self.charToInd

    def getVocabIndToChar(self):
        return self.indToChar

    def getOneHot(self, cList):
        inds = [self.charToInd[i] for i in cList]
        return self.identity[inds]

    def get_vocab_size(self):
        return len(self.vocabulary)


class DataBuilder():
    def get_ptb_dataset(self, dataset='train'):
        fn = './ptb/ptb.{}.txt'
        for line in open(fn.format(dataset), encoding="utf-8"):
            for word in line.split():
                yield word
            # Add token to the end of the line
            # Equivalent to <eos> in:
            # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
            yield '<eos>'

    def get_data(self, word_vocab, dataset='train'):
        X = []
        for i in self.get_ptb_dataset(dataset):
            X.append(word_vocab.encode(i))
        return np.array(X, dtype=np.int32)

    def build_vocabulary(self):
        v_ = WordVoab()
        for i in self.get_ptb_dataset():
            v_.add_word(i)
        return v_


class WordVoab():
    def __init__(self):
        self.word2ind = dict()
        self.ind2word = dict()
        self.unknown = '<unk>'
        self.word_freq = dict()
        self.add_word(self.unknown)

    def add_word(self, word):
        if word not in self.word2ind:
            index = len(self.word2ind)
            self.word2ind[word] = index
            self.ind2word[index] = word
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def encode(self, word):
        if word not in self.word2ind:
            word = self.unknown
        return self.word2ind[word]

    def decode(self, ind):
        return self.ind2word[ind]

'''
if __name__ == '__main__':
    db = DataBuilder()
    v = db.build_vocabulary()
    print(np.size(db.get_data(v,dataset='valid')))
'''