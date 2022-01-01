import numpy as np
from nltk.probability import FreqDist
import random
from nltk import RegexpTokenizer
import regex
from itertools import islice
import collections
from util import words_to_index

class Random_baseline(object):
    
    def fit(self, sentences, words, label_vectors):
        pass
    
    def predict(self, sentences, words):
        print(type(np.random.randint(2, size=len(words), dtype=int)))
        return np.random.randint(2, size=len(words), dtype=int)
    
    def save(self, sentences, words, name):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')
        
        
class Frequency_baseline(object):
    
    def fit(self, sentences, words, label_vectors):
        sentences, words = words_to_index(sentences, words)
        words = []
        for sentence in sentences:
            words += sentence
            
        temp_dict = FreqDist(words)
        self.words_dict = {word: temp_dict.freq(word) for word in temp_dict}

    
    def predict(self, sentences, words, threshold=0.1):
        predictions = np.zeros(len(words))
        sentences, words = words_to_index(sentences, words)
        for i, word in enumerate(words):
            if word not in self.words_dict.keys():
                predictions[i] = random.randint(0, 1)
            elif self.words_dict[word] < threshold:
                predictions[i] = 1
        return predictions
    
    def save(self, sentences, words, name, threshold=0.1):
        np.savetxt(name, self.predict(sentences, words, threshold), delimiter='\n', fmt='%1.0d')

     
class All_complex(object):

    def fit(self, sentences, words, label_vectors):
        pass

    def predict(self, sentences, words):
        return np.ones(len(words), dtype=int)

    def save(self, sentences, words, name):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')


class All_simple(object):

    def fit(self, sentences, words, label_vectors):
        pass

    def predict(self, sentences, words):
        return np.zeros(len(words), dtype=int)

    def save(self, sentences, words, name):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')


class Simple_wikipedia(object):
    
    def fit(self, sentences, words, label_vectors):
        #wget https://github.com/LGDoor/Dump-of-Simple-English-Wiki/raw/master/corpus.tgz
        with open('../data/freq_corpora/simple_wikipedia.txt', encoding="utf-8") as fp:
            text = fp.read()
            toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
            words = toknizer.tokenize(text)
            filtered_words = [word.lower() for word in words if regex.search(r'(\p{Ll}|\d)', word.lower())]
            self.freq = collections.defaultdict(int)
            for word in filtered_words:
                self.freq[word] += 1
    
    def predict(self, sentences, words, threshold = 500):
        predictions = np.zeros(len(words))
        for i, word in enumerate(words):
            print(sentences[1])
            if word not in self.freq.keys():
                predictions[i] = 1
            elif self.freq[word] < threshold:
                predictions[i] = 1
        return predictions
    
    def save(self, sentences, words, name, threshold):
        np.savetxt(name, self.predict(sentences, words, threshold), delimiter='\n', fmt='%1.0d')