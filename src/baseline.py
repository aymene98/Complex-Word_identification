import numpy as np
from nltk.probability import FreqDist
import random


class Random_baseline(object):
    
    def fit(self, sentences, words, label_vectors):
        pass
    
    def predict(self, sentences, words):
        return np.random.randint(2, size=len(words), dtype=int)
    
    def save(self, sentences, words, name):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')
        
        
class Frequency_baseline(object):
    
    def fit(self, sentences, words, label_vectors):
        words = []
        for sentence in sentences:
            words += sentence
            
        temp_dict = FreqDist(words)
        self.words_dict = {word: temp_dict.freq(word) for word in temp_dict}

    
    def predict(self, sentences, words, threshold=0.1):
        predictions = np.zeros(len(words))
        for i, word in enumerate(words):
            if word not in self.words_dict.keys():
                predictions[i] = random.randint(0, 1)
            elif self.words_dict[word] < threshold:
                predictions[i] = 1
        return predictions
    
    def save(self, sentences, words, name, threshold=0.1):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')
        
        