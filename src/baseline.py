import numpy as np

class Random_baseline(object):
    
    def fit(self, sentences, words, label_vectors):
        pass
    
    def predict(self, sentences, words):
        return np.random.randint(2, size=len(words), dtype=int)
    
    def save(self, sentences, words, name):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')
        
        
class Frequency_baseline(object):
    
    def fit(self, sentences, words, label_vectors):
        pass
    
    def predict(self, sentences, words):
        return np.random.randint(2, size=len(words), dtype=int)
    
    def save(self, sentences, words, name):
        np.savetxt(name, self.predict(sentences, words), delimiter='\n', fmt='%1.0d')
        
        