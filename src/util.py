import os, collections
import numpy as np

def read_data(path):
    sentences = []
    words = []
    label_vectors = []
    file = open(path)
    lines = file.readlines()
    
    for line in lines : 
        line_split = line.split('\t')
        number_of_words = len(line_split[0].split())
        cwi_index = int(line_split[2])
        sentences.append(line_split[0])
        words.append(line_split[1])
        label = np.zeros(number_of_words)
        label[cwi_index] = 1
        label_vectors.append(label)
        
    return sentences, words, label_vectors

def words_to_index(sentences, words):
    vocab = collections.defaultdict(lambda: len(vocab))
    vocab['<eos>'] = 0
    
    int_sentences = []
    int_words = []
    for i, sentence  in enumerate(sentences):
        sentence_words = sentence.split()
        int_sentences.append([vocab[token.lower()] for token in sentence_words])
        int_words.append(vocab[words[i].lower()])
            
    return int_sentences, int_words

def words_to_embeddings(sentences, words):
    # TODO for neural networks ...
    pass