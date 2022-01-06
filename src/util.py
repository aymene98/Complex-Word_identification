import os, collections
import numpy as np
from transformers import DistilBertTokenizer

def read_data(path):
    sentences = []
    words = []
    labels = []
    complex_word_index = []
    file = open(path)
    lines = file.readlines()
    
    for line in lines : 
        line_split = line.split('\t')
        sentences.append(line_split[0]) # sentence
        words.append(line_split[1]) # word
        cwi_index = int(line_split[2])
        complex_word_index.append(cwi_index) # word index needed for context
        labels.append(line_split[3]) # label
        
    return sentences, words, labels, complex_word_index

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

def word_context(sentences, indecies, context_size=2):
    pre_context, post_context = [], []
    for i, sentence in enumerate(sentences):
        #pre_word_context, post_word_context = [], []
        word_index = indecies[i]
        upper_bound = word_index + context_size + 1
        lower_bound = word_index - context_size - 1
        if lower_bound <= 0:
            lower_bound = None
        pre_context.append(sentence.split()[word_index-1 : lower_bound: -1][::-1])
        post_context.append(sentence.split()[word_index+1 : upper_bound])
        
    return pre_context, post_context

def tokenize(sentences):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized = []
    for context in sentences:
        tokens = []
        for word in context:
            tokenised_word = tokenizer.encode(word, add_special_tokens=True)
            tokens += tokenised_word
        tokenized.append(tokens)
    return tokenized

def build_attention_map(pre_tokens, word_tokens, max_size):
    # pre_tokens, word_tokens are lists for each word ...
    attention_maps = []
    for i in range(len(pre_tokens)):
        attention_map = [0]*max_size
        max = len(pre_tokens[i])+len(word_tokens[i])
        attention_map[len(pre_tokens[i]): max] = [1]*len(word_tokens[i])
        attention_maps.append(attention_maps)
            
    return attention_maps
        