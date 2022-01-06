from baseline import Random_baseline, Frequency_baseline, All_simple, All_complex, Simple_wikipedia
from util import read_data, word_context
from evaluate_system import evaluate
import os

train_data_file = './data/cwi_training/cwi_training.txt'
test_data_file = './data/cwi_testing_annotated/cwi_testing_annotated.txt'
output_file = './output/test.txt'

train_sentences, train_words, train_label, x = read_data(train_data_file)
pre, post = word_context(train_sentences, x, context_size=2)
print(train_sentences[0])
print(pre[0], train_words[0], post[0])
#test_sentences, test_words, test_label = read_data(test_data_file)
"""
model = Frequency_baseline()
for i in range(0,10):
    print(i)
    model.fit(train_sentences, train_words, train_label)
    model.save(test_sentences, test_words, output_file, threshold=10**-i)
    evaluate(test_data_file, output_file)"""
    


