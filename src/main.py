from baseline import Random_baseline, Frequency_baseline, All_simple, All_complex, Simple_wikipedia
from util import read_data
from evaluate_system import evaluate
import os

train_data_file = '../data/cwi_training/cwi_training.txt'
test_data_file = '../data/cwi_testing_annotated/cwi_testing_annotated.txt'
output_file = '../output/test.txt'

train_sentences, train_words, train_label = read_data(train_data_file)
test_sentences, test_words, test_label = read_data(test_data_file)

model = Simple_wikipedia()

model.fit(train_sentences, train_words, train_label)
model.save(test_sentences, test_words, output_file)

evaluate(test_data_file, output_file)

