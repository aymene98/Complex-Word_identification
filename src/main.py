from baseline import Random_baseline
from util import read_data, words_to_index
from evaluate_system import evaluate
import os

train_data_file = 'data/cwi_training/cwi_training.txt'
test_data_file = 'data/cwi_testing_annotated/cwi_testing_annotated.txt'
output_file = 'output/test.txt'

train_sentences, train_words, train_label = read_data(train_data_file)
train_sentences, train_words = words_to_index(train_sentences, train_words)

test_sentences, test_words, test_label = read_data(test_data_file)
test_sentences, test_words = words_to_index(test_sentences, test_words)

model = Random_baseline()

model.fit(train_sentences, train_words, train_label)
prediction = model.predict(test_sentences, test_words)
model.save(test_sentences, test_words, output_file)

evaluate(test_data_file, output_file)

