from __future__ import division, print_function
import sys
import argparse

"""
	1) Introduction:
	This package contains the evaluation script for Task 11: Complex Word Identification of SemEval 2016. 2) Content:
	- README.txt: This file.
	- evaluate_system.py: System evaluation script in Python. 3) Running:
	The command line that runs the evaluation script is:
	
		python evaluate_system.py [-h] --gold GOLD --pred PRED
		
	If you use the "-h" option, you will get detailed instructions on the parameters required.
	The "--gold" parameter must be a dataset with gold-standard labels in the format provided by the task's organizers.
	The "--pred" parameter must be the file containing the predicted labels.
"""

def evaluateIdentifier(gold, pred):
	"""
	Performs an intrinsic evaluation of a Complex Word Identification approach.
	@param gold: A vector containing gold-standard labels.
	@param pred: A vector containing predicted labels.
	@return: Accuracy, Recall and F-1.
	"""
	
	#Initialize variables:
	accuracyc = 0
	accuracyt = 0
	recallc = 0
	recallt = 0
	
	#Calculate measures:
	for gold_label, predicted_label in zip(gold, pred):
		if gold_label==predicted_label:
			accuracyc += 1
			if gold_label==1:
				recallc += 1
		if gold_label==1:
			recallt += 1
		accuracyt += 1
	
	accuracy = accuracyc / accuracyt
	recall = recallc / recallt
	fmean = 0
	
	try:
		fmean = 2 * (accuracy * recall) / (accuracy + recall)
	except ZeroDivisionError:
		fmean = 0
	
	#Return measures:
	return accuracy, recall, fmean


def evaluate(gold_file, pred_file):
    gold = [int(line.strip().split('\t')[3]) for line in open(gold_file)]
    pred = [int(line.strip()) for line in open(pred_file)]
    p, r, f = evaluateIdentifier(gold, pred)
    print('Accuracy: %.4f' % p)
    print('Recall: %.4f' % r)
    print('F1: %.4f' % f)
    return p, r, f
