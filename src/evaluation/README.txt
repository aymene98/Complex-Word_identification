1) Introduction:
	This package contains the evaluation script for Task 11: Complex Word Identification of SemEval 2016.

2) Content:
	- README.txt: This file.
	- evaluate_system.py: System evaluation script in Python.

3) Running:
	The command line that runs the evaluation script is:
	
		python evaluate_system.py [-h] --gold GOLD --pred PRED
		
	If you use the "-h" option, you will get detailed instructions on the parameters required.
	The "--gold" parameter must be a dataset with gold-standard labels in the format provided by the task's organizers.
	The "--pred" parameter must be the file containing the predicted labels.