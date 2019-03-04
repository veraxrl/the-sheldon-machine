import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

'''
Read data from file and return a list of list
every item in the list is a list of original text, response text and label
label = 0 -> not sarcasm
label = 1 -> sarcasm
'''

'''
Read data from discussion forum data
'''
def readDiscussionForum(file = "dicussion-forum-data.csv"):
	with open(file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		data = []
		for row in csv_reader:
			if line_count == 0:
				print('Column names are {}'.format(", ".join(row)))
				line_count += 1
			else:
				data.append([row[3], row[4], 1 if row[2] == 'sarc' else 0])
				line_count += 1
		print('Processed {} lines.'.format(line_count))
		return data

def readReddit(file = "reddit/train-balanced-sarcasm.csv"):
	# Import and data analysis
	print ("-"*80)
	print("Importing reddit training data")
	print ("-"*80)
	df = pd.read_csv(file)
	print(df.shape)
	print(df['label'].value_counts())
	# Spliting training and validation sets
	train_comment, valid_comment, train_label, valid_label = train_test_split(df['comment'], df['label'], train_size=0.8, test_size=0.2)

if __name__ == '__main__':
    readReddit()