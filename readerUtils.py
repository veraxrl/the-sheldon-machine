import csv

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
