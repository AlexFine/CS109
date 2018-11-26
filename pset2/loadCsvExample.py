# File: Load CSV Example
# ----------------------
# This is some sample code. Use it in any way
# that you would like. It is meant to both give
# you a head start on the assignments and show you
# what some python code looks like :-). - Piech.

# A useful library for reading files
# with "comma seperated values".
import csv

# The main method
def main():
	data = loadCsvData('prior.csv')
	printData(data)

# Reads a files into a 2d array. There are
# other ways of doing this (do check out)
# numpy. But this shows
def loadCsvData(fileName):
	matrix = []
	# open a file
	with open(fileName) as f:
		reader = csv.reader(f)

		# loop over each row in the file
		for row in reader:

			# cast each value to a float
			doubleRow = []
			for value in row:
				doubleRow.append(str(value))

			# store the row into our matrix
			matrix.append(doubleRow)
	return matrix

# Prints out a 2d array
def printData(matrix):
	for row in matrix:
		print row

# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
	main()
