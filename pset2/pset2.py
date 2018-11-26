import numpy as np
# #
# # p = 0.7
# #
# # def rand():
# #     r1 = np.random.choice([0, 1], p=[1-p, p])
# #     while True:
# #         r2 = np.random.choice([0, 1], p=[1-p, p])
# #         if (r1 != r2):
# #             break
# #         r1 = r2
# #     return r2
# #
# # def test():
# #     num_1 = 0
# #     num_0 = 0
# #     for i in range(10000):
# #         temp = rand()
# #         if temp == 0:
# #             num_0 += 1
# #         else:
# #             num_1 += 1
# #
# #     print(num_1)
# #     print(num_0)
# #
#
#
import csv

def main():
    data = loadCsvData('bats.csv')
    data = np.asarray(data)
    # Calculate basic probabilities
    # P_T(data)
    # for i in range(5):
    #     print(P_T(data, i=i))
    # Transform Matrix
    # Get sub matrix with just Ebola possitive
    T_Only = Transform_Matrix(data, i=5)
    for i in range(6):
        for j in range(i+1,5):
            print("I,J", i+1, j+1)
            num = P_T(T_Only, i=i, j=j)
            print(round(num/30079, 4))

    arr = [.705, 0.3021, 0.9711, 0.9874, 0.97855]
    for i in range(5):
        for j in range(i+1,5):
            print("I,J", i+1, j+1)
            print(round(arr[i]*arr[j], 4))
    # for i in range(5):
    #     print(P_T(G34_Only, i=i))

def P_T(matrix, i=0, j=2):
    count = 0
    for row in matrix:
        if (row[i] == row[j]):
            count += 1
    print(count)
    return count


def Transform_Matrix(matrix, i=5):
    ret_matrix = []
    count = 0
    for row in matrix:
        if row[i] == 'True':
            ret_matrix.append(row)
            count += 1
        else:
            # ret_matrix = np.delete(ret_matrix, (count), axis=0)
            pass

    ret_matrix = np.asarray(ret_matrix)
    print(ret_matrix.shape)
    return ret_matrix


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

def printData(matrix):
	for row in matrix:
		print (row)



main()

#
import csv

# The main method
def main():
    prior = loadCsvData('prior.csv')
    conditional = loadCsvData('conditional.csv')
    calculateNewMatrix(prior, conditional)

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
                doubleRow.append(float(value))
            # store the row into our matrix
            matrix.append(doubleRow)
    return matrix

# Prints out a 2d array
def printData(matrix):
    for row in matrix:
        print(row)

def calculateNewMatrix(prior, conditional):
    newMatrix = []
    count = 0
    for row in range(len(prior)):
        # cast each value to a float
        doubleRow = []
        for col in range(len(prior[0])):
            doubleRow.append(prior[row][col]*conditional[row][col])
            count += prior[row][col]*conditional[row][col]
        newMatrix.append(doubleRow)

    sum_num = 0
    for row in range(len(prior)):
        # cast each value to a float
        for col in range(len(prior[0])):
            newMatrix[row][col] = newMatrix[row][col]/count
            sum_num += newMatrix[row][col]

    print("newMatrix: ", newMatrix)
    print("Total Sum: ", sum_num)
# This if statement passes if this
# was the file that was executed
if __name__ == '__main__':
    main()
