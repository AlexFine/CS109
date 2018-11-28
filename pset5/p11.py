import csv
import numpy as np
import random

# read arrays and populate accordingly
def read(experience_filter):
    matrix = loadCsvData('learningOutcomes.csv')
    skill_matrix = loadCsvData('background.csv')
    arr1 = []
    arr2 = []
    for i in range(len(matrix)):
        if (matrix[i][1] == "activity1") and (skill_matrix[i][1] == experience_filter):
            arr1.append(int(matrix[i][2]))
        elif (matrix[i][1] == "activity2") and (skill_matrix[i][1] == experience_filter):
            arr2.append(int(matrix[i][2]))

    return arr1, arr2

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

# Calculates the sum
def calc_sum(arr1, arr2):
    sum1 = 0
    sum2 = 0
    for i in arr1:
        sum1 += i
    for i in arr2:
        sum2 += i

    print("Arr1 Len: ", len(arr1))
    print("Arr2 Len: ", len(arr2))
    print("Average arr1: ", sum1/len(arr1))
    print("Average arr2: ", sum2/len(arr2))
    return sum1/len(arr1), sum2/len(arr2), len(arr1), len(arr2)

# Calculate p value
def calc_p():
    experience_filter = "less"
    arr1, arr2 = read(experience_filter)
    avg1, avg2, len1, len2 = calc_sum(arr1, arr2)
    # Master distribution
    arr = arr1+arr2
    # Create distribution through bootstrapping
    sim_times = 10000
    sims = []
    p_count = 0
    for i in range(sim_times):
        diff1 = sum_calc(arr, arr_len=len1)
        diff2 = sum_calc(arr, arr_len=len2)
        if (diff1-diff2) > (avg1-avg2):
            p_count += 1
        if i % 1000 == 0:
            print(i)

    # print("Expected Value: ", calc_exp(sims))
    # print("Variance: ", calc_e2(sims) - calc_exp(sims)*calc_exp(sims))
    print("P Value: ", p_count/sim_times)

# Calculate expected value
def calc_exp(arr):
    arr_sum = 0
    for i in arr:
        arr_sum += i

    e_val = arr_sum/len(arr)

    return e_val

# Calculate variance
def calc_e2(arr):
    arr_sum = 0
    for i in arr:
        arr_sum += i*i

    var = arr_sum/len(arr)

    return var
# Calculate sum
def sum_calc(arr, arr_len=500):
    arr_sum = 0
    for i in range(arr_len):
        arr_sum += random.choice(arr)

    return arr_sum/arr_len


calc_p()
